/*
Copyright (c) 2015-2020, Intel Corporation
Copyright (c) 2019-2020, University of Utah

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

//////////////////////////////////////////////////////////////////////////////

#include "program.h"
#include "data.h"
#include "emit_policy.h"
#include "stmt.h"
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>

using namespace yarpgen;

ProgramGenerator::ProgramGenerator() : hash_seed(0) {
    // Generate the general structure of the test
    auto gen_ctx = std::make_shared<GenCtx>();
    new_test = ScopeStmt::generateStructure(gen_ctx);

    // Prepare to generate some math inside the structure
    ext_inp_sym_tbl = std::make_shared<SymbolTable>();
    ext_out_sym_tbl = std::make_shared<SymbolTable>();
    auto pop_ctx = std::make_shared<PopulateCtx>();
    auto gen_pol = pop_ctx->getGenPolicy();

    // Create some number of ScalarVariables that we will use to provide input
    // data to the test program
    size_t inp_vars_num = rand_val_gen->getRandValue(gen_pol->min_inp_vars_num,
                                                     gen_pol->max_inp_vars_num);
    for (size_t i = 0; i < inp_vars_num; ++i) {
        auto new_var = ScalarVar::create(pop_ctx);
        ext_inp_sym_tbl->addVar(new_var);
        ext_inp_sym_tbl->addVarExpr(
            std::make_shared<ScalarVarUseExpr>(new_var));
    }

    pop_ctx->setExtInpSymTable(ext_inp_sym_tbl);
    pop_ctx->setExtOutSymTable(ext_out_sym_tbl);

    new_test->populate(pop_ctx);

    // Create a special variable that we use to hide the information from
    // compiler
    auto zero_var = std::make_shared<ScalarVar>(
        "zero", IntegralType::init(IntTypeID::INT),
        IRValue(IntTypeID::INT, IRValue::AbsValue{false, 0}));
    zero_var->setIsDead(false);
    ext_inp_sym_tbl->addVar(zero_var);

    // We have to do it after the generation, when we know what arrays are
    // actually used
    if (Options::getInstance().useHeapViews())
        layoutBuffers(gen_pol);
}

// Distance between the start of the "target" view and the start of the view
// that should overlap it
static int64_t pickAliasDist(std::shared_ptr<GenPolicy> gen_pol,
                             const std::shared_ptr<Data> &target,
                             const std::shared_ptr<Data> &view) {
    auto target_len = static_cast<int64_t>(target->getElemsNum());
    auto view_len = static_cast<int64_t>(view->getElemsNum());
    // The views overlap iff the distance is in [min_dist, max_dist]
    int64_t min_dist = -(view_len - 1);
    int64_t max_dist = target_len - 1;
    // Two scalars
    if (min_dist == max_dist)
        return 0;

    int64_t dist = 0;
    switch (rand_val_gen->getRandId(gen_pol->alias_dist_distr)) {
        case AliasDistKind::ZERO:
            return 0;
        case AliasDistKind::SMALL:
            dist = static_cast<int64_t>(
                rand_val_gen->getRandId(gen_pol->alias_small_dist_distr));
            break;
        case AliasDistKind::STRIDE: {
            std::vector<size_t> strides;
            for (auto &data : {target, view})
                if (data->isArray()) {
                    auto arr_strides =
                        std::static_pointer_cast<Array>(data)->getStrides();
                    strides.insert(strides.end(), arr_strides.begin(),
                                   arr_strides.end());
                }
            if (!strides.empty())
                dist = static_cast<int64_t>(rand_val_gen->getRandElem(strides));
            break;
        }
        case AliasDistKind::RANDOM:
            return rand_val_gen->getRandValue(min_dist, max_dist);
        case AliasDistKind::MAX_ALIAS_DIST_KIND:
            ERROR("Bad AliasDistKind");
    }
    if (rand_val_gen->getRandValue(false, true))
        dist = -dist;
    if (dist < min_dist || dist > max_dist)
        dist = -dist;
    if (dist < min_dist || dist > max_dist)
        dist = rand_val_gen->getRandValue(min_dist, max_dist);
    return dist;
}

static bool rangesOverlap(int64_t a_begin, int64_t a_end, int64_t b_begin,
                          int64_t b_end) {
    return a_begin < b_end && b_begin < a_end;
}

// Two outputs may overlap only if the result doesn't depend on the order of
// the stores to them, or the compiler has to keep that order
static bool outputsOverlapAllowed(const std::shared_ptr<Data> &a,
                                  const std::shared_ptr<Data> &b) {
    if (!a->getOverlapAllowed() || !b->getOverlapAllowed())
        return false;
    // "#pragma omp simd" lets the compiler reorder stores of different
    // iterations
    return a->getOMPSimdRegion() == 0 ||
           a->getOMPSimdRegion() != b->getOMPSimdRegion();
}

void ProgramGenerator::layoutBuffers(std::shared_ptr<GenPolicy> gen_pol) {
    Options &options = Options::getInstance();
    auto is_live = [&options](const std::shared_ptr<Data> &data) {
        return options.getAllowDeadData() || !data->getIsDead();
    };
    // Scalar outputs of "#pragma omp simd" loops are named in the reduction
    // clause, so they can't be pointers
    auto can_be_view = [&is_live](const std::shared_ptr<Data> &data) {
        if (data->isScalarVar() && data->getOMPSimdRegion() != 0)
            return false;
        return is_live(data);
    };

    // The views that may overlap something: outputs and opaque inputs
    std::vector<std::shared_ptr<Data>> nodes;
    for (auto &var : ext_out_sym_tbl->getVars())
        nodes.push_back(var);
    for (auto &array : ext_out_sym_tbl->getArrays())
        nodes.push_back(array);
    size_t outs_num = nodes.size();
    for (auto &array : ext_inp_sym_tbl->getArrays())
        if (array->isOpaque())
            nodes.push_back(array);
    std::unordered_map<Data *, size_t> node_idx;
    for (size_t i = 0; i < nodes.size(); ++i)
        node_idx[nodes.at(i).get()] = i;
    auto is_out = [&node_idx, outs_num](const std::shared_ptr<Data> &data) {
        auto find_res = node_idx.find(data.get());
        return find_res != node_idx.end() && find_res->second < outs_num;
    };

    auto overlap_allowed = [&is_out](const std::shared_ptr<Data> &a,
                                     const std::shared_ptr<Data> &b) {
        bool a_out = is_out(a);
        bool b_out = is_out(b);
        // We need to know the values of regular inputs. Opaque inputs are
        // read only by the expressions that are well defined for any values.
        if ((!a_out && !a->isOpaque()) || (!b_out && !b->isOpaque()))
            return false;
        if (a_out && b_out)
            return outputsOverlapAllowed(a, b);
        // Inputs are never written
        if (!a_out && !b_out)
            return true;
        auto &inp = a_out ? b : a;
        auto &out = a_out ? a : b;
        // "#pragma omp simd" promises that there are no loop-carried
        // dependencies
        return out->getOMPSimdRegion() == 0 ||
               !inp->isReadInOMPSimdRegion(out->getOMPSimdRegion());
    };

    // A block is a set of views with fixed offsets relative to each other.
    // Each block goes into a buffer as a whole.
    struct Block {
        std::vector<std::pair<std::shared_ptr<Data>, size_t>> views;
        size_t size = 0;
    };
    std::vector<Block> blocks;
    std::vector<bool> in_group(nodes.size(), false);

    // Alias hints form trees: each output points to an older output or to an
    // opaque input. Each tree becomes a block, where the views overlap their
    // targets.
    auto valid_target = [&](size_t i) -> int64_t {
        auto &view = nodes.at(i);
        auto target = view->getAliasTarget();
        if (!target || !can_be_view(view))
            return -1;
        auto find_res = node_idx.find(target.get());
        if (find_res == node_idx.end() || !can_be_view(target) ||
            !overlap_allowed(view, target))
            return -1;
        return static_cast<int64_t>(find_res->second);
    };
    std::vector<std::vector<size_t>> children(nodes.size());
    std::vector<bool> has_target(nodes.size(), false);
    for (size_t i = 0; i < nodes.size(); ++i) {
        int64_t target = valid_target(i);
        if (target < 0)
            continue;
        children.at(target).push_back(i);
        has_target.at(i) = true;
    }
    for (size_t root = 0; root < nodes.size(); ++root) {
        if (has_target.at(root) || children.at(root).empty())
            continue;
        // Breadth-first, so that the targets are placed before the views
        // that overlap them
        std::vector<size_t> order = {root};
        for (size_t i = 0; i < order.size(); ++i)
            for (auto child : children.at(order.at(i)))
                order.push_back(child);

        std::unordered_map<size_t, int64_t> pos;
        pos[root] = 0;
        int64_t group_begin = 0;
        auto group_end = static_cast<int64_t>(nodes.at(root)->getElemsNum());
        for (size_t i = 1; i < order.size(); ++i) {
            size_t cur = order.at(i);
            auto &view = nodes.at(cur);
            auto len = static_cast<int64_t>(view->getElemsNum());
            auto target = static_cast<size_t>(valid_target(cur));
            // The view may also overlap something other than its target, and
            // not everything can overlap, so we may need a few attempts
            bool placed = false;
            for (size_t attempt = 0; attempt < 4 && !placed; ++attempt) {
                int64_t begin = pos.at(target) +
                                pickAliasDist(gen_pol, nodes.at(target), view);
                bool conflict = false;
                for (size_t j = 0; j < i; ++j) {
                    size_t other = order.at(j);
                    auto other_len =
                        static_cast<int64_t>(nodes.at(other)->getElemsNum());
                    if (rangesOverlap(begin, begin + len, pos.at(other),
                                      pos.at(other) + other_len) &&
                        !overlap_allowed(view, nodes.at(other)))
                        conflict = true;
                }
                if (!conflict) {
                    pos[cur] = begin;
                    placed = true;
                }
            }
            // Give up and put it after everything else
            if (!placed)
                pos[cur] =
                    group_end + static_cast<int64_t>(rand_val_gen->getRandId(
                                    gen_pol->view_gap_distr));
            group_begin = std::min(group_begin, pos.at(cur));
            group_end = std::max(group_end, pos.at(cur) + len);
        }

        Block block;
        for (auto idx : order) {
            block.views.emplace_back(
                nodes.at(idx), static_cast<size_t>(pos.at(idx) - group_begin));
            in_group.at(idx) = true;
        }
        block.size = static_cast<size_t>(group_end - group_begin);
        blocks.push_back(block);
    }

    // Everything else may become a view on its own
    auto add_single = [&](const std::shared_ptr<Data> &data,
                          std::vector<Probability<bool>> &prob) {
        if (!can_be_view(data) || !rand_val_gen->getRandId(prob))
            return;
        Block block;
        block.views.emplace_back(data, 0);
        block.size = data->getElemsNum();
        blocks.push_back(block);
    };
    for (auto &array : ext_inp_sym_tbl->getArrays()) {
        auto find_res = node_idx.find(array.get());
        if (find_res != node_idx.end() && in_group.at(find_res->second))
            continue;
        add_single(array, gen_pol->heap_view_prob);
    }
    for (size_t i = 0; i < outs_num; ++i) {
        if (in_group.at(i))
            continue;
        add_single(nodes.at(i), nodes.at(i)->isScalarVar()
                                    ? gen_pol->scalar_view_prob
                                    : gen_pol->heap_view_prob);
    }

    // Shuffle, so that inputs and outputs are interleaved in buffers
    rand_val_gen->shuffleVector(blocks);

    // For each type, the last buffer that new views can be appended to
    std::map<IntTypeID, std::shared_ptr<Buffer>> open_buffers;
    for (auto &block : blocks) {
        auto first = block.views.front().first;
        auto elem_type = first->getType();
        if (elem_type->isArrayType())
            elem_type =
                std::static_pointer_cast<ArrayType>(elem_type)->getBaseType();
        auto int_elem_type = std::static_pointer_cast<IntegralType>(elem_type);
        IntTypeID type_id = int_elem_type->getIntTypeId();

        std::shared_ptr<Buffer> buffer;
        size_t offset = 0;
        auto find_res = open_buffers.find(type_id);
        if (find_res != open_buffers.end() &&
            rand_val_gen->getRandId(gen_pol->share_buffer_prob)) {
            buffer = find_res->second;
            offset = buffer->getSize() +
                     rand_val_gen->getRandId(gen_pol->view_gap_distr);
        }
        else {
            buffer = std::make_shared<Buffer>(
                "buf_" + std::to_string(buffers.size()), int_elem_type);
            buffer->setFiller(rand_val_gen->getRandValue(type_id));
            buffers.push_back(buffer);
            open_buffers[type_id] = buffer;
            offset = rand_val_gen->getRandId(gen_pol->buffer_misalign_distr);
        }

        for (auto &[view, rel_offset] : block.views) {
            view->setBuffer(buffer, offset + rel_offset);
            if (view->isArray())
                std::static_pointer_cast<Array>(view)->setViewKind(
                    rand_val_gen->getRandId(gen_pol->view_kind_distr));
            buffer->addView(view);
        }
        buffer->setSize(offset + block.size);
    }

    for (auto &buffer : buffers) {
        // Leave some space after the last view, so that stray writes past it
        // still land in the buffer and change the hash
        buffer->setSize(buffer->getSize() +
                        rand_val_gen->getRandId(gen_pol->view_gap_distr));

        auto &buf_views = buffer->getViews();
        for (auto &view : buf_views) {
            auto begin = static_cast<int64_t>(view->getBufOffset());
            auto end = begin + static_cast<int64_t>(view->getElemsNum());
            if (end > static_cast<int64_t>(buffer->getSize()))
                ERROR("View is out of its buffer");
            bool overlaps = false;
            for (auto &other : buf_views) {
                if (other == view)
                    continue;
                auto other_begin = static_cast<int64_t>(other->getBufOffset());
                auto other_end =
                    other_begin + static_cast<int64_t>(other->getElemsNum());
                if (!rangesOverlap(begin, end, other_begin, other_end))
                    continue;
                overlaps = true;
                if (!overlap_allowed(view, other) ||
                    options.getAlias() < AliasLevel::STORES)
                    ERROR("Bad overlap of " +
                          view->getName(EmitCtx::default_emit_ctx) + " and " +
                          other->getName(EmitCtx::default_emit_ctx));
            }
            // restrict promises that nothing else accesses the same memory
            bool emit_restrict = false;
            if (!overlaps) {
                if (options.getEmitRestrict() == OptionLevel::ALL)
                    emit_restrict = true;
                else if (options.getEmitRestrict() == OptionLevel::SOME)
                    emit_restrict =
                        rand_val_gen->getRandId(gen_pol->emit_restrict_prob);
            }
            view->setEmitRestrict(emit_restrict);
        }
    }
}

std::vector<std::shared_ptr<Data>> ProgramGenerator::getParamViews() {
    std::vector<std::shared_ptr<Data>> ret;
    for (auto &sym_tbl : {ext_inp_sym_tbl, ext_out_sym_tbl}) {
        for (auto &array : sym_tbl->getArrays())
            if (array->isView())
                ret.push_back(array);
        for (auto &var : sym_tbl->getVars())
            if (var->isView())
                ret.push_back(var);
    }
    return ret;
}

void ProgramGenerator::emitCheckFunc(std::ostream &stream) {
    std::ostream &out_file = stream;
    out_file << "#include <stdio.h>\n";
    // We need malloc and free for heap buffers
    if (!buffers.empty())
        out_file << "#include <stdlib.h>\n";
    out_file << "\n";

    Options &options = Options::getInstance();
    if (options.getCheckAlgo() == CheckAlgo::ASSERTS) {
        stream << "static ";
        stream << (options.isC() ? "_Bool" : "bool") << " value_mismatch = ";
        stream << (options.isC() ? "0" : "false") << ";\n";
    }

    // The exact same function should be used for hash pre-computation!

    out_file << "unsigned long long int seed = 0;\n";
    out_file << "void hash(unsigned long long int *seed, unsigned long long "
                "int const v) {\n";
    out_file << "    *seed ^= v + 0x9e3779b9 + ((*seed)<<6) + ((*seed)>>2);\n";
    out_file << "}\n\n";
}

static void emitVarsDecl(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                         std::vector<std::shared_ptr<ScalarVar>> vars) {
    Options &options = Options::getInstance();
    if (options.isSYCL())
        ctx->setSYCLPrefix("app_");
    for (auto &var : vars) {
        if (!options.getAllowDeadData() && var->getIsDead())
            continue;
        // Views live in heap buffers
        if (var->isView())
            continue;
        auto init_val = std::make_shared<ConstantExpr>(var->getInitValue());
        auto decl_stmt = std::make_shared<DeclStmt>(var, init_val);
        decl_stmt->emit(ctx, stream);
        stream << "\n";
    }
    ctx->setSYCLPrefix("");
}

// Emit an alignment specifier as a declaration *prefix*. We use the standard
// C++11 `alignas`, which GCC, Clang and MSVC all accept. MSVC does not accept
// the trailing GNU `__attribute__((aligned(...)))`, and its `__declspec` is a
// prefix specifier, so a prefix `alignas` is the portable choice. For C we keep
// the GNU attribute (MSVC support is C++ only).
static std::string alignAttr(size_t alignment) {
    if (Options::getInstance().isC())
        return "__attribute__((aligned(" + std::to_string(alignment) + ")))";
    return "alignas(" + std::to_string(alignment) + ")";
}

static void emitArrayDecl(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          std::vector<std::shared_ptr<Array>> arrays) {
    Options &options = Options::getInstance();
    for (auto &array : arrays) {
        if (!options.getAllowDeadData() && array->getIsDead())
            continue;
        // Views live in heap buffers
        if (array->isView())
            continue;
        auto type = array->getType();
        assert(type->isArrayType() && "Array should have an Array type");
        auto array_type = std::static_pointer_cast<ArrayType>(type);
        if (array->getAlignment() != 0)
            stream << alignAttr(array->getAlignment()) << " ";
        stream << array_type->getBaseType()->getName(ctx) << " ";
        stream << array->getName(ctx) << " ";
        for (const auto &dimension : array_type->getDimensions()) {
            stream << "[" << dimension << "] ";
        }
        stream << ";\n";
    }
}

void ProgramGenerator::emitDecl(std::shared_ptr<EmitCtx> ctx,
                                std::ostream &stream) {
    emitVarsDecl(ctx, stream, ext_inp_sym_tbl->getVars());
    emitVarsDecl(ctx, stream, ext_out_sym_tbl->getVars());

    emitArrayDecl(ctx, stream, ext_inp_sym_tbl->getArrays());
    emitArrayDecl(ctx, stream, ext_out_sym_tbl->getArrays());

    emitBuffersDecl(ctx, stream);
}

void ProgramGenerator::emitBuffersDecl(std::shared_ptr<EmitCtx> ctx,
                                       std::ostream &stream) {
    if (buffers.empty())
        return;
    stream << "\n// Heap buffers and the views (element ranges) that they "
              "hold\n";
    for (auto &buffer : buffers) {
        stream << "// " << buffer->getName() << ":";
        for (auto &view : buffer->getViews()) {
            stream << " " << view->getName(ctx) << " [" << view->getBufOffset()
                   << ", " << view->getBufOffset() + view->getElemsNum() << ")";
            if (view->isOpaque())
                stream << " (opaque)";
        }
        stream << "\n";
    }
    for (auto &buffer : buffers) {
        stream << buffer->getElemType()->getName(ctx) << " *"
               << buffer->getName() << ";\n";
    }
}

// Returns an access to the element [i_0] [i_1] ... of the array, as the driver
// sees it. Views are accessed through their buffers.
static std::string driverElemAccess(std::shared_ptr<EmitCtx> ctx,
                                    std::shared_ptr<Array> array) {
    std::stringstream ss;
    auto array_type = std::static_pointer_cast<ArrayType>(array->getType());
    size_t dims_num = array_type->getDimensions().size();
    if (!array->isView()) {
        ss << array->getName(ctx) << " ";
        for (size_t i = 0; i < dims_num; ++i)
            ss << "[i_" << i << "] ";
        return ss.str();
    }

    auto strides = array->getStrides();
    ss << array->getBuffer()->getName() << " [" << array->getBufOffset();
    for (size_t i = 0; i < dims_num; ++i) {
        ss << " + i_" << i;
        if (strides.at(i) != 1)
            ss << " * " << strides.at(i);
    }
    ss << "] ";
    return ss.str();
}

// Same as driverElemAccess, but for scalar variables
static std::string driverVarAccess(std::shared_ptr<EmitCtx> ctx,
                                   std::shared_ptr<ScalarVar> var) {
    if (!var->isView())
        return var->getName(ctx);
    return var->getBuffer()->getName() + " [" +
           std::to_string(var->getBufOffset()) + "]";
}

static void emitArrayInit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          std::vector<std::shared_ptr<Array>> arrays) {
    Options &options = Options::getInstance();
    for (const auto &array : arrays) {
        if (!options.getAllowDeadData() && array->getIsDead())
            continue;
        std::string offset = "    ";
        auto type = array->getType();
        assert(type->isArrayType() && "Array should have an Array type");
        auto array_type = std::static_pointer_cast<ArrayType>(type);
        size_t idx = 0;
        for (const auto &dimension : array_type->getDimensions()) {
            stream << offset << "for (size_t i_" << idx << " = 0; i_" << idx
                   << " < " << dimension << "; ++i_" << idx << ") \n";
            offset += "    ";
            idx++;
        }
        stream << offset << driverElemAccess(ctx, array);
        stream << "= ";
        auto emit_const_expr = [&array, &ctx, &stream](bool use_main_vals) {
            auto init_val = array->getInitValues(use_main_vals);
            auto init_const = std::make_shared<ConstantExpr>(init_val);
            init_const->emit(ctx, stream);
        };
        if (array->getMulValsAxisIdx() != -1) {
            stream << "(i_" << array->getMulValsAxisIdx() << " % "
                   << Options::vals_number << " == " << Options::main_val_idx
                   << ") ? ";
        }
        emit_const_expr(true);
        if (array->getMulValsAxisIdx() != -1) {
            stream << " : ";
            emit_const_expr(false);
        }
        stream << ";\n";
    }
}

void ProgramGenerator::emitInit(std::shared_ptr<EmitCtx> ctx,
                                std::ostream &stream) {
    stream << "void init() {\n";
    emitBuffersInit(ctx, stream);
    emitArrayInit(ctx, stream, ext_inp_sym_tbl->getArrays());
    emitArrayInit(ctx, stream, ext_out_sym_tbl->getArrays());
    stream << "}\n\n";
}

void ProgramGenerator::emitBuffersInit(std::shared_ptr<EmitCtx> ctx,
                                       std::ostream &stream) {
    for (auto &buffer : buffers) {
        std::string type_name = buffer->getElemType()->getName(ctx);
        stream << "    " << buffer->getName() << " = (" << type_name
               << " *) malloc(" << buffer->getSize() << " * sizeof("
               << type_name << "));\n";
        // The views are initialized later. This only fills the gaps.
        stream << "    for (size_t i = 0; i < " << buffer->getSize()
               << "; ++i)\n";
        stream << "        " << buffer->getName() << " [i] = ";
        std::make_shared<ConstantExpr>(buffer->getFiller())->emit(ctx, stream);
        stream << ";\n";
    }
    // Arrays are initialized in emitArrayInit
    for (auto &var : ext_out_sym_tbl->getVars()) {
        if (!var->isView())
            continue;
        stream << "    " << driverVarAccess(ctx, var) << " = ";
        std::make_shared<ConstantExpr>(var->getInitValue())->emit(ctx, stream);
        stream << ";\n";
    }
}

void ProgramGenerator::emitBuffersCheck(std::shared_ptr<EmitCtx> ctx,
                                        std::ostream &stream) {
    Options &options = Options::getInstance();
    // In the asserts mode the output views are checked one by one
    if (options.getCheckAlgo() != CheckAlgo::HASH)
        return;
    // Inputs are never written, but we hash whole buffers anyway, so that
    // stray writes are caught
    for (auto &buffer : buffers) {
        stream << "    for (size_t i = 0; i < " << buffer->getSize()
               << "; ++i)\n";
        stream << "        hash(&seed, " << buffer->getName() << " [i]);\n";
    }
}

void ProgramGenerator::emitCheck(std::shared_ptr<EmitCtx> ctx,
                                 std::ostream &stream) {
    stream << "void checksum() {\n";

    Options &options = Options::getInstance();

    auto emit_pol = ctx->getEmitPolicy();

    if (options.isSYCL())
        ctx->setSYCLPrefix("app_");

    for (auto &var : ext_out_sym_tbl->getVars()) {
        // Views are hashed as a part of their buffers
        if (var->isView() && options.getCheckAlgo() == CheckAlgo::HASH)
            continue;
        std::string var_name = driverVarAccess(ctx, var);

        if (options.getCheckAlgo() == CheckAlgo::HASH ||
            options.getCheckAlgo() == CheckAlgo::PRECOMPUTE) {
            stream << "    hash(&seed, " << var_name << ");\n";
            if (options.getCheckAlgo() == CheckAlgo::PRECOMPUTE)
                hash(var->getCurrentValue().getAbsValue().value);
        }
        else if (options.getCheckAlgo() == CheckAlgo::ASSERTS) {
            auto const_val =
                std::make_shared<ConstantExpr>(var->getCurrentValue());
            stream << "    value_mismatch |= " << var_name << " != ";
            const_val->emit(ctx, stream);
            stream << ";\n";
        }
        else {
            ERROR("Unsupported");
        }
    }

    ctx->setSYCLPrefix("");

    for (const auto &array : ext_out_sym_tbl->getArrays()) {
        // Views are hashed as a part of their buffers
        if (array->isView() && options.getCheckAlgo() == CheckAlgo::HASH)
            continue;
        std::string offset = "    ";
        auto type = array->getType();
        assert(type->isArrayType() && "Array should have an Array type");
        auto array_type = std::static_pointer_cast<ArrayType>(type);
        size_t idx = 0;
        for (const auto &dimension : array_type->getDimensions()) {
            stream << offset << "for (size_t i_" << idx << " = 0; i_" << idx
                   << " < " << dimension << "; ++i_" << idx << ") \n";
            offset += "    ";
            idx++;
        }

        if (options.getCheckAlgo() == CheckAlgo::HASH ||
            options.getCheckAlgo() == CheckAlgo::PRECOMPUTE) {
            stream << offset << "hash(&seed, ";
            if (options.getCheckAlgo() == CheckAlgo::PRECOMPUTE)
                hashArray(array);
        }
        else if (options.getCheckAlgo() == CheckAlgo::ASSERTS)
            stream << offset << "value_mismatch |= ";
        else
            ERROR("Unsupported");

        std::string arr_name = driverElemAccess(ctx, array);
        stream << arr_name;

        if (options.getCheckAlgo() == CheckAlgo::ASSERTS) {
            auto const_val =
                std::make_shared<ConstantExpr>((array->getCurrentValues(true)));
            stream << "!= ";
            const_val->emit(ctx, stream);
            auto emit_cmp = [&arr_name, &ctx, &stream](IRValue val) {
                stream << " && " << arr_name << "!= ";
                auto const_val = std::make_shared<ConstantExpr>(val);
                const_val->emit(ctx, stream);
            };
            emit_cmp(array->getInitValues(true));
            if (array->getMulValsAxisIdx() != -1) {
                emit_cmp(array->getCurrentValues(false));
                emit_cmp(array->getInitValues(false));
            }
        }
        else
            stream << ")";
        stream << ";\n";
    }
    emitBuffersCheck(ctx, stream);
    stream << "}\n";
}

// This buffer tracks what input data we pass as a parameters to test functions
static std::vector<std::string> pass_as_param_buffer;
static bool any_vars_as_params = false;
static bool any_arrays_as_params = false;

static void emitVarExtDecl(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                           std::vector<std::shared_ptr<ScalarVar>> vars,
                           bool inp_category) {
    auto emit_pol = ctx->getEmitPolicy();
    Options &options = Options::getInstance();
    if (options.isSYCL())
        ctx->setSYCLPrefix("app_");
    for (auto &var : vars) {
        if (!options.getAllowDeadData() && var->getIsDead())
            continue;
        // Views are always passed as parameters (see emitViewFuncParam)
        if (var->isView())
            continue;
        bool pass_as_param = false;
        if (inp_category) {
            if (options.inpAsArgs() == OptionLevel::SOME)
                pass_as_param =
                    rand_val_gen->getRandId(emit_pol->pass_as_param_distr);
            else if (options.inpAsArgs() == OptionLevel::ALL)
                pass_as_param = true;
        }

        if (pass_as_param) {
            pass_as_param_buffer.push_back(var->getName(ctx));
            any_vars_as_params = true;
            continue;
        }
        stream << "extern ";
        stream << var->getType()->getName(ctx);
        stream << " ";
        stream << var->getName(ctx) << ";\n";
    }
    ctx->setSYCLPrefix("");
}

static void emitArrayExtDecl(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                             std::vector<std::shared_ptr<Array>> arrays,
                             bool inp_category) {
    auto emit_pol = ctx->getEmitPolicy();
    Options &options = Options::getInstance();
    for (auto &array : arrays) {
        if (!options.getAllowDeadData() && array->getIsDead())
            continue;
        // Views are always passed as parameters (see emitViewFuncParam)
        if (array->isView())
            continue;
        bool pass_as_param = false;
        if (inp_category) {
            if (options.inpAsArgs() == OptionLevel::SOME)
                pass_as_param =
                    rand_val_gen->getRandId(emit_pol->pass_as_param_distr);
            else if (options.inpAsArgs() == OptionLevel::ALL)
                pass_as_param = true;
        }

        if (pass_as_param) {
            pass_as_param_buffer.push_back(array->getName(ctx));
            any_arrays_as_params = true;
            continue;
        }

        auto type = array->getType();
        assert(type->isArrayType() && "Array should have an Array type");
        auto array_type = std::static_pointer_cast<ArrayType>(type);

        // Decide on the alignment before emitting the declaration: the
        // alignment specifier has to be emitted as a prefix (see alignAttr).
        size_t alignment = 0;
        if (options.isCXX() &&
            options.getEmitAlignAttr() != OptionLevel::NONE) {
            bool emit_align_attr = true;
            if (options.getEmitAlignAttr() == OptionLevel::SOME)
                emit_align_attr =
                    rand_val_gen->getRandId(emit_pol->emit_align_attr_distr);
            if (emit_align_attr) {
                AlignmentSize align_size = options.getAlignSize();
                if (!options.getUniqueAlignSize())
                    align_size =
                        rand_val_gen->getRandId(emit_pol->align_size_distr);
                switch (align_size) {
                    case AlignmentSize::A16:
                        alignment = 16;
                        break;
                    case AlignmentSize::A32:
                        alignment = 32;
                        break;
                    case AlignmentSize::A64:
                        alignment = 64;
                        break;
                    case AlignmentSize::MAX_ALIGNMENT_SIZE:
                        ERROR("Bad alignment size");
                }
                array->setAlignment(alignment);
            }
        }

        // The alignment specifier must precede all decl-specifiers (i.e. come
        // before `extern`); placing it between `extern` and the type is
        // ill-formed for a standard attribute.
        if (alignment != 0)
            stream << alignAttr(alignment) << " ";
        stream << "extern ";
        stream << array_type->getBaseType()->getName(ctx);
        stream << " ";
        stream << array->getName(ctx) << " ";
        for (const auto &dimension : array_type->getDimensions()) {
            stream << "[" << dimension << "] ";
        }
        stream << ";\n";
    }
}

void ProgramGenerator::emitExtDecl(std::shared_ptr<EmitCtx> ctx,
                                   std::ostream &stream) {
    Options &options = Options::getInstance();
    if (options.isISPC())
        ctx->setIspcTypes(true);
    emitVarExtDecl(ctx, stream, ext_inp_sym_tbl->getVars(), true);
    emitVarExtDecl(ctx, stream, ext_out_sym_tbl->getVars(), false);
    emitArrayExtDecl(ctx, stream, ext_inp_sym_tbl->getArrays(), true);
    emitArrayExtDecl(ctx, stream, ext_out_sym_tbl->getArrays(), false);
    ctx->setIspcTypes(false);
}

static std::string placeSep(bool cond) { return cond ? ", " : ""; }

static bool emitVarFuncParam(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                             std::vector<std::shared_ptr<ScalarVar>> vars,
                             bool emit_type, bool ispc_type) {
    bool emit_any = false;
    Options &options = Options::getInstance();
    if (options.isSYCL())
        ctx->setSYCLPrefix("app_");
    for (auto &var : vars) {
        if (!options.getAllowDeadData() && var->getIsDead())
            continue;
        if (std::find(pass_as_param_buffer.begin(), pass_as_param_buffer.end(),
                      var->getName(ctx)) == pass_as_param_buffer.end())
            continue;

        stream << placeSep(emit_any);
        if (emit_type)
            stream << var->getType()->getName(ctx) << " ";
        stream << var->getName(ctx);
        emit_any = true;
    }
    ctx->setSYCLPrefix("");
    return emit_any;
}

static bool emitArrayFuncParam(std::shared_ptr<EmitCtx> ctx,
                               std::ostream &stream, bool prev_category_exist,
                               std::vector<std::shared_ptr<Array>> arrays,
                               bool emit_type, bool ispc_type, bool emit_dims) {
    bool first = true;
    Options &options = Options::getInstance();
    for (auto &array : arrays) {
        if (!options.getAllowDeadData() && array->getIsDead())
            continue;
        if (std::find(pass_as_param_buffer.begin(), pass_as_param_buffer.end(),
                      array->getName(ctx)) == pass_as_param_buffer.end())
            continue;

        auto type = array->getType();
        assert(type->isArrayType() && "Array should have an Array type");
        auto array_type = std::static_pointer_cast<ArrayType>(type);
        stream << placeSep(prev_category_exist || !first);
        if (emit_type)
            stream << array_type->getBaseType()->getName(ctx) << " ";
        stream << array->getName(ctx) << " ";
        if (emit_dims)
            for (const auto &dimension : array_type->getDimensions())
                stream << "[" << dimension << "] ";

        first = false;
    }
    return prev_category_exist || !first;
}

static std::string restrictKeyword() {
    return Options::getInstance().isC() ? "restrict" : "__restrict";
}

// Views are passed to the test function as pointers into heap buffers: either
// "T *p" (flat arrays and scalar variables) or "T (*p)[D1][D2]" (pointer to
// array). With emit_type we emit the parameter declarations, otherwise the
// arguments that the driver passes.
static bool emitViewFuncParam(std::shared_ptr<EmitCtx> ctx,
                              std::ostream &stream, bool prev_category_exist,
                              std::vector<std::shared_ptr<Data>> views,
                              bool emit_type) {
    bool emit_any = prev_category_exist;
    for (auto &view : views) {
        std::string type_name = view->getBuffer()->getElemType()->getName(ctx);
        bool ptr_to_arr = false;
        std::string inner_dims;
        if (view->isArray()) {
            auto array = std::static_pointer_cast<Array>(view);
            auto array_type =
                std::static_pointer_cast<ArrayType>(array->getType());
            auto &dims = array_type->getDimensions();
            ptr_to_arr =
                array->getViewKind() == ViewKind::PTR_TO_ARR && dims.size() > 1;
            for (size_t i = 1; i < dims.size(); ++i)
                inner_dims += "[" + std::to_string(dims.at(i)) + "]";
        }

        stream << placeSep(emit_any);
        emit_any = true;
        if (emit_type) {
            std::string restr =
                view->getEmitRestrict() ? restrictKeyword() + " " : "";
            stream << type_name << " ";
            if (ptr_to_arr)
                stream << "(*" << restr << view->getName(ctx) << ")"
                       << inner_dims;
            else
                stream << "*" << restr << view->getName(ctx);
        }
        else {
            std::string ptr = view->getBuffer()->getName() + " + " +
                              std::to_string(view->getBufOffset());
            if (ptr_to_arr)
                stream << "(" << type_name << " (*)" << inner_dims << ")("
                       << ptr << ")";
            else
                stream << ptr;
        }
    }
    return emit_any;
}

void emitSYCLBuffers(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                     std::string offset,
                     std::vector<std::shared_ptr<ScalarVar>> vars) {
    Options &options = Options::getInstance();
    for (auto &var : vars) {
        if (!options.getAllowDeadData() && var->getIsDead())
            continue;
        stream << offset << "buffer<";
        stream << var->getType()->getName(ctx);
        stream << ", 1> " << var->getName(ctx) << "_buf { ";
        stream << "&app_" << var->getName(ctx) << ", range<1>(1) };\n";
    }
}

void emitSYCLAccessors(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                       std::string offset,
                       std::vector<std::shared_ptr<ScalarVar>> vars,
                       bool is_inp) {
    Options &options = Options::getInstance();
    for (auto &var : vars) {
        if (!options.getAllowDeadData() && var->getIsDead())
            continue;
        stream << offset << "auto " << var->getName(ctx) << " = ";
        stream << var->getName(ctx) << "_buf.get_access<access::mode::";
        stream << (is_inp ? "read" : "write") << ">(cgh);\n";
    }
}

void ProgramGenerator::emitTest(std::shared_ptr<EmitCtx> ctx,
                                std::ostream &stream) {
    Options &options = Options::getInstance();
    stream << "#include \"init.h\"\n";
    if (options.isC()) {
        MinCall::emitCDefinition(ctx, stream);
        MaxCall::emitCDefinition(ctx, stream);
    }
    if (options.isCXX())
        stream << "#include <algorithm>\n";
    else if (options.isSYCL()) {
        stream << "#include <CL/sycl.hpp>\n";
    }

    if (options.isISPC()) {
        ctx->setIspcTypes(true);
        stream << "export ";
    }
    stream << "void test(";

    bool emit_any = emitVarFuncParam(ctx, stream, ext_inp_sym_tbl->getVars(),
                                     true, options.isISPC());

    emit_any =
        emitArrayFuncParam(ctx, stream, emit_any, ext_inp_sym_tbl->getArrays(),
                           true, options.isISPC(), true);
    emitViewFuncParam(ctx, stream, emit_any, getParamViews(), true);

    stream << ") ";

    if (options.isSYCL()) {
        stream << "{\n";
        stream << "    using namespace cl::sycl;\n\n";
        stream << "    {\n";

        stream << "#if defined(CPU)\n";
        stream << "        cpu_selector selector;\n";
        stream << "#elif defined(GPU)\n";
        stream << "        gpu_selector selector;\n";
        stream << "#else\n";
        stream << "        default_selector selector;\n";
        stream << "#endif\n";
        stream << "        queue myQueue(selector);\n";
        emitSYCLBuffers(ctx, stream, "        ", ext_inp_sym_tbl->getVars());
        emitSYCLBuffers(ctx, stream, "        ", ext_out_sym_tbl->getVars());

        stream << "        myQueue.submit([&](handler & cgh) {\n";
        emitSYCLAccessors(ctx, stream, "            ",
                          ext_inp_sym_tbl->getVars(), true);
        emitSYCLAccessors(ctx, stream, "            ",
                          ext_out_sym_tbl->getVars(), false);
        stream << "            cgh.single_task<class test_func>([=] ()\n";
    }

    if (options.isSYCL())
        ctx->setSYCLAccess(true);
    new_test->emit(ctx, stream, !options.isSYCL() ? "" : "            ");

    if (options.isSYCL()) {
        stream << "            );\n";
        stream << "        });\n";
        stream << "    }\n";
        stream << "}\n";
    }
    ctx->setSYCLAccess(false);
    ctx->setIspcTypes(false);
}

void ProgramGenerator::emitMain(std::shared_ptr<EmitCtx> ctx,
                                std::ostream &stream) {
    Options &options = Options::getInstance();
    if (options.isISPC())
        stream << "extern \"C\" { ";

    stream << "void test(";

    bool emit_any =
        emitVarFuncParam(ctx, stream, ext_inp_sym_tbl->getVars(), true, false);
    emit_any = emitArrayFuncParam(
        ctx, stream, emit_any, ext_inp_sym_tbl->getArrays(), true, false, true);
    emitViewFuncParam(ctx, stream, emit_any, getParamViews(), true);

    stream << ");";
    if (options.isISPC())
        stream << " }\n";
    stream << "\n\n";
    stream << "int main() {\n";
    stream << "    init();\n";
    stream << "    test(";

    emit_any =
        emitVarFuncParam(ctx, stream, ext_inp_sym_tbl->getVars(), false, false);

    emit_any =
        emitArrayFuncParam(ctx, stream, emit_any, ext_inp_sym_tbl->getArrays(),
                           false, false, false);
    emitViewFuncParam(ctx, stream, emit_any, getParamViews(), false);

    stream << ");\n";
    stream << "    checksum();\n";
    for (auto &buffer : buffers)
        stream << "    free(" << buffer->getName() << ");\n";
    stream << "    printf(\"%llu\\n\", seed);\n";
    if (options.getCheckAlgo() == CheckAlgo::PRECOMPUTE) {
        stream << "    if (seed != " << hash_seed << "ULL) \n";
        stream << "        printf(\"ERROR: hash mismatch\\n\");\n";
    }
    if (options.getCheckAlgo() == CheckAlgo::ASSERTS) {
        stream << "    if (value_mismatch) \n";
        stream << "        printf(\"ERROR: value mismatch\\n\");\n";
    }
    stream << "}\n";
}

void ProgramGenerator::emit() {
    Options &options = Options::getInstance();
    auto emit_ctx = std::make_shared<EmitCtx>();
    // We need to narrow options if we were asked to do so
    if (options.getUniqueAlignSize() &&
        options.getAlignSize() == AlignmentSize::MAX_ALIGNMENT_SIZE) {
        AlignmentSize align_size = rand_val_gen->getRandId(
            emit_ctx->getEmitPolicy()->align_size_distr);
        options.setAlignSize(align_size);
    }

    std::ofstream out_file;

    // TODO: probably won't work on Windows
    std::string out_dir = options.getOutDir() + "/";

    auto open_file = [&out_file, &out_dir](std::string file_name) {
        out_file.open(out_dir + file_name);
        if (!out_file)
            ERROR(std::string("Can't open file ") + file_name);
    };

    open_file("init.h");
    emitExtDecl(emit_ctx, out_file);
    out_file.close();

    std::string func_file_ext, driver_file_ext;
    if (options.isC()) {
        func_file_ext = "c";
        driver_file_ext = "c";
    }
    else if (options.isCXX() || options.isSYCL()) {
        func_file_ext = "cpp";
        driver_file_ext = "cpp";
    }
    else if (options.isISPC()) {
        func_file_ext = "ispc";
        driver_file_ext = "cpp";
    }
    open_file("func." + func_file_ext);
    out_file << "/*\n";
    options.dump(out_file);
    out_file << "*/\n";
    emitTest(emit_ctx, out_file);
    out_file.close();

    open_file("driver." + driver_file_ext);
    emitCheckFunc(out_file);
    emitDecl(emit_ctx, out_file);
    emitInit(emit_ctx, out_file);
    emitCheck(emit_ctx, out_file);
    emitMain(emit_ctx, out_file);
    out_file.close();
}

void ProgramGenerator::hash(unsigned long long int const v) {
    // This function has to be exactly the same as the one that we use for hash
    // computation
    hash_seed ^= v + 0x9e3779b9 + (hash_seed << 6) + (hash_seed >> 2);
}

void ProgramGenerator::hashArray(std::shared_ptr<Array> const &arr) {
    assert(arr->getType()->isArrayType() && "Array should have array type");
    auto arr_type = std::static_pointer_cast<ArrayType>(arr->getType());
    std::vector<size_t> idx_vec(arr_type->getDimensions().size(), 0);
    auto &dims = arr_type->getDimensions();
    // TODO: this is broken now
    uint64_t init_val =
        arr->getInitValues(Options::main_val_idx).getAbsValue().value;
    uint64_t cur_val =
        arr->getCurrentValues(Options::main_val_idx).getAbsValue().value;
    std::vector<size_t> steps = {};
    hashArrayStep(arr, dims, idx_vec, 0, false, init_val, cur_val, steps);
}

void ProgramGenerator::hashArrayStep(std::shared_ptr<Array> const &arr,
                                     std::vector<size_t> &dims,
                                     std::vector<size_t> &idx_vec,
                                     size_t cur_idx, bool has_to_use_init_val,
                                     uint64_t &init_val, uint64_t &cur_val,
                                     std::vector<size_t> &steps) {
    size_t vec_last_idx = idx_vec.size() - 1;
    // TODO: this is also broken
    size_t cur_val_size = 0;
    size_t cur_dim = dims[cur_idx];
    size_t cur_step = steps[cur_idx];

    for (size_t i = 0; i < cur_dim; ++i) {
        has_to_use_init_val |= (i >= cur_val_size);
        if (cur_idx != vec_last_idx) {
            idx_vec[cur_idx] = i;
            hashArrayStep(arr, dims, idx_vec, cur_idx + 1,
                          has_to_use_init_val || (i % cur_step != 0), init_val,
                          cur_val, steps);
        }
        else {
            hash(has_to_use_init_val || (i % cur_step != 0) ? init_val
                                                            : cur_val);
        }
    }
}
