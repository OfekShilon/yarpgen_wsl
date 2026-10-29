/*
Copyright (c) 2020, Intel Corporation
Copyright (c) 2020, University of Utah

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

#include "context.h"
#include "options.h"
#include "stmt.h"

#include <iostream>
#include <regex>
#include <set>
#include <sstream>

using namespace yarpgen;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "ERROR at " << __FILE__ << ":" << __LINE__            \
                      << ", function " << __func__ << "():\n    " << (msg)     \
                      << std::endl;                                            \
            abort();                                                           \
        }                                                                      \
    } while (false)

// The simple-loop profiles override parts of the policy that other options
// have already set up, and must not contradict them.
static void simpleLoopPolicyTest() {
    Options &options = Options::getInstance();
    VectorizerTarget orig_target = options.getVectorizerTarget();

    // "--max-array-dims=1": the GCC/Clang profile used to raise the limit to
    // 2, beyond what stencil_in_dim_prob was set up for.
    options.setMaxArrayDims(1);
    options.setVectorizerTarget(VectorizerTarget::GCC_CLANG);
    GenPolicy capped_gen_pol;
    capped_gen_pol.makeVectorizable(/*simple*/ true);
    CHECK(capped_gen_pol.array_dims_num_limit == 1, "array dims cap");
    options.setMaxArrayDims(0);

    // "--emit-pragmas=all": the MSVC profile allows no pragmas at all, which
    // used to make Pragma::create sample an empty distribution.
    options.setVectorizerTarget(VectorizerTarget::MSVC);
    auto msvc_gen_pol = std::make_shared<GenPolicy>();
    msvc_gen_pol->makeVectorizable(/*simple*/ true);
    auto pop_ctx = std::make_shared<PopulateCtx>();
    pop_ctx->setGenPolicy(msvc_gen_pol);
    auto pragmas = Pragma::create(
        static_cast<size_t>(PragmaKind::MAX_PRAGMA_KIND) - 1, pop_ctx);
    CHECK(pragmas.empty(), "no pragmas allowed");

    options.setVectorizerTarget(orig_target);
}

// Opaque expressions read values that outputs may overwrite through aliasing
// pointers, so they have to be well defined for any values: everything is
// computed in unsigned long long, and there are no divisions or shifts.
static void opaqueExprTest() {
    Options &options = Options::getInstance();
    AliasLevel orig_alias = options.getAlias();
    options.setAlias(AliasLevel::LOADS);

    auto ctx = std::make_shared<PopulateCtx>();
    for (size_t i = 0; i < 3; ++i) {
        auto var = ScalarVar::create(ctx);
        ctx->getExtInpSymTable()->addVar(var);
        ctx->getExtInpSymTable()->addVarExpr(
            std::make_shared<ScalarVarUseExpr>(var));
    }
    const size_t dim = 20;
    auto iter = Iterator::create(ctx, dim);
    ctx->addDimension(dim);
    std::vector<std::shared_ptr<Array>> opaque_arrs;
    for (size_t i = 0; i < 4; ++i) {
        auto arr = Array::create(ctx, true);
        if (i % 2 == 0) {
            arr->setOpaque(true);
            opaque_arrs.push_back(arr);
        }
        ctx->getExtInpSymTable()->addArray(arr);
    }
    ctx->getLocalSymTable()->addIters(iter);

    // Outside of opaque mode, opaque inputs are invisible
    for (auto &arr : SubscriptExpr::getSuitableArrays(ctx))
        CHECK(!arr->isOpaque(), "opaque input outside of opaque mode");

    ctx->setOpaqueMode(true);
    auto gen_pol = std::make_shared<GenPolicy>(*ctx->getGenPolicy());
    gen_pol->makeOpaque();
    ctx->setGenPolicy(gen_pol);

    auto emit_ctx = std::make_shared<EmitCtx>();
    // Every leaf that reads a variable or an array is converted explicitly,
    // unless it is unsigned long long already
    auto is_ullong = [](std::shared_ptr<Type> type) {
        if (type->isArrayType())
            type = std::static_pointer_cast<ArrayType>(type)->getBaseType();
        return std::static_pointer_cast<IntegralType>(type)->getIntTypeId() ==
               IntTypeID::ULLONG;
    };
    std::set<std::string> ullong_leaves;
    for (auto &var : ctx->getExtInpSymTable()->getVars())
        if (is_ullong(var->getType()))
            ullong_leaves.insert(var->getName(emit_ctx));
    for (auto &arr : ctx->getExtInpSymTable()->getArrays())
        if (is_ullong(arr->getType()))
            ullong_leaves.insert(arr->getName(emit_ctx));
    std::regex leaf_re(R"((\S*)\s*\b((?:var|arr)_\d+)\b)");
    size_t opaque_reads_num = 0;
    for (size_t i = 0; i < 300; ++i) {
        auto reads = std::make_shared<std::vector<std::shared_ptr<Data>>>();
        ctx->setOpaqueReads(reads);
        auto expr = ArithmeticExpr::create(ctx);
        expr->propagateType();
        auto type = expr->getValue()->getType();
        CHECK(
            type->isIntType() &&
                std::static_pointer_cast<IntegralType>(type)->getIntTypeId() ==
                    IntTypeID::ULLONG,
            "opaque expression is not unsigned long long");
        opaque_reads_num += reads->size();

        std::stringstream sstream;
        expr->emit(emit_ctx, sstream);
        std::string str = sstream.str();
        for (auto op : {" / ", " % ", " << ", " >> "})
            CHECK(str.find(op) == std::string::npos,
                  "bad operator in opaque expression: " + str);
        for (auto it = std::sregex_iterator(str.begin(), str.end(), leaf_re);
             it != std::sregex_iterator(); ++it)
            CHECK(it->str(1) == "int)" || ullong_leaves.count(it->str(2)),
                  "unconverted leaf " + it->str(2) + " in " + str);
    }
    CHECK(opaque_reads_num > 0, "opaque inputs are never read");
    for (auto &arr : opaque_arrs)
        CHECK(!arr->isReadInOMPSimdRegion(1), "no simd regions here");

    options.setAlias(orig_alias);
}

int main() {
    rand_val_gen = std::make_shared<RandValGen>(0);
    simpleLoopPolicyTest();
    opaqueExprTest();

    auto gen_ctx = std::make_shared<GenCtx>();
    auto scope_stmt = ScopeStmt::generateStructure(gen_ctx);
    auto emit_ctx = std::make_shared<EmitCtx>();
    scope_stmt->emit(emit_ctx, std::cout);
    std::cout << std::endl;
    return 0;
}
