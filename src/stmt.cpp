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

#include "stmt.h"
#include "options.h"
#include "statistics.h"

#include <algorithm>
#include <utility>

using namespace yarpgen;

void ExprStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                    std::string offset) {
    stream << offset;
    expr->emit(ctx, stream);
    stream << ";";
}

std::shared_ptr<ExprStmt> ExprStmt::create(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();

    auto new_active_ctx = std::make_shared<PopulateCtx>(*ctx);

    std::shared_ptr<AssignmentExpr> expr;
    int64_t total_iters_num =
        std::accumulate(new_active_ctx->getLocalSymTable()->getIters().begin(),
                        new_active_ctx->getLocalSymTable()->getIters().end(), 1,
                        [](size_t a, const std::shared_ptr<Iterator> &b) {
                            return a * b->getTotalItersNum();
                        });

    // TODO: relax constraints on reduction expressions
    auto expr_kind =
        (total_iters_num >
             static_cast<int64_t>(ITERATIONS_THRESHOLD_FOR_REDUCTION) ||
         ctx->isInsideForeach())
            ? IRNodeKind::ASSIGN
            : rand_val_gen->getRandId(gen_pol->expr_stmt_kind_pop_distr);

    if (expr_kind == IRNodeKind::ASSIGN) {
        expr = AssignmentExpr::create(new_active_ctx);
        total_iters_num = -1;
    }
    else if (expr_kind == IRNodeKind::REDUCTION) {
        // TODO: add support for multiple values in reduction expressions
        new_active_ctx->setAllowMulVals(false);
        expr = ReductionExpr::create(new_active_ctx);
    }

    EvalCtx eval_ctx;
    eval_ctx.total_iter_num = total_iters_num;
    auto eval_res = expr->evaluate(eval_ctx);
    if (eval_res->hasUB())
        expr->rebuild(eval_ctx);
    expr->propagateValue(eval_ctx);
    if (new_active_ctx->getAllowMulVals()) {
        eval_ctx.mul_vals_iter = new_active_ctx->getMulValsIter();
        eval_ctx.use_main_vals = false;
        eval_res = expr->evaluate(eval_ctx);
    }

    if (eval_res->hasUB()) {
        expr->rebuild(eval_ctx);
    }

    if (new_active_ctx->getAllowMulVals())
        expr->propagateValue(eval_ctx);

    return std::make_shared<ExprStmt>(expr);
}

void DeclStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                    std::string offset) {
    stream << offset;
    // TODO: we need to do the right thing here
    stream << cvQualifierPrefix(data->getDeclCVQualifier());
    stream << data->getType()->getName(ctx) << " ";
    stream << data->getName(ctx);
    if (init_expr.use_count() != 0) {
        stream << " = ";
        init_expr->emit(ctx, stream);
    }
    stream << ";";
}

void StmtBlock::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                     std::string offset) {
    for (const auto &stmt : stmts) {
        stmt->emit(ctx, stream, offset);
        // TODO: will that work if we have suffix?
        if (stmt->getKind() != IRNodeKind::LOOP_SEQ &&
            stmt->getKind() != IRNodeKind::LOOP_NEST)
            stream << "\n";
    }
}

std::shared_ptr<StmtBlock>
StmtBlock::generateStructure(std::shared_ptr<GenCtx> ctx) {
    std::vector<std::shared_ptr<Stmt>> stmts;

    auto gen_policy = ctx->getGenPolicy();
    size_t stmt_num = rand_val_gen->getRandId(gen_policy->scope_stmt_num_distr);
    stmts.reserve(stmt_num);

    Statistics &stats = Statistics::getInstance();

    std::shared_ptr<Stmt> new_stmt;
    for (size_t i = 0; i < stmt_num; ++i) {
        IRNodeKind stmt_kind =
            rand_val_gen->getRandId(gen_policy->stmt_kind_struct_distr);

        bool fallback = false;
        // Last stmt that we can fit
        fallback |= stats.getStmtNum() + 1 >= gen_policy->stmt_num_lim;
        // LoopSeq and If-else create two new stmt
        fallback |= (stmt_kind == IRNodeKind::LOOP_SEQ ||
                     stmt_kind == IRNodeKind::IF_ELSE) &&
                    (stats.getStmtNum() + 2 >= gen_policy->stmt_num_lim);
        // Loop nest creates at least three new stmt (single loop is a loop
        // stmt)
        fallback |= stmt_kind == IRNodeKind::LOOP_NEST &&
                    (stats.getStmtNum() + 3 >= gen_policy->stmt_num_lim);
        // A switch creates one scope per case plus an optional default.
        fallback |= stmt_kind == IRNodeKind::SWITCH &&
                    (stats.getStmtNum() +
                         gen_policy->switch_case_num_distr.size() + 2 >=
                     gen_policy->stmt_num_lim);
        if (fallback)
            break;

        if (stmt_kind == IRNodeKind::LOOP_SEQ &&
            ctx->getLoopDepth() < gen_policy->loop_depth_limit)
            new_stmt = LoopSeqStmt::generateStructure(ctx);
        else if (stmt_kind == IRNodeKind::LOOP_NEST &&
                 ctx->getLoopDepth() + 2 <= gen_policy->loop_depth_limit) {
            new_stmt = LoopNestStmt::generateStructure(ctx);
        }
        else if (stmt_kind == IRNodeKind::IF_ELSE &&
                 ctx->getIfElseDepth() + 1 <= gen_policy->if_else_depth_limit)
            new_stmt = IfElseStmt::generateStructure(ctx);
        else if (stmt_kind == IRNodeKind::SWITCH &&
                 ctx->getIfElseDepth() + 1 <= gen_policy->if_else_depth_limit)
            new_stmt = SwitchStmt::generateStructure(ctx);
        else {
            new_stmt = StubStmt::generateStructure(ctx);
            stats.addStmt();
        }
        stmts.push_back(new_stmt);
    }

    return std::make_shared<StmtBlock>(stmts);
}

void StmtBlock::populate(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();

    for (auto &stmt : stmts) {
        if (stmt->getKind() != IRNodeKind::STUB)
            stmt->populate(ctx);
        else
            stmt = ExprStmt::create(ctx);
    }
}

void ScopeStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                     std::string offset) {
    stream << offset << "{\n";
    StmtBlock::emit(ctx, stream, offset + "    ");
    stream << offset << "}\n";
}

std::shared_ptr<ScopeStmt>
ScopeStmt::generateStructure(std::shared_ptr<GenCtx> ctx) {
    // TODO: will that work?
    auto new_scope = std::make_shared<ScopeStmt>();
    auto stmt_block = StmtBlock::generateStructure(std::move(ctx));
    new_scope->stmts = stmt_block->getStmts();
    return new_scope;
}

void LoopHead::emitPrefix(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          std::string offset) {
    if (prefix.use_count() != 0)
        prefix->emit(ctx, stream, std::move(offset));
}

// Emit one of an iterator's parameter expressions (start / end / step). Unless
// --expl-loop-params was requested we emit the expression plus its value as a
// comment; with the option we emit only the value.
static void emitIterParam(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          const std::shared_ptr<Expr> &expr) {
    Options &options = Options::getInstance();
    if (!options.getExplLoopParams())
        expr->emit(ctx, stream);
    EvalCtx eval_ctx;
    // TODO: do we want to recalculate it every time?
    auto eval_res = expr->evaluate(eval_ctx);
    assert(eval_res->isScalarVar() && "Iterator should have a scalar value");
    IRValue val =
        std::static_pointer_cast<ScalarVar>(eval_res)->getCurrentValue();
    if (!options.getExplLoopParams())
        stream << "/*";
    stream << val;
    if (!options.getExplLoopParams())
        stream << "*/";
}

// The concrete value of an iterator parameter, for the arithmetic that sizes a
// break/continue guard.
static int64_t iterParamValue(const std::shared_ptr<Expr> &expr) {
    EvalCtx eval_ctx;
    auto eval_res = expr->evaluate(eval_ctx);
    assert(eval_res->isScalarVar() && "Iterator should have a scalar value");
    IRValue val =
        std::static_pointer_cast<ScalarVar>(eval_res)->getCurrentValue();
    auto abs_val = val.getAbsValue();
    return abs_val.isNegative ? -static_cast<int64_t>(abs_val.value)
                              : static_cast<int64_t>(abs_val.value);
}

void LoopHead::emitCondition(std::shared_ptr<EmitCtx> ctx,
                             std::ostream &stream) {
    for (auto iter = iters.begin(); iter != iters.end(); ++iter) {
        stream << (*iter)->getName(ctx) << " < ";
        emitIterParam(ctx, stream, (*iter)->getEnd());
        if (iter != iters.end() - 1)
            stream << ", ";
    }
}

void LoopHead::emitIterDecls(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                             std::string offset) {
    for (const auto &iter : iters) {
        stream << offset << iter->getType()->getName(ctx) << " "
               << iter->getName(ctx) << " = ";
        emitIterParam(ctx, stream, iter->getStart());
        stream << ";\n";
    }
}

void LoopHead::emitIterSteps(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                             std::string offset) {
    for (const auto &iter : iters) {
        stream << offset << iter->getName(ctx) << " += ";
        emitIterParam(ctx, stream, iter->getStep());
        stream << ";\n";
    }
}

void LoopHead::emitHeader(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          std::string offset) {
    if (vectorizable)
        stream << offset << "/* vectorizable */\n";

    auto emit_pragmas = [this, &ctx, &stream, &offset]() {
        for (auto &pragma : pragmas) {
            pragma->emit(ctx, stream, offset);
            stream << "\n";
        }
    };

    // A while/do-while form has to hoist the iterator declarations out of the
    // loop, so it needs an enclosing scope to keep their names local.
    if (form != LoopForm::FOR) {
        stream << offset << "{\n";
        emitIterDecls(ctx, stream, offset);
        // A loop pragma has to sit immediately in front of the loop it applies
        // to, so it goes after the hoisted declarations, not before the scope.
        emit_pragmas();
        stream << offset;
        if (form == LoopForm::WHILE) {
            stream << "while (";
            emitCondition(ctx, stream);
            stream << ") ";
        }
        else {
            stream << "do ";
        }
        if (same_iter_space)
            stream << "/* same iter space */";
        return;
    }

    emit_pragmas();
    stream << offset;

    auto place_sep = [this](auto iter, std::string sep) -> std::string {
        return iter != iters.end() - 1 ? std::move(sep) : "";
    };

    if (!isForeach()) {
        stream << "for (";

        for (auto iter = iters.begin(); iter != iters.end(); ++iter) {
            stream << (*iter)->getType()->getName(ctx) << " ";
            stream << (*iter)->getName(ctx) << " = ";
            emitIterParam(ctx, stream, (*iter)->getStart());
            stream << place_sep(iter, ", ");
        }
        stream << "; ";

        emitCondition(ctx, stream);
        stream << "; ";

        for (auto iter = iters.begin(); iter != iters.end(); ++iter) {
            stream << (*iter)->getName(ctx) << " += ";
            emitIterParam(ctx, stream, (*iter)->getStep());
            stream << place_sep(iter, ", ");
        }
        stream << ") ";
    }
    else {
        stream << (iters.size() == 1 ? "foreach" : "foreach_tiled") << "(";

        for (auto iter = iters.begin(); iter != iters.end(); ++iter) {
            stream << (*iter)->getName(ctx) << " = (";
            emitIterParam(ctx, stream, (*iter)->getStart());
            stream << ")...(";
            emitIterParam(ctx, stream, (*iter)->getEnd());
            stream << ")";
            stream << place_sep(iter, ", ");
        }

        stream << ") ";
    }

    if (same_iter_space)
        stream << "/* same iter space */";
}

void LoopHead::emitBodyStart(std::shared_ptr<EmitCtx> ctx,
                            std::ostream &stream, std::string offset) {
    if (jump_kind == LoopJumpKind::NONE)
        return;

    auto iter = iters.front();
    auto int_type = std::static_pointer_cast<IntegralType>(iter->getType());
    IRValue bound(int_type->getIntTypeId(),
                  IRValue::AbsValue{jump_bound < 0,
                                    static_cast<uint64_t>(
                                        jump_bound < 0 ? -jump_bound
                                                       : jump_bound)});
    auto bound_expr = std::make_shared<ConstantExpr>(bound);

    stream << offset << "if (" << iter->getName(ctx);
    // BREAK leaves once the iterator reaches the bound; both continue kinds
    // skip while it is still below it.
    stream << (jump_kind == LoopJumpKind::BREAK ? " >= " : " < ");
    stream << "(";
    bound_expr->emit(ctx, stream);
    stream << ")";
    if (hide_jump_bound)
        stream << " + zero";
    stream << ") " << (jump_kind == LoopJumpKind::BREAK ? "break" : "continue")
           << ";\n";
}

void LoopHead::emitBodyEnd(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                           std::string offset) {
    // Only the while/do-while forms carry their update in the body.
    if (form == LoopForm::FOR)
        return;
    emitIterSteps(ctx, stream, std::move(offset));
}

void LoopHead::emitFooter(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          std::string offset) {
    if (form == LoopForm::FOR)
        return;
    if (form == LoopForm::DO_WHILE) {
        stream << offset << "while (";
        emitCondition(ctx, stream);
        stream << ");\n";
    }
    // Close the scope that emitHeader() opened for the hoisted declarations.
    stream << offset << "}\n";
}

void LoopHead::copyFormAndJump(const std::shared_ptr<LoopHead> &other) {
    form = other->form;
    jump_kind = other->jump_kind;
    jump_bound = other->jump_bound;
    hide_jump_bound = other->hide_jump_bound;
}

void LoopHead::chooseFormAndJump(const std::shared_ptr<PopulateCtx> &ctx) {
    Options &options = Options::getInstance();
    if (!options.extendedFeaturesSupported() || isForeach())
        return;
    // A head with several iterators spells its condition and update as comma
    // expressions, which have no while/do-while equivalent, and a guard on one
    // of several iterators is not what we want either.
    if (iters.size() != 1)
        return;

    auto gen_pol = ctx->getGenPolicy();
    auto iter = iters.front();
    size_t total = iter->getTotalItersNum();

    // OpenMP's `simd` construct must be attached to a canonical for loop, and
    // branching out of it is forbidden, so such a loop keeps both the for form
    // and an unguarded body.
    if (hasSIMDPragma())
        return;

    if (options.getLoopForms() != OptionLevel::NONE) {
        LoopForm rolled =
            options.getLoopForms() == OptionLevel::ALL
                ? LoopForm::WHILE
                : rand_val_gen->getRandId(gen_pol->loop_form_distr);
        // do-while runs the body unconditionally once, so it cannot express a
        // loop that must not run at all. A degenerate loop's body is populated
        // as dead code and may even contain UB (--allow-ub-in-dc), so executing
        // it would be a real bug.
        if (rolled == LoopForm::DO_WHILE &&
            (iter->isDegenerate() || total == 0))
            rolled = LoopForm::WHILE;
        form = rolled;
    }

    if (options.getLoopJumps() == OptionLevel::NONE)
        return;
    // Nothing to guard, and the body is dead code.
    if (iter->isDegenerate() || total == 0)
        return;

    LoopJumpKind rolled = rand_val_gen->getRandId(gen_pol->loop_jump_distr);
    if (rolled == LoopJumpKind::NONE)
        return;
    bool is_continue = rolled != LoopJumpKind::BREAK;

    // In a while/do-while form the iterator update sits at the end of the body,
    // and `continue` would jump over it -- an infinite loop. In a `for` the
    // update is part of the header, which `continue` does run.
    if (is_continue && form != LoopForm::FOR)
        return;

    // BREAK and CONT_PREFIX shorten the trip count. Only the default HASH
    // algorithm compares hashes between compilers; the other two embed values
    // that yarpgen predicted, and a shorter loop leaves more of an array
    // holding its initial values than the single-value-per-array model
    // assumes. CONT_NEVER does not touch the trip count, so it stays allowed.
    if (options.getCheckAlgo() != CheckAlgo::HASH &&
        rolled != LoopJumpKind::CONT_NEVER)
        return;

    int64_t start = iterParamValue(iter->getStart());
    int64_t step = iterParamValue(iter->getStep());
    if (step <= 0)
        return;

    hide_jump_bound = rand_val_gen->getRandId(gen_pol->hide_jump_bound_distr);

    if (rolled == LoopJumpKind::BREAK) {
        size_t percent =
            rand_val_gen->getRandId(gen_pol->break_keep_percent_distr);
        size_t keep = (total * percent) / 100;
        keep = std::min(std::max(keep, static_cast<size_t>(1)), total);
        jump_kind = LoopJumpKind::BREAK;
        // The guard fires on the first iteration whose value reaches the bound.
        // keep == total puts the bound one step past the final iteration, so
        // the exit exists but is never taken -- the shape that stresses
        // early-exit vectorization hardest, at no cost to the value model.
        jump_bound = start + static_cast<int64_t>(keep) * step;
        if (keep != total) {
            iter->setTotalItersNum(keep);
            // The final executed iteration moved, so which of an array's two
            // value sets it reads may have moved with it.
            int64_t last = start + static_cast<int64_t>(keep - 1) * step;
            iter->setMainValsOnLastIter(
                static_cast<size_t>(last) % Options::vals_number ==
                Options::main_val_idx);
        }
    }
    else if (rolled == LoopJumpKind::CONT_NEVER) {
        jump_kind = LoopJumpKind::CONT_NEVER;
        // `i < start` is false on every iteration. This only earns its keep
        // when the bound is opaque; otherwise the compiler deletes the guard.
        jump_bound = start;
        hide_jump_bound = true;
    }
    else {
        size_t skip = rand_val_gen->getRandId(gen_pol->cont_skip_num_distr);
        // Keep at least the final iteration, so that the last write to every
        // scalar target is the one the value model predicted.
        if (skip >= total)
            skip = total - 1;
        if (skip == 0)
            return;
        jump_kind = LoopJumpKind::CONT_PREFIX;
        jump_bound = start + static_cast<int64_t>(skip) * step;
        iter->setTotalItersNum(total - skip);
    }
}

void LoopHead::emitSuffix(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                          std::string offset) {
    if (suffix.use_count() != 0)
        suffix->emit(ctx, stream, std::move(offset));
}

void LoopHead::createPragmas(std::shared_ptr<PopulateCtx> ctx) {
    Options &options = Options::getInstance();
    if (!options.isCXX() || options.getEmitPragmas() == OptionLevel::NONE)
        return;

    auto gen_pol = ctx->getGenPolicy();
    size_t pragmas_num = rand_val_gen->getRandId(gen_pol->pragma_num_distr);
    if (options.getEmitPragmas() == OptionLevel::ALL)
        pragmas_num = static_cast<size_t>(PragmaKind::MAX_PRAGMA_KIND) - 1;
    pragmas = Pragma::create(pragmas_num, ctx);
}

bool LoopHead::hasSIMDPragma() {
    auto search_func = [](std::shared_ptr<Pragma> pragma) -> bool {
        return pragma->getKind() == PragmaKind::OMP_SIMD;
    };

    return std::find_if(pragmas.begin(), pragmas.end(), search_func) !=
           pragmas.end();
}

void LoopHead::populateArrays(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    size_t new_arrays_num = rand_val_gen->getRandId(gen_pol->new_arr_num_distr);
    for (size_t i = 0; i < new_arrays_num; ++i) {
        ctx->getExtInpSymTable()->addArray(Array::create(ctx, true));
    }
}

std::shared_ptr<Iterator>
LoopHead::populateIterators(std::shared_ptr<PopulateCtx> ctx, size_t _end_val) {
    auto gen_pol = ctx->getGenPolicy();
    auto new_iter =
        Iterator::create(ctx, _end_val, /*is_uniform*/ !isForeach());
    new_iter->setIsDead(false);
    new_iter->populate(ctx);
    addIterator(new_iter);
    return new_iter;
}

void LoopSeqStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                       std::string offset) {
    stream << offset << "/* LoopSeq " << std::to_string(loops.size())
           << " */\n";

    for (const auto &loop : loops) {
        loop.first->emitPrefix(ctx, stream, offset);
        loop.first->emitHeader(ctx, stream, offset);
        stream << "\n";
        // We emit the body's braces here rather than letting the ScopeStmt do
        // it, because the loop needs to place its break/continue guard and (for
        // the while/do-while forms) its iterator update inside them.
        stream << offset << "{\n";
        loop.first->emitBodyStart(ctx, stream, offset + "    ");
        loop.second->emitBody(ctx, stream, offset + "    ");
        loop.first->emitBodyEnd(ctx, stream, offset + "    ");
        stream << offset << "}\n";
        loop.first->emitFooter(ctx, stream, offset);
        loop.first->emitSuffix(ctx, stream, offset);
    }
}

std::shared_ptr<LoopSeqStmt>
LoopSeqStmt::generateStructure(std::shared_ptr<GenCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    size_t loop_num = rand_val_gen->getRandId(gen_pol->loop_seq_num_distr);

    Options &options = Options::getInstance();

    auto new_loop_seq = std::make_shared<LoopSeqStmt>();
    auto new_ctx = std::make_shared<GenCtx>(*ctx);
    // TODO: is it the right place to do it?
    new_ctx->incLoopDepth(1);
    for (size_t i = 0; i < loop_num; ++i) {
        bool gen_foreach = false;
        auto new_loop_head = std::make_shared<LoopHead>();

        if (options.isISPC())
            gen_foreach = !ctx->isInsideForeach() &&
                          rand_val_gen->getRandId(gen_pol->foreach_distr);
        if (gen_foreach) {
            new_ctx->setInsideForeach(true);
            new_loop_head->setIsForeach(true);
        }

        auto new_loop_body = ScopeStmt::generateStructure(new_ctx);
        new_loop_seq->addLoop(std::make_pair(new_loop_head, new_loop_body));

        if (gen_foreach)
            new_ctx->setInsideForeach(false);
    }

    Statistics &stats = Statistics::getInstance();
    stats.addStmt(loop_num);

    return new_loop_seq;
}

// Auxiliary mutation function
// It checks mutation conditions and performs one when it is possible (with some
// probability).
template <typename F>
static auto makeMutableRoll(std::shared_ptr<GenPolicy> gen_pol,
                            F function_call) {
    auto res = function_call();
    Options &options = Options::getInstance();
    if (options.getMutationKind() == MutationKind::ALL) {
        rand_val_gen->switchMutationStates();
        bool mutate = rand_val_gen->getRandId(gen_pol->mutation_probability);
        if (mutate)
            res = function_call();
        rand_val_gen->switchMutationStates();
    }
    return res;
}

void LoopSeqStmt::populate(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();

    size_t same_iter_space_counter = 0;
    size_t same_iter_space_dim = 0;
    size_t cur_idx = 0;

    for (auto &loop : loops) {
        auto active_gen_pol = gen_pol;

        auto &loop_head = loop.first;
        if (loop_head->getPrefix().use_count() != 0)
            loop_head->getPrefix()->populate(ctx);

        auto new_ctx = std::make_shared<PopulateCtx>(ctx);

        bool vectorizable_loop =
            rand_val_gen->getRandId(gen_pol->vectorizable_loop_distr);
        if (vectorizable_loop) {
            active_gen_pol = std::make_shared<GenPolicy>(*gen_pol);
            active_gen_pol->makeVectorizable();
            loop_head->setVectorizable();
            new_ctx->setGenPolicy(active_gen_pol);
        }

        loop_head->createPragmas(new_ctx);
        bool old_simd_state = new_ctx->isInsideOMPSimd();
        new_ctx->setInsideOMPSimd(loop_head->hasSIMDPragma() || old_simd_state);

        size_t new_dim = 0;

        std::shared_ptr<Iterator> new_iters = nullptr;
        if (same_iter_space_counter == 0) {
            if (new_ctx->getDimensions().empty()) {
                new_dim = makeMutableRoll(active_gen_pol, [&active_gen_pol]() {
                    return rand_val_gen->getRandValue(
                        active_gen_pol->iters_end_limit_min,
                        active_gen_pol->iter_end_limit_max);
                });
                Options &options = Options::getInstance();
                if (options.isISPC() && detectNestedForeach())
                    new_dim = std::max(gen_pol->ispc_iter_end_limit_max,
                                       (new_dim / ISPC_MAX_VECTOR_SIZE) *
                                               ISPC_MAX_VECTOR_SIZE +
                                           gen_pol->max_stencil_span);
            }
            else
                new_dim = new_ctx->getDimensions().front();

            new_iters = loop_head->populateIterators(new_ctx, new_dim);
            // Must run after createPragmas() (an omp simd loop may not be
            // branched out of) and before the body is populated, because
            // shortening the trip count changes the values that reductions in
            // the body evaluate to.
            loop_head->chooseFormAndJump(new_ctx);
        }
        else {
            new_dim = same_iter_space_dim;
            auto prev_loop = loops.at(cur_idx - 1);
            auto prev_iter = prev_loop.first->getIterators().front();
            NameHandler &nh = NameHandler::getInstance();
            new_iters = std::make_shared<Iterator>(
                nh.getIterName(), prev_iter->getType(), prev_iter->getStart(),
                prev_iter->getMaxLeftOffset(), prev_iter->getEnd(),
                prev_iter->getMaxRightOffset(), prev_iter->getStep(),
                prev_iter->isDegenerate(), prev_iter->getTotalItersNum());
            new_iters->setIsDead(false);
            new_iters->setSupportsMulValues(prev_iter->getSupportsMulValues());
            new_iters->setMainValsOnLastIter(
                prev_iter->getMainValsOnLastIter());

            // Foreach loop requires the iterator to have a step of 1 and
            // ISPC does not support nested foreach loops. Therefore,
            // this is a hack that allows us to generate foreach loop
            // that satisfy these requirements.
            loop_head->setIsForeach(prev_loop.first->isForeach() &&
                                    loop_head->isForeach());

            loop_head->addIterator(new_iters);
            // new_iters was cloned from prev_iter, trip count included. That
            // count is only correct if this loop carries the same guard.
            loop_head->copyFormAndJump(prev_loop.first);
            loop_head->setSameIterSpace();
            same_iter_space_counter--;
        }

        bool body_with_mul_vals =
            new_ctx->getMulValsIter() == nullptr &&
            new_iters->getSupportsMulValues() &&
            rand_val_gen->getRandId(gen_pol->loop_body_with_mul_vals_prob);
        if (body_with_mul_vals) {
            new_ctx->setMulValsIter(new_iters);
            new_ctx->setAllowMulVals(true);
        }

        new_ctx->addDimension(new_dim);
        LoopHead::populateArrays(new_ctx);

        new_ctx->getLocalSymTable()->addIters(new_iters);

        if (same_iter_space_counter == 0 &&
            rand_val_gen->getRandId(active_gen_pol->same_iter_space)) {
            same_iter_space_counter =
                std::min(loops.size() - cur_idx,
                         rand_val_gen->getRandId(
                             active_gen_pol->same_iter_space_span)) -
                1;
            same_iter_space_dim = new_dim;
            if (same_iter_space_counter > 0)
                loop_head->setSameIterSpace();
        }

        new_ctx->incLoopDepth(1);
        bool old_ctx_state = new_ctx->isTaken();
        // TODO: what if we have multiple iterators
        if (loop_head->getIterators().front()->isDegenerate())
            new_ctx->setTaken(false);
        new_ctx->setInsideForeach(loop_head->isForeach() ||
                                  ctx->isInsideForeach());

        loop.second->populate(new_ctx);

        // TODO: we create new context for each loop, so some of the cleanup is
        // not needed
        new_ctx->decLoopDepth(1);
        new_ctx->getLocalSymTable()->deleteLastIters();
        new_ctx->deleteLastDim();
        new_ctx->setInsideForeach(ctx->isInsideForeach());
        new_ctx->setTaken(old_ctx_state);
        new_ctx->setInsideOMPSimd(old_simd_state);
        if (loop_head->getSuffix().use_count() != 0)
            loop_head->getSuffix()->populate(new_ctx);

        ++cur_idx;
    }
}

void LoopNestStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                        std::string offset) {
    stream << offset << "/* LoopNest " << std::to_string(loops.size())
           << " */\n";

    std::string new_offset = offset;
    for (const auto &loop : loops) {
        loop->emitPrefix(ctx, stream, new_offset);
        loop->emitHeader(ctx, stream, new_offset);
        stream << "\n" << new_offset << "{\n";
        // The guard belongs to this loop, so it goes before the next header.
        loop->emitBodyStart(ctx, stream, new_offset + "    ");
        new_offset += "    ";
    }

    body->emit(ctx, stream, new_offset);

    // Close from the innermost loop outwards. Each loop's iterator update has
    // to come after every loop nested inside it has been closed, so this walks
    // the loops in reverse -- the previous forward walk paired each closing
    // brace with the wrong LoopHead.
    for (auto loop = loops.rbegin(); loop != loops.rend(); ++loop) {
        (*loop)->emitBodyEnd(ctx, stream, new_offset);
        new_offset.erase(new_offset.size() - 4, 4);
        stream << new_offset << "} \n";
        (*loop)->emitFooter(ctx, stream, new_offset);
        (*loop)->emitSuffix(ctx, stream, new_offset);
    }
}

std::shared_ptr<LoopNestStmt>
LoopNestStmt::generateStructure(std::shared_ptr<GenCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    size_t nest_depth = rand_val_gen->getRandId(gen_pol->loop_nest_depth_distr);

    // We need to limit a maximal loop depth that we can generate
    nest_depth =
        std::min(gen_pol->loop_depth_limit - ctx->getLoopDepth(), nest_depth);

    Options &options = Options::getInstance();

    auto new_loop_nest = std::make_shared<LoopNestStmt>();
    auto new_ctx = std::make_shared<GenCtx>(*ctx);
    for (size_t i = 0; i < nest_depth; ++i) {
        auto new_loop = std::make_shared<LoopHead>();

        bool gen_foreach = false;
        if (options.isISPC())
            gen_foreach = !new_ctx->isInsideForeach() &&
                          rand_val_gen->getRandId(gen_pol->foreach_distr);
        if (gen_foreach) {
            new_loop->setIsForeach(true);
            new_ctx->setInsideForeach(true);
        }

        new_loop_nest->addLoop(new_loop);
    }

    Statistics &stats = Statistics::getInstance();
    stats.addStmt(nest_depth);

    new_ctx->incLoopDepth(nest_depth);
    new_loop_nest->addBody(ScopeStmt::generateStructure(new_ctx));

    return new_loop_nest;
}

void LoopNestStmt::populate(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    auto new_ctx = std::make_shared<PopulateCtx>(ctx);
    bool old_ctx_state = new_ctx->isTaken();
    auto taken_switch_id = loops.end();
    auto simd_switch_id = loops.end();
    auto mul_val_loop_idx = loops.end();
    for (auto i = loops.begin(); i != loops.end(); ++i) {
        if ((*i)->getPrefix().use_count() != 0) {
            (*i)->getPrefix()->populate(new_ctx);
            taken_switch_id = i;
        }

        (*i)->createPragmas(new_ctx);
        if (simd_switch_id == loops.end() && (*i)->hasSIMDPragma()) {
            new_ctx->setInsideOMPSimd(true);
            simd_switch_id = i;
        }

        size_t new_dim = 0;
        if (new_ctx->getDimensions().empty()) {
            new_dim = makeMutableRoll(gen_pol, [&gen_pol]() {
                return rand_val_gen->getRandValue(gen_pol->iters_end_limit_min,
                                                  gen_pol->iter_end_limit_max);
            });
            Options &options = Options::getInstance();
            if (options.isISPC() && detectNestedForeach())
                new_dim = std::max(gen_pol->ispc_iter_end_limit_max,
                                   (new_dim / ISPC_MAX_VECTOR_SIZE) *
                                           ISPC_MAX_VECTOR_SIZE +
                                       gen_pol->max_stencil_span);
        }
        else
            new_dim = new_ctx->getDimensions().front();

        auto new_iters = (*i)->populateIterators(new_ctx, new_dim);
        (*i)->chooseFormAndJump(new_ctx);

        bool body_with_mul_vals =
            new_ctx->getMulValsIter() == nullptr &&
            new_iters->getSupportsMulValues() &&
            rand_val_gen->getRandId(gen_pol->loop_body_with_mul_vals_prob);
        if (body_with_mul_vals) {
            new_ctx->setMulValsIter(new_iters);
            new_ctx->setAllowMulVals(true);
            mul_val_loop_idx = i;
        }

        new_ctx->addDimension(new_dim);
        LoopHead::populateArrays(new_ctx);

        new_ctx->getLocalSymTable()->addIters(new_iters);

        new_ctx->incLoopDepth(1);
        new_ctx->setInsideForeach((*i)->isForeach() ||
                                  new_ctx->isInsideForeach());
        if ((*i)->getIterators().front()->isDegenerate())
            new_ctx->setTaken(false);
    }

    body->populate(new_ctx);

    for (auto i = loops.begin(); i != loops.end(); ++i) {
        new_ctx->decLoopDepth(1);
        new_ctx->getLocalSymTable()->deleteLastIters();
        new_ctx->deleteLastDim();
        if (i == mul_val_loop_idx) {
            new_ctx->setMulValsIter(nullptr);
            new_ctx->setAllowMulVals(false);
        }
        if (i == taken_switch_id)
            new_ctx->setTaken(old_ctx_state);
        if (i == simd_switch_id)
            new_ctx->setInsideOMPSimd(ctx->isInsideOMPSimd());
        if ((*i)->isForeach())
            new_ctx->setInsideForeach(ctx->isInsideForeach());
        if ((*i)->getSuffix().use_count() != 0)
            (*i)->getSuffix()->populate(new_ctx);
    }
}

void IfElseStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                      std::string offset) {
    stream << offset << "if (";
    // We can dump test structure before populating it
    if (cond.use_count() != 0)
        cond->emit(ctx, stream);
    stream << ")\n";
    then_br->emit(ctx, stream, offset);
    if (else_br.use_count() != 0) {
        stream << offset << "else\n";
        else_br->emit(ctx, stream, offset);
    }
}

std::shared_ptr<IfElseStmt>
IfElseStmt::generateStructure(std::shared_ptr<GenCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    auto new_ctx = std::make_shared<GenCtx>(*ctx);
    new_ctx->incIfElseDepth();
    auto then_br = ScopeStmt::generateStructure(new_ctx);
    bool else_br_exist = rand_val_gen->getRandId(gen_pol->else_br_distr);
    std::shared_ptr<ScopeStmt> else_br;
    if (else_br_exist)
        else_br = ScopeStmt::generateStructure(new_ctx);

    Statistics &stats = Statistics::getInstance();
    stats.addStmt();

    return std::make_shared<IfElseStmt>(nullptr, then_br, else_br);
}

void IfElseStmt::populate(std::shared_ptr<PopulateCtx> ctx) {
    auto new_ctx = std::make_shared<PopulateCtx>(ctx);
    new_ctx->setAllowMulVals(false);
    // TODO: for now, we do not allow multiple if-else statements' conditions
    // this leads to divergent taken branches and is incompatible with
    // the current implementation of value tracking
    cond = ArithmeticExpr::create(new_ctx);

    // ArithmeticExpr::create() only propagates types as part of eliminating UB,
    // and it skips both for dead code when --allow-ub-in-dc permits UB there.
    // Without this the condition has no value at all and the check below
    // dereferences null.
    cond->propagateType();

    if (!cond->getValue()->isScalarVar() && !cond->getValue()->isTypedData()) {
        ERROR("Can perform conversion to bool only on scalar variables");
    }

    std::shared_ptr<IntegralType> int_type =
        std::static_pointer_cast<IntegralType>(cond->getValue()->getType());
    if (int_type->getIntTypeId() != IntTypeID::BOOL) {
        cond = std::make_shared<TypeCastExpr>(
            cond,
            IntegralType::init(IntTypeID::BOOL, false, CVQualifier::NONE,
                               cond->getValue()->getType()->isUniform()),
            true);
    }

    EvalCtx eval_ctx;
    std::shared_ptr<Data> cond_eval_res = cond->evaluate(eval_ctx);
    IRValue cond_val =
        std::static_pointer_cast<ScalarVar>(cond_eval_res)->getCurrentValue();

    new_ctx = std::make_shared<PopulateCtx>(*ctx);
    new_ctx->incIfElseDepth();
    bool cond_taken = cond_val.getValueRef<bool>();
    new_ctx->setTaken(ctx->isTaken() && cond_taken);

    then_br->populate(new_ctx);
    if (else_br.use_count() != 0) {
        new_ctx->setTaken(ctx->isTaken() && !cond_taken);
        else_br->populate(new_ctx);
    }
}

// Produce `base + delta` as a value of type_id, or report failure if it would
// leave the type's range.
//
// This has to be done in plain integers: IRValue's arithmetic operators are
// only defined for the types that survive integral promotion (int and wider),
// because everywhere else in the generator they are applied to
// already-promoted operands. Case labels are generated in the condition's own
// (possibly narrow) type so that they are trivially in range of the promoted
// type used for the comparison.
static bool denseLabel(IntTypeID type_id, IRValue base, size_t delta,
                       IRValue &out) {
    auto int_type = IntegralType::init(type_id);
    auto to_signed = [](IRValue v) {
        auto abs_val = v.getAbsValue();
        return abs_val.isNegative ? -static_cast<int64_t>(abs_val.value)
                                  : static_cast<int64_t>(abs_val.value);
    };

    if (int_type->getIsSigned()) {
        int64_t hi = to_signed(int_type->getMax());
        int64_t val = to_signed(base);
        if (val > hi - static_cast<int64_t>(delta))
            return false;
        val += static_cast<int64_t>(delta);
        out = IRValue(type_id,
                      IRValue::AbsValue{val < 0, static_cast<uint64_t>(
                                                     val < 0 ? -val : val)});
    }
    else {
        uint64_t hi = int_type->getMax().getAbsValue().value;
        uint64_t val = base.getAbsValue().value;
        if (val > hi - delta)
            return false;
        out = IRValue(type_id, IRValue::AbsValue{false, val + delta});
    }
    out.setUBCode(UBKind::NoUB);
    return true;
}

std::shared_ptr<SwitchStmt>
SwitchStmt::generateStructure(std::shared_ptr<GenCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    auto new_ctx = std::make_shared<GenCtx>(*ctx);
    // A switch nests like an if-else, so it shares the same depth budget.
    new_ctx->incIfElseDepth();

    auto ret = std::make_shared<SwitchStmt>();
    size_t case_num = rand_val_gen->getRandId(gen_pol->switch_case_num_distr);
    for (size_t i = 0; i < case_num; ++i) {
        CaseBlock block;
        block.body = ScopeStmt::generateStructure(new_ctx);
        ret->cases.push_back(block);
    }
    if (rand_val_gen->getRandId(gen_pol->switch_has_default_distr))
        ret->default_br = ScopeStmt::generateStructure(new_ctx);

    Statistics &stats = Statistics::getInstance();
    stats.addStmt();

    return ret;
}

void SwitchStmt::populate(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();

    auto cond_ctx = std::make_shared<PopulateCtx>(ctx);
    // As for if-else, a divergent controlling value would give the two value
    // sets different taken branches, which the value tracking cannot express.
    cond_ctx->setAllowMulVals(false);
    cond = ArithmeticExpr::create(cond_ctx);
    // See IfElseStmt::populate: create() leaves the type unpropagated when it
    // is allowed to leave UB in dead code, so do it here before inspecting the
    // condition.
    cond->propagateType();

    EvalCtx cond_eval_ctx;
    auto cond_eval_res = cond->evaluate(cond_eval_ctx);
    if (!cond_eval_res->isScalarVar())
        ERROR("Switch condition has to be a scalar variable");
    auto cond_int_type =
        std::static_pointer_cast<IntegralType>(cond_eval_res->getType());

    // The controlling expression undergoes integral promotion, and every label
    // is converted to the promoted type. Generating the labels in the
    // condition's own type keeps them trivially in range of that type, so no
    // label can be out of range or collide after conversion. bool would leave
    // only two usable labels, so promote it away.
    IntTypeID label_type_id = cond_int_type->getIntTypeId();
    if (label_type_id == IntTypeID::BOOL) {
        label_type_id = IntTypeID::INT;
        cond = std::make_shared<TypeCastExpr>(
            cond, IntegralType::init(IntTypeID::INT), /*is_implicit*/ true);
        cond->propagateType();
    }

    IRValue cond_val =
        std::static_pointer_cast<ScalarVar>(cond_eval_res)->getCurrentValue();
    if (cond_val.getIntTypeID() != label_type_id)
        cond_val = cond_val.castToType(label_type_id);
    // In dead code the value may carry a UB marker (--allow-ub-in-dc). It is
    // only used here to pick labels, and every body is populated as not taken
    // anyway, so the marker is irrelevant -- but it must not leak into a label.
    cond_val.setUBCode(UBKind::NoUB);

    bool matches = rand_val_gen->getRandId(gen_pol->switch_case_matches_distr);
    // Dense labels are a consecutive run, which compilers lower to a jump
    // table; sparse ones become a comparison chain or a bit test.
    bool dense = rand_val_gen->getRandId(gen_pol->switch_dense_labels_distr);
    // Which case receives the controlling value (only used when matches).
    // switch_case_num_distr never offers fewer than two cases, but guard the
    // subtraction rather than rely on that.
    size_t match_idx = 0;
    if (cases.empty())
        matches = false;
    else
        match_idx = rand_val_gen->getRandValue(static_cast<size_t>(0),
                                               cases.size() - 1);

    // Duplicate case labels are ill-formed, so track what we have used and
    // retry on a collision.
    std::vector<uint64_t> used_labels;
    auto is_used = [&used_labels](uint64_t v) {
        return std::find(used_labels.begin(), used_labels.end(), v) !=
               used_labels.end();
    };
    // Reserve the controlling value either way: as this switch's match, or as a
    // value no label may take so that `default` is what runs.
    used_labels.push_back(cond_val.getAbsValue().value);

    size_t filled = 0;
    for (; filled < cases.size(); ++filled) {
        if (matches && filled == match_idx) {
            cases[filled].label = cond_val;
            continue;
        }
        // A narrow condition type has few values, so pick with bounded retries.
        bool found = false;
        for (size_t attempt = 0; attempt < 64 && !found; ++attempt) {
            IRValue candidate;
            // Fall back to a random label when a dense run would walk off the
            // end of the type.
            if (!dense || attempt != 0 ||
                !denseLabel(label_type_id, cond_val, filled + 1, candidate))
                candidate = rand_val_gen->getRandValue(label_type_id);
            if (candidate.hasUB())
                continue;
            if (is_used(candidate.getAbsValue().value))
                continue;
            candidate.setUBCode(UBKind::NoUB);
            cases[filled].label = candidate;
            used_labels.push_back(candidate.getAbsValue().value);
            found = true;
        }
        if (!found)
            break;
    }
    // Drop any case we could not find a distinct label for.
    if (filled < cases.size()) {
        cases.resize(filled);
        // The match was going to live in a case that no longer exists, so no
        // label matches and `default` runs instead.
        if (match_idx >= filled)
            matches = false;
    }
    if (cases.empty())
        matches = false;

    for (auto &c : cases)
        c.falls_through =
            rand_val_gen->getRandId(gen_pol->switch_fallthrough_distr);
    // Falling out of the last case would just run `default`, which the taken
    // bookkeeping below treats as "no label matched" -- keep the two separate.
    if (!cases.empty())
        cases.back().falls_through = false;

    // Work out which bodies run: control enters at the matching label and keeps
    // going into each following case that its predecessor falls through into.
    bool any_case_taken = false;
    bool entered = false;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (matches && i == match_idx)
            entered = true;
        bool runs = entered;
        any_case_taken |= runs;

        auto case_ctx = std::make_shared<PopulateCtx>(*ctx);
        case_ctx->incIfElseDepth();
        case_ctx->setTaken(ctx->isTaken() && runs);
        cases[i].body->populate(case_ctx);

        // Control leaves the switch here unless this case falls through.
        if (entered && !cases[i].falls_through)
            entered = false;
    }

    if (default_br.use_count() != 0) {
        auto default_ctx = std::make_shared<PopulateCtx>(*ctx);
        default_ctx->incIfElseDepth();
        // `default` runs exactly when no label matched. It cannot be reached by
        // fall-through here because it is always emitted last.
        default_ctx->setTaken(ctx->isTaken() && !any_case_taken);
        default_br->populate(default_ctx);
    }
}

void SwitchStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                      std::string offset) {
    // As for if-else, the test structure can be dumped before it is populated,
    // in which case there is no condition and the labels have no type yet.
    bool populated = cond.use_count() != 0;

    stream << offset << "switch (";
    if (populated)
        cond->emit(ctx, stream);
    stream << ") {\n";

    for (auto &c : cases) {
        stream << offset << "case ";
        if (populated) {
            auto label_expr = std::make_shared<ConstantExpr>(c.label);
            label_expr->emit(ctx, stream);
        }
        stream << ":\n";
        // The body is a compound statement, which keeps any declaration inside
        // it out of the reach of the other labels.
        c.body->emit(ctx, stream, offset + "    ");
        if (c.falls_through)
            stream << offset << "    /* fallthrough */\n";
        else
            stream << offset << "    break;\n";
    }

    if (default_br.use_count() != 0) {
        stream << offset << "default:\n";
        default_br->emit(ctx, stream, offset + "    ");
        stream << offset << "    break;\n";
    }

    stream << offset << "}\n";
}

void StubStmt::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                    std::string offset) {
    stream << offset << text;
}

std::shared_ptr<StubStmt>
StubStmt::generateStructure(std::shared_ptr<GenCtx> ctx) {
    NameHandler &nh = NameHandler::getInstance();
    return std::make_shared<StubStmt>("Stub stmt #" + nh.getStubStmtIdx());
}

void Pragma::emit(std::shared_ptr<EmitCtx> ctx, std::ostream &stream,
                  std::string offset) {
    stream << offset << "#pragma ";
    auto clang_emit_helper = [&stream](std::string name) {
        stream << "clang loop " << name << "(enable)";
    };

    switch (kind) {
        case PragmaKind::CLANG_VECTORIZE:
            clang_emit_helper("vectorize");
            break;
        case PragmaKind::CLANG_INTERLEAVE:
            clang_emit_helper("interleave");
            break;
        case PragmaKind::CLANG_VEC_PREDICATE:
            clang_emit_helper("vectorize_predicate");
            break;
        case PragmaKind::CLANG_UNROLL:
            clang_emit_helper("unroll");
            break;
        case PragmaKind::OMP_SIMD:
            stream << "omp simd";
            break;
        case PragmaKind::MAX_PRAGMA_KIND:
            ERROR("Bad PragmaKind");
    }
}

std::shared_ptr<Pragma> Pragma::create(std::shared_ptr<PopulateCtx> ctx) {
    auto gen_pol = ctx->getGenPolicy();
    PragmaKind pragma_kind =
        rand_val_gen->getRandId(gen_pol->pragma_kind_distr);
    if (pragma_kind == PragmaKind::MAX_PRAGMA_KIND)
        ERROR("Bad PragmaKind");
    return std::make_shared<Pragma>(pragma_kind);
}

std::vector<std::shared_ptr<Pragma>>
Pragma::create(size_t num, std::shared_ptr<PopulateCtx> ctx) {
    std::vector<std::shared_ptr<Pragma>> pragmas;
    pragmas.reserve(num);
    auto tmp_ctx = std::make_shared<PopulateCtx>(*ctx);
    auto tmp_gen_pol = std::make_shared<GenPolicy>(*(tmp_ctx->getGenPolicy()));

    auto modify_disrt = [&tmp_gen_pol](PragmaKind _kind) {
        auto search_func = [&_kind](Probability<PragmaKind> &elem) -> bool {
            return elem.getId() == _kind;
        };
        auto &vec = tmp_gen_pol->pragma_kind_distr;
        vec.erase(std::remove_if(vec.begin(), vec.end(), search_func),
                  vec.end());
    };

    if (ctx->isInsideOMPSimd()) {
        modify_disrt(PragmaKind::OMP_SIMD);
        if (tmp_gen_pol->pragma_kind_distr.empty())
            return {};
    }
    tmp_ctx->setGenPolicy(tmp_gen_pol);

    for (size_t i = 0; i < num; ++i) {
        auto new_pragma = create(tmp_ctx);
        PragmaKind kind =
            std::static_pointer_cast<Pragma>(new_pragma)->getKind();
        // TODO: we need a smarter way to put them in right order
        if (kind == PragmaKind::OMP_SIMD)
            // Clang's pragmas want to be next to a loop
            pragmas.insert(pragmas.begin(), new_pragma);
        else
            pragmas.push_back(new_pragma);
        modify_disrt(kind);
        if (tmp_gen_pol->pragma_kind_distr.empty())
            break;
        tmp_ctx->setGenPolicy(tmp_gen_pol);
    }
    return pragmas;
}
