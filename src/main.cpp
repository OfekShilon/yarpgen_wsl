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
#include "options.h"
#include "program.h"
#include "utils.h"

#include <iostream>
#include <string>
#include <vector>

using namespace yarpgen;

int main(int argc, char *argv[]) {
    OptionParser::initOptions();
    OptionParser::parse(argc, argv);

    Options &options = Options::getInstance();
    if (options.useHeapViews()) {
        if (options.isISPC() || options.isSYCL()) {
            std::cerr << "ERROR: --alias is supported only for C and C++"
                      << std::endl;
            return 1;
        }
        // Overlapping views make the exact values of some arrays unknown to
        // the generator, so we can only compare hashes between runs
        if (options.getAlias() != AliasLevel::DISJOINT &&
            options.getCheckAlgo() != CheckAlgo::HASH) {
            std::cerr << "ERROR: --alias=stores|loads requires "
                         "--check-algo=hash"
                      << std::endl;
            return 1;
        }
    }

    rand_val_gen = std::make_shared<RandValGen>(options.getSeed());
    options.setSeed(rand_val_gen->getSeed());

    if (options.getMutationKind() == MutationKind::EXPRS ||
        options.getMutationKind() == MutationKind::ALL) {
        rand_val_gen->setMutationSeed(options.getMutationSeed());
    }

    ProgramGenerator new_program;
    new_program.emit();

    return 0;
}
