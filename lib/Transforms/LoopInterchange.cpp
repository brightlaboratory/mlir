//===- LoopInterchange.cpp - Code to perform loop interchange --------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <iostream>

using namespace mlir;
using namespace std;

#define DEBUG_TYPE "affine-loop-interchange"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

namespace {
	struct LoopInterchange : public FunctionPass<LoopInterchange> {


		void runOnFunction() override;

		/// Unroll this for op. Returns failure if nothing was done.
		LogicalResult runOnAffineForOp(AffineForOp forOp);
	};
} // end anonymous namespace

void LoopInterchange::runOnFunction() {
	cout << "In LoopInterchange's runOnFunction()" << endl;
	// Gathers all innermost loops through a post order pruned walk.
	struct InnermostLoopGatherer {
		// Store innermost loops as we walk.
		std::vector<AffineForOp> loops;

		void walkPostOrder(FuncOp f) {
			for (auto &b : f)
				walkPostOrder(b.begin(), b.end());
		}

		bool walkPostOrder(Block::iterator Start, Block::iterator End) {
			bool hasInnerLoops = false;
			// We need to walk all elements since all innermost loops need to be
			// gathered as opposed to determining whether this list has any inner
			// loops or not.
			while (Start != End)
				hasInnerLoops |= walkPostOrder(&(*Start++));
			return hasInnerLoops;
		}
		bool walkPostOrder(Operation *opInst) {
			bool hasInnerLoops = false;
			for (auto &region : opInst->getRegions())
				for (auto &block : region)
					hasInnerLoops |= walkPostOrder(block.begin(), block.end());
			if (isa<AffineForOp>(opInst)) {
				if (!hasInnerLoops)
					loops.push_back(cast<AffineForOp>(opInst));
				return true;
			}
			return hasInnerLoops;
		}
	};

	{
		// Store short loops as we walk.
		std::vector<AffineForOp> loops;
		getFunction().walk([&](AffineForOp forOp) {
			loops.push_back(forOp);
		});

		if (loops.size() >= 2) {
			interchangeLoops(loops.at(1), loops.at(0));
		}

		return;
	}
}

/// Unrolls a 'affine.for' op. Returns success if the loop was unrolled,
/// failure otherwise. The default unroll factor is 4.
LogicalResult LoopInterchange::runOnAffineForOp(AffineForOp forOp) {
	cout << "In LoopInterchange's runOnAffineForOp()" << endl;
	return success();
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createLoopInterchangePass() {
	return std::make_unique<LoopInterchange>();
}

static PassRegistration<LoopInterchange> pass("affine-loop-interchange", "Interchange loops");
