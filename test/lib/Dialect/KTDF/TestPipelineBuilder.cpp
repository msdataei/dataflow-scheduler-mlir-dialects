//===-- TestPipelineBuilder.cpp ---------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler MLIR Dialects project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass that drives the mlir::ktdf::PipelineBuilder
// from attributes in the input IR.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/Pass.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

/// Builds one pipeline per function from the ops marked `test.root`, in
/// program order. Every op that has a `test.units` attribute is placed into
/// the stage for those units; all other ops stay outside the pipeline.
struct TestPipelineBuilderPass
    : PassWrapper<TestPipelineBuilderPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPipelineBuilderPass)

  [[nodiscard]] auto getArgument() const -> StringRef final {
    return "test-ktdf-pipeline-builder";
  }
  [[nodiscard]] auto getDescription() const -> StringRef final {
    return "Test the mlir::ktdf::PipelineBuilder";
  }
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<KTDFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    SmallVector<Operation*> roots;
    func.walk([&](Operation* op) {
      if (op->hasAttr("test.root")) {
        roots.push_back(op);
      }
    });
    if (roots.empty()) {
      return;
    }

    auto& dominance = getAnalysis<DominanceInfo>();
    OpBuilder builder(func.getBody().front().getTerminator());
    PipelineBuilder pipeline_builder(builder, func.getLoc());
    const auto place = [](PipelineBuilder& pipeline_builder,
                          Operation* op) -> PipelineBuilder::Placement {
      const auto units = op->getAttr("test.units");
      if (!units) {
        return nullptr;
      }
      return pipeline_builder.tryPlacement(units);
    };
    pipeline_builder.insert(roots, place, dominance);
  }
};

}  // namespace

namespace mlir::ktdf::test {

void registerTestPipelineBuilderPass() {
  PassRegistration<TestPipelineBuilderPass>();
}

}  // namespace mlir::ktdf::test
