# Test passes are only linked into the opt tool and are never installed.
add_dataflow_scheduler_dialects_library(DataflowSchedulerDialectsTestKTDF
    TestPipelineBuilder.cpp

  LINK_LIBS PUBLIC
    DataflowSchedulerDialectsKTDFTransforms

  LINK_MLIR_LIBS
    MLIRFuncDialect
    MLIRPass
)
