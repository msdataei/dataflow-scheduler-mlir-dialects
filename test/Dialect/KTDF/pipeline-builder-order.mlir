// RUN: dataflow-scheduler-dialects-opt -allow-unregistered-dialect --test-ktdf-pipeline-builder %s | FileCheck %s

// The PipelineBuilder keeps the program order of the ops it moves into a stage,
// and creates the FIFO reads and writes of a stage in the order it forwards the
// values. With the default allocator, each slot is its own FIFO, so the slots
// also appear in that order.

// Two loads feed one compute with two results, each consumed by its own store.
// CHECK-LABEL: func.func @two_inputs_two_results
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private

// The loads keep their program order.
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}

// The reads come in the order the compute uses them, and the results are
// written in result order.
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        %[[C:.+]]:2 = "test.compute"(%[[RA]], %[[RB]])
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]]#0, %[[P]]#[[C0_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[C]]#1, %[[P]]#[[C1_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["compute"]}

// The stores keep their program order.
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[R0:.+]] = ktdf.read_from_fifo %[[P]]#[[C0_SLOT]]
// CHECK-NEXT:        %[[R1:.+]] = ktdf.read_from_fifo %[[P]]#[[C1_SLOT]]
// CHECK-NEXT:        "test.store"(%[[R0]]) {test.name = "x"
// CHECK-NEXT:        "test.store"(%[[R1]]) {test.name = "y"
// CHECK-NEXT:      } {applicable_units = ["store"]}
func.func @two_inputs_two_results() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  %c:2 = "test.compute"(%a, %b) {test.units = "compute"} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  "test.store"(%c#0) {test.name = "x", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  "test.store"(%c#1) {test.name = "y", test.root, test.units = "store"} : (tensor<4xf32>) -> ()
  return
}

// A value that reaches the compute through a relay is inserted after the direct
// load. The relay's stage is still moved in front of the stage that uses it, so
// the stages stay in topological order, and the slots keep program order.
// CHECK-LABEL: func.func @relay
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:{{[0-9]+}} = ktdf.private
// CHECK:           ktdf.stage
// CHECK-NEXT:        %[[A:.+]] = "test.load"() {test.name = "a"
// CHECK-NEXT:        %[[B:.+]] = "test.load"() {test.name = "b"
// CHECK-NEXT:        ktdf.write_to_fifo %[[A]], %[[P]]#[[A_SLOT:[0-9]+]]
// CHECK-NEXT:        ktdf.write_to_fifo %[[B]], %[[P]]#[[B_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["load"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RB:.+]] = ktdf.read_from_fifo %[[P]]#[[B_SLOT]]
// CHECK-NEXT:        %[[R:.+]] = "test.relay"(%[[RB]])
// CHECK-NEXT:        ktdf.write_to_fifo %[[R]], %[[P]]#[[R_SLOT:[0-9]+]]
// CHECK-NEXT:      } {applicable_units = ["relay"]}
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[RA:.+]] = ktdf.read_from_fifo %[[P]]#[[A_SLOT]]
// CHECK-NEXT:        %[[RR:.+]] = ktdf.read_from_fifo %[[P]]#[[R_SLOT]]
// CHECK-NEXT:        "test.compute"(%[[RA]], %[[RR]])
// CHECK-NEXT:      } {applicable_units = ["compute"]}
func.func @relay() {
  %a = "test.load"() {test.name = "a", test.units = "load"} : () -> tensor<4xf32>
  %b = "test.load"() {test.name = "b", test.units = "load"} : () -> tensor<4xf32>
  %r = "test.relay"(%b) {test.units = "relay"} : (tensor<4xf32>) -> tensor<4xf32>
  "test.compute"(%a, %r) {test.root, test.units = "compute"} : (tensor<4xf32>, tensor<4xf32>) -> ()
  return
}
