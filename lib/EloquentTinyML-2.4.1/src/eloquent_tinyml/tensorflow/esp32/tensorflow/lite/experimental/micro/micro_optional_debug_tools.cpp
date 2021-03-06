#if defined(ESP32)
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "Arduino.h"
#include "eloquent_tinyml/tensorflow/esp32/tensorflow/lite/experimental/micro/micro_optional_debug_tools.h"

#include "eloquent_tinyml/tensorflow/esp32/tensorflow/lite/schema/schema_generated.h"
namespace tflite {

std::vector<int> flatbuffersVector2StdVector(
    const flatbuffers::Vector<int32_t>* fVector) {
  std::vector<int> stdVector;
  for (size_t i = 0; i < fVector->size(); i++) {
    stdVector.push_back(fVector->Get(i));
  }
  return stdVector;
}

void PrintIntVector(const std::vector<int>& v) {
  for (const auto& it : v) {
    Serial.printf(" %d", it);
  }
  Serial.printf("\n");
}

void PrintTfLiteIntVector(const TfLiteIntArray* v) {
  if (!v) {
    Serial.printf(" (null)\n");
    return;
  }
  for (int k = 0; k < v->size; k++) {
    Serial.printf(" %d", v->data[k]);
  }
  Serial.printf("\n");
}

const char* TensorTypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "kTfLiteNoType";
    case kTfLiteFloat32:
      return "kTfLiteFloat32";
    case kTfLiteInt32:
      return "kTfLiteInt32";
    case kTfLiteUInt8:
      return "kTfLiteUInt8";
    case kTfLiteInt8:
      return "kTfLiteInt8";
    case kTfLiteInt64:
      return "kTfLiteInt64";
    case kTfLiteString:
      return "kTfLiteString";
    case kTfLiteBool:
      return "kTfLiteBool";
    case kTfLiteInt16:
      return "kTfLiteInt16";
    case kTfLiteComplex64:
      return "kTfLiteComplex64";
    case kTfLiteFloat16:
      return "kTfLiteFloat16";
  }
  return "(invalid)";
}

const char* AllocTypeName(TfLiteAllocationType type) {
  switch (type) {
    case kTfLiteMemNone:
      return "kTfLiteMemNone";
    case kTfLiteMmapRo:
      return "kTfLiteMmapRo";
    case kTfLiteDynamic:
      return "kTfLiteDynamic";
    case kTfLiteArenaRw:
      return "kTfLiteArenaRw";
    case kTfLiteArenaRwPersistent:
      return "kTfLiteArenaRwPersistent";
  }
  return "(invalid)";
}

// Prints a dump of what tensors and what nodes are in the interpreter.
void PrintInterpreterState(MicroInterpreter* interpreter) {
  Serial.printf("Interpreter has %zu tensors and %zu nodes\n",
         interpreter->tensors_size(), interpreter->operators_size());
  Serial.printf("Inputs:");
  PrintIntVector(flatbuffersVector2StdVector(interpreter->inputs()));
  Serial.printf("Outputs:");
  PrintIntVector(flatbuffersVector2StdVector(interpreter->outputs()));
  Serial.printf("\n");

  for (size_t tensor_index = 0; tensor_index < interpreter->tensors_size();
       tensor_index++) {
    TfLiteTensor* tensor = interpreter->tensor(static_cast<int>(tensor_index));
    Serial.printf("Tensor %3zu %-20s %10s %15s %10zu bytes (%4.1f MB) ", tensor_index,
           tensor->name, TensorTypeName(tensor->type),
           AllocTypeName(tensor->allocation_type), tensor->bytes,
           static_cast<double>(tensor->bytes / (1 << 20)));
    PrintTfLiteIntVector(tensor->dims);
  }
  Serial.printf("\n");

  for (size_t node_index = 0; node_index < interpreter->operators_size();
       node_index++) {
    struct pairTfLiteNodeAndRegistration node_and_reg =
        interpreter->node_and_registration(static_cast<int>(node_index));
    const TfLiteNode& node = node_and_reg.node;
    const TfLiteRegistration* reg = node_and_reg.registration;
    if (reg->custom_name != nullptr) {
      Serial.printf("Node %3zu Operator Custom Name %s\n", node_index,
             reg->custom_name);
    } else {
      Serial.printf("Node %3zu Operator Builtin Code %3d %s\n", node_index,
             reg->builtin_code, EnumNamesBuiltinOperator()[reg->builtin_code]);
    }
    Serial.printf("  Inputs:");
    PrintTfLiteIntVector(node.inputs);
    Serial.printf("  Outputs:");
    PrintTfLiteIntVector(node.outputs);
  }
}

}  // namespace tflite

#endif // end of #if defined(ESP32)