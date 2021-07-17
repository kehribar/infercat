// ----------------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------------
#ifndef INFERCAT
#define INFERCAT

// ----------------------------------------------------------------------------
#include <stdint.h>

// ----------------------------------------------------------------------------
typedef enum{
  InfercatLayerType_DENSE = 0
}InfercatLayerType;

// ----------------------------------------------------------------------------
typedef enum{
  InfercatLayerActivation_RELU = 0,
  InfercatLayerActivation_SOFTMAX = 1,
  InfercatLayerActivation_SIGMOID = 2
}InfercatLayerActivation;

// ----------------------------------------------------------------------------
typedef struct{
  const float* bias;
  const float* weight;
  float* output_buffer;
  const int32_t input_size;
  const int32_t output_size;
  const InfercatLayerType type;
  const InfercatLayerActivation activation;
}InfercatLayer;

// ----------------------------------------------------------------------------
void infercat_iterate(
  float* input,
  InfercatLayer** ptr, int32_t layerCount,
  float** output, int32_t* output_size
);

// ----------------------------------------------------------------------------
#endif