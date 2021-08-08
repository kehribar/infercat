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
  InfercatLayerType_DENSE = 0,
  InfercatLayerType_CONV2D = 1,
  InfercatLayerType_MAXPOOLING2D = 2,
  InfercatLayerType_GRU = 3
}InfercatLayerType;

// ----------------------------------------------------------------------------
typedef enum{
  InfercatLayerActivation_RELU = 0,
  InfercatLayerActivation_SOFTMAX = 1,
  InfercatLayerActivation_SIGMOID = 2,
  InfercatLayerActivation_TANH = 3
}InfercatLayerActivation;

// ----------------------------------------------------------------------------
typedef struct{
  void* mem;
  const InfercatLayerType type;
}InfercatLayer;

// ----------------------------------------------------------------------------
typedef struct{
  const float* bias;
  const float* weight;
  const int32_t input_size;
  const int32_t output_size;
  const InfercatLayerActivation activation;
  float* output_buffer;
}InfercatLayer_DENSE;

// ----------------------------------------------------------------------------
typedef struct{
  const float* bias;
  const float* weight;
  const int32_t stride;
  const int32_t in_width;
  const int32_t in_height;
  const int32_t in_depth;
  const int32_t out_width;
  const int32_t out_height;
  const int32_t out_depth;
  const int32_t kernel_width;
  const InfercatLayerActivation activation;
  float* output_buffer;
}InfercatLayer_CONV2D;

// ----------------------------------------------------------------------------
typedef struct{
  const int32_t stride;
  const int32_t in_width;
  const int32_t in_height;
  const int32_t in_depth;
  const int32_t out_width;
  const int32_t out_height;
  const int32_t out_depth;
  const int32_t pool_width;
  float* output_buffer;
}InfercatLayer_MAXPOOLING2D;

// ----------------------------------------------------------------------------
typedef struct{
  const float* bias;
  const float* weight;
  const float* recurrentWeight;
  const int32_t in_size;
  const int32_t out_size;
  const InfercatLayerActivation activation;
  const InfercatLayerActivation recurrentActivation;
  float* output_buffer;
}InfercatLayer_GRU;

// ----------------------------------------------------------------------------
void infercat_rnnLayersResetMemory(InfercatLayer** ptr, int32_t layerCount);

// ----------------------------------------------------------------------------
void infercat_iterate(
  float* input,
  InfercatLayer** ptr, int32_t layerCount,
  float** output, int32_t* output_size
);

// ----------------------------------------------------------------------------
#endif
