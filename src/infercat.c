// ----------------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------------
#include "infercat.h"
#include <stdio.h>
#include <math.h>

// ----------------------------------------------------------------------------
static float infercat_relu(float input)
{
  return input < 0 ? 0 : input;
}

// ----------------------------------------------------------------------------
static float infercat_sigmoid(float input)
{
  return 1.0 / (1.0 + expf(-input));
}

// ----------------------------------------------------------------------------
static void infercat_softmax(float** input, int32_t len)
{
  float sum = 0;
  for(int32_t i=0;i<len;i++)
  {
    sum += expf((*input)[i]);
  }

  for(int32_t i=0;i<len;i++)
  {
    (*input)[i] = expf((*input)[i]) / sum;
  }
}

// ----------------------------------------------------------------------------
static void infercat_dense(float* input, InfercatLayer* ptr)
{
  float* weight = (float*)(ptr->weight);
  for(int32_t i=0;i<(ptr->output_size);i++)
  {
    // ...
    float* inp = input;
    float sum = ptr->bias[i];
    for(int32_t j=0;j<(ptr->input_size);j++)
    {
      sum += (*inp++) * (*weight++);
    }

    // ...
    if(ptr->activation == InfercatLayerActivation_RELU)
    {
      sum = infercat_relu(sum);
    }
    else if(ptr->activation == InfercatLayerActivation_SIGMOID)
    {
      sum = infercat_sigmoid(sum);
    }
    else if(ptr->activation == InfercatLayerActivation_SOFTMAX)
    {
      // Do nothing for now
    }
    else
    {
      // ... ?
    }

    // ...
    ptr->output_buffer[i] = sum;
  }

  // ...
  if(ptr->activation == InfercatLayerActivation_SOFTMAX)
  {
    infercat_softmax(&(ptr->output_buffer), ptr->output_size);
  }
}

// ----------------------------------------------------------------------------
void infercat_iterate(
  float* input,
  InfercatLayer** ptr,
  int32_t layerCount,
  float** output,
  int32_t* output_size
)
{
  // ...
  float* layer_input = input;

  // ...
  for(int32_t i=0;i<layerCount;i++)
  {
    if(ptr[i]->type == InfercatLayerType_DENSE)
    {
      infercat_dense(layer_input, ptr[i]);
      layer_input = ptr[i]->output_buffer;
    }
    else
    {
      // ... ?
    }
  }

  // ...
  (*output) = ptr[layerCount - 1]->output_buffer;
  (*output_size) = ptr[layerCount - 1]->output_size;
}
