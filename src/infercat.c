// ----------------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------------
#include "infercat.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// ----------------------------------------------------------------------------
static float infercat_relu(float* input, const int32_t len)
{
  for(int32_t i=0;i<len;i++)
  {
    (*input) = (*input) < 0 ? 0 : (*input);
    input++;
  }
}

// ----------------------------------------------------------------------------
static float infercat_sigmoid(float* input, const int32_t len)
{
  for(int32_t i=0;i<len;i++)
  {
    (*input) = 1.0 / (1.0 + expf(-(*input)));  
    input++;
  }
}

// ----------------------------------------------------------------------------
static void infercat_softmax(float* input, const int32_t len)
{
  float* ptr = input;

  // ...
  float sum = 0;
  for(int32_t i=0;i<len;i++)
  {
    sum += expf((*ptr));
    ptr++;
  }

  // ...
  ptr = input;
  for(int32_t i=0;i<len;i++)
  {
    (*ptr) = expf(*ptr) / sum;
    ptr++;
  }
}

// ----------------------------------------------------------------------------
static void infercat_dense(float* input, InfercatLayer_DENSE* ptr)
{
  // ...
  const int32_t iSize = ptr->input_size;
  const int32_t oSize = ptr->output_size;

  // ...
  memcpy(ptr->output_buffer, ptr->bias, (oSize * sizeof(float)));

  // ...
  const float* weight = ptr->weight;
  for(int32_t i=0;i<iSize;i++)
  {
    const float inp = (*input);
    float* output = ptr->output_buffer;
    for(int32_t j=0;j<oSize;j++)
    {
      (*output) += (inp * (*weight));
      weight++;
      output++;
    }
    input++;
  }

  // ...
  if(ptr->activation == InfercatLayerActivation_RELU)
  {
    infercat_relu(ptr->output_buffer, oSize);
  }
  else if(ptr->activation == InfercatLayerActivation_SIGMOID)
  {
    infercat_sigmoid(ptr->output_buffer, oSize);
  }
  else if(ptr->activation == InfercatLayerActivation_SOFTMAX)
  {
    infercat_softmax(ptr->output_buffer, oSize);
  }
}

// ----------------------------------------------------------------------------
static void infercat_conv2d(float* input, InfercatLayer_CONV2D* ptr)
{
  // TODO ...
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
      InfercatLayer_DENSE* layer = (InfercatLayer_DENSE*)(ptr[i]->mem);
      infercat_dense(layer_input, layer);
      layer_input = layer->output_buffer;
    }
    else if(ptr[i]->type == InfercatLayerType_CONV2D)
    {
      InfercatLayer_CONV2D* layer = (InfercatLayer_CONV2D*)(ptr[i]->mem);
      infercat_conv2d(layer_input, layer);
      layer_input = layer->output_buffer;
    }
    else if(ptr[i]->type == InfercatLayerType_MAXPOOLING2D)
    {
      // TODO!
    }
    else
    {
      // ... ?
    }
  }

  // ...
  InfercatLayer_DENSE* layer = (InfercatLayer_DENSE*)(ptr[layerCount - 1]->mem);
  (*output) = layer->output_buffer;
  (*output_size) = layer->output_size;
}
