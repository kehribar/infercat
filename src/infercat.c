// ----------------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------------
#include "infercat.h"
#include <stdio.h>
#include <string.h>
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
static void infercat_dense(float* input, InfercatLayer_DENSE* ptr)
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
static void infercat_conv2d_single(
  const float* in, const float* kernel, const float bias,
  float* out, int32_t iw, int32_t ih, int32_t k, int32_t ch, int32_t offset
)
{
  const int32_t ow = iw - k + 1; // output width
  const int32_t oh = ih - k + 1; // output height
  for(int32_t i=0;i<oh;i++)
  {
    for(int32_t j=0;j<ow;j++)
    {
      for(int32_t y=0;y<k;y++)
      {
        for(int32_t x=0;x<k;x++)
        {
          const float img = in[j + x + ((i + y) * iw)];
          const float w = kernel[x + (y * k)];
          out[offset + (ch * (j + (i * ow)))] += img * w;;
        }
      }
      out[offset + (ch * (j + (i * ow)))] += bias;
    }
  }
}

// ----------------------------------------------------------------------------
static void infercat_conv2d(float* input, InfercatLayer_CONV2D* ptr)
{
  // ...
  const int32_t iw = ptr->in_width;
  const int32_t ih = ptr->in_height;
  const int32_t k = ptr->kernel_width;
  const int32_t ch = ptr->out_depth;

  // ...
  float* out = ptr->output_buffer;
  const int32_t out_size = ptr->out_height * ptr->out_width;
  memset(out, 0, sizeof(float) * out_size * ptr->out_depth);

  // ...
  for(int32_t i=0;i<(ptr->out_depth);i++)
  {
    // ...
    const float bias = ptr->bias[i];

    // ...
    for(int32_t j=0;j<(ptr->in_depth);j++)
    {
      // ...
      const int32_t input_size = ptr->in_width * ptr->in_height;
      const float* in = &(input[j * input_size]);

      // ...
      const int32_t kernel_size = ptr->kernel_width * ptr->kernel_width;
      const float* kernel = &(ptr->weight[i * kernel_size]);

      // ...
      infercat_conv2d_single(in, kernel, bias, out, iw, ih, k, ch, i);
    }
  }

  // ...
  for(int32_t x=0;x<(ptr->out_depth * out_size);x++)
  {
    out[x] = infercat_relu(out[x]);
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
      InfercatLayer_MAXPOOLING2D* layer = (InfercatLayer_MAXPOOLING2D*)(ptr[i]->mem);
      // TODO!
      layer_input = layer->output_buffer;
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
