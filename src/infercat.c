// ----------------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------------
#include "infercat.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// ----------------------------------------------------------------------------
inline float infercat_relu_single(float input)
{
  return input < 0 ? 0 : input;
}

// ----------------------------------------------------------------------------
inline float infercat_sigmoid_single(float input)
{
  return 1.0 / (1.0 + expf(-input));  
}

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
  // 
  // NOTE
  // ====
  // This is nowhere near optimised ...
  // 

  // 
  // MEMORY LAYOUTS
  // ==============
  // Biases:  [out_ch]
  // Images:  [ix, iy, in_ch]
  // Kernels: [kx, ky, in_ch, out_ch]
  // 

  // ...
  const int32_t iDepth = ptr->in_depth;
  const int32_t oDepth = ptr->out_depth;
  const int32_t kWidth = ptr->kernel_width;

  // Start with bias values in output channels
  float* out = ptr->output_buffer;
  const int32_t oSize = ptr->out_width * ptr->out_height;
  for(int32_t i=0;i<oSize;i++)
  {
    for(int32_t j=0;j<oDepth;j++)
    {
      (*out) = ptr->bias[j];
      out++;
    }
  }

  // Reset output pointer
  out = ptr->output_buffer;

  // Do the convolution
  const float* weight = ptr->weight;
  for(int32_t kx=0;kx<kWidth;kx++)
  {
    for(int32_t ky=0;ky<kWidth;ky++)
    {
      for(int32_t i=0;i<iDepth;i++)
      {
        for(int32_t j=0;j<oDepth;j++)
        {
          // TODO: Proper padding handling!
          for(int32_t ox=0;ox<(ptr->out_width);ox++)
          {
            for(int32_t oy=0;oy<(ptr->out_height);oy++)
            {
              const int32_t ind_o = (
                (ox * oDepth * ptr->out_height) +
                (oy * oDepth                  ) + j
              );

              const int32_t ind_i = (
                (((ox * ptr->stride) + kx) * iDepth * ptr->in_height) +
                (((oy * ptr->stride) + ky) * iDepth                 ) + i
              );

              out[ind_o] += (*weight) * input[ind_i];
            }
          }

          // ...
          weight++;
        }
      }
    }
  }

  // Final activation process
  if(ptr->activation == InfercatLayerActivation_RELU)
  {
    infercat_relu(
      ptr->output_buffer, 
      (ptr->out_width * ptr->out_height * ptr->out_depth)
    );
  }
  else if(ptr->activation == InfercatLayerActivation_SIGMOID)
  {
    infercat_sigmoid(
      ptr->output_buffer, 
      (ptr->out_width * ptr->out_height * ptr->out_depth)
    );
  }
}

// ----------------------------------------------------------------------------
static void infercat_maxpooling2d(float* input, InfercatLayer_MAXPOOLING2D* ptr)
{
  // 
  // NOTE
  // ====
  // This is nowhere near optimised ...
  // 

  // Initialise output buffer with 'known low' values
  for(int32_t i=0;i<(ptr->out_width * ptr->out_width * ptr->out_depth);i++)
  {
    ptr->output_buffer[i] = -1e39;
  }

  // ...
  for(int32_t ix=0;ix<(ptr->in_width);ix++)
  {
    for(int32_t iy=0;iy<(ptr->in_height);iy++)
    {
      for(int32_t ch=0;ch<(ptr->in_depth);ch++)
      {
        // ...
        const float pix = (*input);

        // ...
        const int32_t check_ind = (
          ((ix / ptr->pool_width) * ptr->out_depth * ptr->out_width) + 
          ((iy / ptr->pool_width) * ptr->out_depth                 ) + ch
        );

        // ...
        if(pix > (ptr->output_buffer[check_ind]))
        {
          ptr->output_buffer[check_ind] = pix;
        }

        // ...
        input++;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Influenced from rnnoise project
static void infercat_gru(float* input, InfercatLayer_GRU* ptr)
{
  // ...
  #define MAX_NEURONS 128

  // ...
  float z[MAX_NEURONS];
  float r[MAX_NEURONS];
  float h[MAX_NEURONS];

  // ...
  const int32_t M = ptr->in_size;
  const int32_t N = ptr->out_size;
  const int32_t stride = 3 * ptr->out_size;

  // ...
  if(N > MAX_NEURONS)
  {
    return;
  }

  // Update gate, reset gate
  for(int32_t i=0;i<N;i++)
  {
    // ...
    float zsum = ptr->bias[i    ];
    float rsum = ptr->bias[i + N];

    // ...
    const float* inp = input;
    const float* zw = &(ptr->weight[i    ]);
    const float* rw = &(ptr->weight[i + N]);
    for(int32_t j=0;j<M;j++)
    {
      // ...
      zsum += (*zw) * (*inp);
      rsum += (*rw) * (*inp);

      // ...
      inp += 1;
      zw += stride;
      rw += stride;
    }

    // ...
    const float* outp = ptr->output_buffer;;
    const float* zrw = &(ptr->recurrentWeight[i    ]);
    const float* rrw = &(ptr->recurrentWeight[i + N]);
    for(int32_t j=0;j<N;j++)
    {
      // ...
      zsum += (*zrw) * (*outp);
      rsum += (*rrw) * (*outp);

      // ...
      outp += 1;
      zrw += stride;
      rrw += stride;
    }

    // ...
    z[i] = zsum;
    r[i] = rsum;
  }
  
  // ...
  if(ptr->recurrentActivation == InfercatLayerActivation_SIGMOID)
  {
    infercat_sigmoid(z, N);    
    infercat_sigmoid(r, N);    
  }
  else
  {
    // ... ?
    return;
  }

  // Output gate
  for(int32_t i=0;i<N;i++)
  {
    // ...
    float hsum = ptr->bias[i + N + N];

    // ...
    const float* inp = input;
    const float* hw = &(ptr->weight[i + N + N]);
    for(int32_t j=0;j<M;j++)
    {
      // ...
      hsum += (*hw) * (*inp);
      
      // ...
      inp += 1;
      hw += stride;
    }

    // ...
    const float* rp = r;
    const float* outp = ptr->output_buffer;
    const float* hrw = &(ptr->recurrentWeight[i + N + N]);
    for(int32_t j=0;j<N;j++)
    {
      // ...
      hsum += (*hrw) * (*outp) * (*rp);

      // ...
      rp += 1;
      outp += 1;
      hrw += stride;
    }

    // ...
    if(ptr->activation == InfercatLayerActivation_SIGMOID)
    {
      hsum = infercat_sigmoid_single(hsum);
    }
    else if(ptr->activation == InfercatLayerActivation_RELU)
    {
      hsum = infercat_relu_single(hsum);
    }
    else
    {
      return;
    }

    // ...
    h[i] = hsum + (z[i] * (ptr->output_buffer[i] - hsum));
  }

  // ...
  memcpy(ptr->output_buffer, h, sizeof(float) * ptr->out_size);
}

// ----------------------------------------------------------------------------
void infercat_rnnLayersResetMemory(InfercatLayer** ptr, int32_t layerCount)
{
  for(int32_t i=0;i<layerCount;i++)
  {
    if(ptr[i]->type == InfercatLayerType_GRU)
    {
      InfercatLayer_GRU* layer = (InfercatLayer_GRU*)(ptr[i]->mem);
      memset(layer->output_buffer, 0, sizeof(float) * layer->out_size);
    }
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
      infercat_maxpooling2d(layer_input, layer);
      layer_input = layer->output_buffer;
    }
    else if(ptr[i]->type == InfercatLayerType_GRU)
    {
      InfercatLayer_GRU* layer = (InfercatLayer_GRU*)(ptr[i]->mem);
      infercat_gru(layer_input, layer);
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
