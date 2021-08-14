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
// Taken from rnnoise project
static const float tansig_table[201] = {
  0.000000, 0.039979, 0.079830, 0.119427, 0.158649,
  0.197375, 0.235496, 0.272905, 0.309507, 0.345214,
  0.379949, 0.413644, 0.446244, 0.477700, 0.507977,
  0.537050, 0.564900, 0.591519, 0.616909, 0.641077,
  0.664037, 0.685809, 0.706419, 0.725897, 0.744277,
  0.761594, 0.777888, 0.793199, 0.807569, 0.821040,
  0.833655, 0.845456, 0.856485, 0.866784, 0.876393,
  0.885352, 0.893698, 0.901468, 0.908698, 0.915420,
  0.921669, 0.927473, 0.932862, 0.937863, 0.942503,
  0.946806, 0.950795, 0.954492, 0.957917, 0.961090,
  0.964028, 0.966747, 0.969265, 0.971594, 0.973749,
  0.975743, 0.977587, 0.979293, 0.980869, 0.982327,
  0.983675, 0.984921, 0.986072, 0.987136, 0.988119,
  0.989027, 0.989867, 0.990642, 0.991359, 0.992020,
  0.992631, 0.993196, 0.993718, 0.994199, 0.994644,
  0.995055, 0.995434, 0.995784, 0.996108, 0.996407,
  0.996682, 0.996937, 0.997172, 0.997389, 0.997590,
  0.997775, 0.997946, 0.998104, 0.998249, 0.998384,
  0.998508, 0.998623, 0.998728, 0.998826, 0.998916,
  0.999000, 0.999076, 0.999147, 0.999213, 0.999273,
  0.999329, 0.999381, 0.999428, 0.999472, 0.999513,
  0.999550, 0.999585, 0.999617, 0.999646, 0.999673,
  0.999699, 0.999722, 0.999743, 0.999763, 0.999781,
  0.999798, 0.999813, 0.999828, 0.999841, 0.999853,
  0.999865, 0.999875, 0.999885, 0.999893, 0.999902,
  0.999909, 0.999916, 0.999923, 0.999929, 0.999934,
  0.999939, 0.999944, 0.999948, 0.999952, 0.999956,
  0.999959, 0.999962, 0.999965, 0.999968, 0.999970,
  0.999973, 0.999975, 0.999977, 0.999978, 0.999980,
  0.999982, 0.999983, 0.999984, 0.999986, 0.999987,
  0.999988, 0.999989, 0.999990, 0.999990, 0.999991,
  0.999992, 0.999992, 0.999993, 0.999994, 0.999994,
  0.999994, 0.999995, 0.999995, 0.999996, 0.999996,
  0.999996, 0.999997, 0.999997, 0.999997, 0.999997,
  0.999997, 0.999998, 0.999998, 0.999998, 0.999998,
  0.999998, 0.999998, 0.999999, 0.999999, 0.999999,
  0.999999, 0.999999, 0.999999, 0.999999, 0.999999,
  0.999999, 0.999999, 0.999999, 0.999999, 0.999999,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000,
};

// ----------------------------------------------------------------------------
// Taken from rnnoise project
static inline float tansig_approx(float x)
{
  // ...
  if(!(x < 8))
  {
    return 1;
  }

  // ...
  if(!(x > -8))
  {
    return -1;
  }

  // ...
  float sign = 1;
  if(x < 0)
  {
    x = -x;
    sign = -1;
  }

  // ...
  int32_t i = (int)(floor(0.5 + (25 * x)));
  x -= 0.04 * i;

  // ...
  float y = tansig_table[i];
  float dy = 1.0 - (y * y);

  // ...
  y = y + (x * dy * (1 - (y * x)));
  return (sign * y);
}

// ----------------------------------------------------------------------------
static inline float infercat_relu_single(float input)
{
  return input < 0 ? 0 : input;
}

// ----------------------------------------------------------------------------
static inline float infercat_sigmoid_single(float input)
{
  return (0.5 + (0.5 * tansig_approx(0.5 * input)));
}

// ----------------------------------------------------------------------------
static void infercat_relu(float* input, const int32_t len)
{
  for(int32_t i=0;i<len;i++)
  {
    (*input) = infercat_relu_single(*input);
    input++;
  }
}

// ----------------------------------------------------------------------------
static void infercat_sigmoid(float* input, const int32_t len)
{
  for(int32_t i=0;i<len;i++)
  {
    (*input) = infercat_sigmoid_single(*input);
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
        // TODO: Proper padding handling!
        for(int32_t ox=0;ox<(ptr->out_width);ox++)
        {
          for(int32_t oy=0;oy<(ptr->out_height);oy++)
          {
            const int32_t ind_o = (
              (ox * oDepth * ptr->out_height) +
              (oy * oDepth                  )
            );

            const int32_t ind_i = (
              (((ox * ptr->stride) + kx) * iDepth * ptr->in_height) +
              (((oy * ptr->stride) + ky) * iDepth                 ) + i
            );

            float* op = &(out[ind_o]);
            const float* wp = weight;
            const float iv = input[ind_i];
            for(int32_t j=0;j<oDepth;j++)
            {
              (*op++) += (*wp++) * iv;
            }
          }
        }

        // ...
        weight += oDepth;
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
    ptr->output_buffer[i] = -1e19;
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
