// -----------------------------------------------------------------------------
//
//
// -----------------------------------------------------------------------------
#include <stdio.h>
#include "infercat.h"
#include "mnistReader.h"
#include "mnist_model.h"

// ...
const int32_t data_count = 10000;
const char* labels = "./dataset/t10k-labels-idx1-ubyte";
const char* images = "./dataset/t10k-images-idx3-ubyte";

// ...
int main(int argc, char const *argv[])
{
  // ...
  int32_t error_count = 0;
  int32_t total_count = 0;

  // ...
  if(mnistReader_init(labels, images) != 0)
  {
    printf("\n");
    printf("Dataset open problem ...\n");
    return -1;
  }

  // ...
  for(int32_t i=0;i<data_count;i++)
  {
    // Get another label / image pair from the MNIST database
    uint8_t img_label;
    uint8_t img_raw[784];
    mnistReader_getByIndex(i, img_raw, &img_label);

    // Normalise raw image to [0, 1]
    float img[784];
    for(int32_t j=0;j<784;j++)
    {
      img[j] = (float)(img_raw[j]) / 255.0;
    }

    // Inference happens here!
    float* output;
    int32_t output_size;
    infercat_iterate(
      img,
      (InfercatLayer**)(mnist_model),
      mnist_model_LAYERCOUNT,
      &output, &output_size
    );

    // ...
    printf("\n");
    printf("IMAGE INDEX\n");
    printf("===========\n");
    printf("%d\n", i);

    // ...
    printf("\n");
    printf("Label\n");
    printf("-----\n");
    printf("%d\n", img_label);

    // ...
    printf("\n");
    printf("Network Output\n");
    printf("--------------\n");
    float maxValue = -1e99;
    int32_t output_maxIndex = -1;
    for(int32_t j=0;j<output_size;j++)
    {
      if(output[j] > maxValue)
      {
        output_maxIndex = j;
        maxValue = output[j];
      }
      printf("[%d]: %f\n", j, output[j]);
    }

    // ...
    total_count += 1;
    if(output_maxIndex != img_label)
    {
      error_count += 1;
    }

    // ...
    float error_rate = (float)error_count / (float)total_count;

    // ...
    printf("\n");
    printf("Statistics\n");
    printf("----------\n");
    printf("Accuracy: %%%.3f (%d/%d)\n",
      (100.0 * (1.0 - error_rate)),
      total_count - error_count,
      total_count
    );
    printf("Error rate: %%%.3f (%d/%d)\n",
      (100.0 * error_rate), error_count, total_count
    );
  }

  // ...
  printf("\n");
  return 0;
}
