// ----------------------------------------------------------------------------
//
//
// ----------------------------------------------------------------------------
#include "mnistReader.h"
#include <stdio.h>

// ----------------------------------------------------------------------------
static FILE* fp_img;
static FILE* fp_label;
static int32_t m_itemCount_img = 0;
static int32_t m_itemCount_label = 0;

// ----------------------------------------------------------------------------
static uint32_t make32b(uint8_t* data)
{
  uint32_t value = 0;
  value += (uint32_t)(data[0]) << 24;
  value += (uint32_t)(data[1]) << 16;
  value += (uint32_t)(data[2]) <<  8;
  value += (uint32_t)(data[3]) <<  0;
  return value;
}

// ----------------------------------------------------------------------------
int32_t mnistReader_init(
  const char* file_path_labels, const char* file_path_images
)
{
  // ...
  int32_t rv;
  uint8_t ch[4];
  uint32_t value;

  // ...
  fp_img = fopen(file_path_images, "r");
  fp_label = fopen(file_path_labels, "r");

  // ...
  rv = fread(&ch, sizeof(uint8_t), 4, fp_label);
  value = make32b(ch);
  if(value != 0x00000801)
  {
    printf("Wrong magic number for label file!\r\n");
    return -1;
  }

  // ...
  rv = fread(&ch, sizeof(uint8_t), 4, fp_img);
  value = make32b(ch);
  if(value != 0x00000803)
  {
    printf("\r\n");
    printf("Wrong magic number for image file!\r\n");
    return -1;
  }

  // ...
  rv = fread(&ch, sizeof(uint8_t), 4, fp_label);
  m_itemCount_label = make32b(ch);
  printf("\r\n");
  printf("Label count: %d\r\n", m_itemCount_label);

  // ...
  rv = fread(&ch, sizeof(uint8_t), 4, fp_img);
  m_itemCount_img = make32b(ch);
  printf("\r\n");
  printf("Image count: %d\r\n", m_itemCount_img);

  // ...
  if(m_itemCount_label != m_itemCount_img)
  {
    printf("\r\n");
    printf("Label count / Image count mismatch!\r\n");
    return -1;
  }

  // ...
  return 0;
}

// ----------------------------------------------------------------------------
int32_t mnistReader_getByIndex(int32_t index, uint8_t* image, uint8_t* label)
{
  // Index check
  if((index >= m_itemCount_label) || (index < 0))
  {
    return -1;
  }

  // ...
  int32_t rv;

  // Read label
  fseek(fp_label, 8 + index, SEEK_SET);
  rv = fread(label, sizeof(uint8_t), 1, fp_label);

  // Read image
  fseek(fp_img, 16 + (index * 784), SEEK_SET);
  rv = fread(image, sizeof(uint8_t), 784, fp_img);

  // ...
  return 0;
}
