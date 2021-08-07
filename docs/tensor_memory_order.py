# -----------------------------------------------------------------------------
# 
# 
# -----------------------------------------------------------------------------
import numpy as np

# ...
tensor_img = np.zeros((5, 5, 3))

# ...
for i in range(0, tensor_img.shape[2]):
  for y in range(0, tensor_img.shape[1]):
    for x in range(0, tensor_img.shape[0]):
      tensor_img[x,y,i] = (100 * i) + (x + (y * tensor_img.shape[0]))

# ...
tensor_img_flat = tensor_img.flatten('C')

# ...
print("")
print("TENSOR")
print("------")
print(tensor_img)

# ...
print("")
print("FLAT")
print("----")
print(tensor_img_flat)

# ...
x = 4
y = 2
ch = 1
print("")
print("Check")
print("-----")
print("%d vs %d" % (
  tensor_img_flat[(x * (5 * 3)) + (y * 3) + ch],
  tensor_img[x,y,ch]
))

# ...
tensor_kernel = np.zeros((3, 3, 2, 4))

# ...
for j in range(0, tensor_kernel.shape[3]):
  for i in range(0, tensor_kernel.shape[2]):
    for y in range(0, tensor_kernel.shape[1]):
      for x in range(0, tensor_kernel.shape[0]):
        tensor_kernel[x,y,i,j] = (
          (100 * j) + (10 * i) + (x + (y * tensor_kernel.shape[0]))
        )

# ...
tensor_kernel_flat = tensor_kernel.flatten('C')

# ...
print("")
print("TENSOR")
print("------")
print(tensor_kernel)

# ...
print("")
print("FLAT")
print("----")
print(tensor_kernel_flat)

# ...
x = 2
y = 2
ch_in = 1
ch_out = 3
print("")
print("Check")
print("-----")
print("%d vs %d" % (
  tensor_kernel_flat[(x * (3 * 4 * 2)) + (y * 4 * 2) + (ch_in * 4) + ch_out],
  tensor_kernel[x,y,ch_in,ch_out]
))
