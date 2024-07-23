import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)

#ensors can be created from NumPy arrays (and vice versa - see Bridge with NumPy).
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#From another tensor:The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data) #retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_twos = torch.zeros_like(x_data) #retains the properties of x_data
print(f"Ones Tensor: \n {x_twos} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) #overrides the datatype of x_data
print(f"Ones Tensor: \n {x_rand} \n")

#With random or constant values:shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

shape = (2,3,)
rand_tensor = torch.rand(shape)
print(f"Random Tensor:\n {rand_tensor}\n")
ones_tensor = torch.ones(shape)
print(f"Ones Tensor: \n {ones_tensor}\n")
zeros_tensor = torch.zeros(shape)
print(f"Zeros Tensor:\n{zeros_tensor}")

#Atrributeas of a tensor
#Tensor attributes describe their shape, datatype, and the device on which they are sorted.

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#Operations on Tensors
#Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described here_.

#Each of these operations can be run on the GPU (at typically higher speeds than on a CPU). If you’re using Colab, allocate a GPU by going to Runtime > Change runtime type > GPU.

#By default, tensors are created on the CPU. We need to explicitly move tensors to the GPU using .to method (after checking for GPU availability). Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!

#We move our tensor to thre GPU if availabale
if torch.cuda.is_available():
 tensor = tensor.to("cuda")

#standard numpy-like indedxing and sciling:

tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#Joining tensors You can use torch.cat to concatenate a sequence of tensors along a given dimension. See also torch.stack_, another tensor joining operator that is subtly different from torch.cat.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#Arithematic Operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor

y1= tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out =y3)

#This computes the element wise product. z1,z2,z3 will now have the same values
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

# Single-element tensors If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with numpy

#Tensor to Numoy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensir reflects in the Numoy array

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#Numpy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

#Changes in the Numpy array reflects in the tensor

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")