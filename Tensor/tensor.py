# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Learning Tensors

# %%
import torch
import numpy as np
import gc

# %% [markdown]
# ## Initialize a Tensor

# %%
# from list data
list_data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(list_data)
print(tensor_from_list)

# from numpy
np_data = np.array(list_data)
tensor_from_np = torch.from_numpy(np_data)
print(tensor_from_np)

# specify dtype
tensor_from_list_int32 = torch.tensor(list_data, dtype=torch.int32)
print(tensor_from_list_int32)

# from others (param type must be torch.Tensor)
tensor_ones = torch.ones_like(tensor_from_list)
print(f"Ones like: \n{tensor_ones}")

tensor_rand = torch.rand_like(tensor_from_list, dtype=torch.float)
print(f"Rand like: \n{tensor_rand}")

# clear


# %%
# from shape

shape = (3, 4, )  # why comma
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor}")
print(f"Ones Tensor: \n {ones_tensor}")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %% [markdown]
# ## Tensor Attribute
#
# - shape
# - dtype
# - device

# %%
print(f"Shape: {tensor_from_list.shape}")
print(f"Dtype: {tensor_from_list.dtype}")
print(f"Device: {tensor_from_list.device}")

# %% [markdown]
# ## Tensor Operation
#
# - indexing and slicing
# - joining
# - arithmetic
# - single element value
# - in-place

# %%
tensor_test = torch.rand((4, 5))
print(f"Origin Matrix: \n{tensor_test}\n")
# indexing
# tensor[i][j] -> Row i, Col j
print(f"Row 1, Col 1: {tensor_test[1][1]}\n")

# slicing
print(f"Col 0: {tensor_test[:, 0]}")
print(f"Row 0: {tensor_test[0, :]}")
print(f"Col 1: {tensor_test[..., 1]}")
print(f"Row 1: {tensor_test[1, ...]}\n")

# joining
tensors = [torch.rand((3, 4)) for i in range(3)]
tensor_concat = torch.cat(tensors, dim=0)
print(f"Concat on Dimension 0: \n{tensor_concat.shape}")
tensor_concat = torch.cat(tensors, dim=1)
print(f"Concat on Dimension 1: \n{tensor_concat.shape}\n")

# arithmetic
tensor_test = torch.rand((5, 5))
print(f"Origin Matrix: \n{tensor_test}\n")
# matmul
res1 = tensor_test @ tensor_test.T
res2 = tensor_test.matmul(tensor_test.T)
res3 = torch.zeros_like(tensor_test)
torch.matmul(tensor_test, tensor_test.T, out=res3)
print(res1.equal(res2) and res2.equal(res3))
# element-wise mul
res1 = tensor_test * tensor_test
res2 = tensor_test.mul(tensor_test)
res3 = torch.zeros_like(tensor_test)
torch.mul(tensor_test, tensor_test, out=res3)
print(res1.equal(res2) and res2.equal(res3))

# %% [markdown]
# ## Single element

# %%
tensor_sum = tensor_test.sum()
print(type(tensor_sum))
print(type(tensor_sum.item()))

# %% [markdown]
# ## In-place

# %%
print(f"Origin Matrix: \n{tensor_test}\n")
# add to every element
tensor_test.add_(5)
print(f"Current Matrix: \n{tensor_test}\n")
