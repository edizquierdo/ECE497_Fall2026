import torch

# Create a 2D tensor of shape (2, 3)
x = torch.tensor([[3.0, 4.0, 0.0],
                  [1.0, 2.0, 2.0]])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# # Compute the norm along the last dimension (columns)
# result = torch.linalg.norm(x, dim=-1)

# print(result)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")