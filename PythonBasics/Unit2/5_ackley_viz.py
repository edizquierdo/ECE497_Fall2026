import torch
import matplotlib.pyplot as plt

def ackley(x):
    return (
        -20 * torch.exp(-0.2 * torch.sqrt(torch.sum(x**2, dim=-1) / 2))
        - torch.exp(torch.sum(torch.cos(2 * torch.pi * x), dim=-1) / 2)
        + 20 + torch.e
    )

# 2-D grid
x = torch.linspace(-5, 5, 300)
X, Y = torch.meshgrid(x, x, indexing="ij")
points = torch.stack((X, Y), dim=-1)

# Evaluate Ackley
Z = ackley(points)

# Plot
plt.contourf(X, Y, Z, levels=50)
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.colorbar(label="Ackley")
plt.show()

# 3-D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("Ackley")
ax.set_title("Ackley Function")
plt.show()