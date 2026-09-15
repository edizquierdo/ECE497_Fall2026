import matplotlib.pyplot as plt
import numpy as np

# Create x values
x = np.linspace(0, 10, 100)

# Create a figure and axes
fig, ax = plt.subplots()

# Plot multiple lines with labels and styles
ax.plot(x, np.sin(x), label='Sine', color='blue', linestyle='-')
ax.plot(x, np.cos(x), label='Cosine', color='red', linestyle='--')

# Add labels, title, and legend
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Multiple Lines Example')
ax.legend()
ax.grid(True)

plt.show()