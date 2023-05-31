#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:16:53 2023

@author: firework
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate a grid of x and y values
x = np.linspace(-5, 5, 101)
y = np.linspace(-5, 5, 101)
x, y = np.meshgrid(x, y)

# Define the parameters of the sinusoidal function
A = 1
B = 2
C = 0.65

# Compute the z values
z = A * np.cos(B * x + C)

# Create a 3D plot
fig = plt.figure(figsize=(10, 10)) # Set the figure size here (width, height)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
fig.colorbar(ax.plot_surface(x, y, z, cmap='viridis'))

# Add labels
ax.set_xlabel('time')
ax.set_ylabel('position')
ax.set_zlabel('value')
ax.set_title('3D Sinusoidal Function')
#plt.savefig('3D Sinusoidal Function')
plt.show()