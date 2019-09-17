#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('04cars-fixed.csv')
cars = np.array(df.axes[0])
data = df.values.T
U, _, _ = np.linalg.svd(data)
pca_data = np.matmul(U[:, :3].T, data)
manufacturers = ['Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Hummer', 'Hyundai', 'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Lexus', 'Lincoln', 'Mazda', 'Mercedes-Benz', 'Mercury', 'Mini Cooper', 'Mitsubishi', 'Nissan', 'Oldsmobile', 'Pontiac', 'Porsche', 'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen', 'Volvo']

manufacturer_map = []
for car in cars:
    for idx, manufacturer in enumerate(manufacturers):
        if manufacturer in car:
            manufacturer_map.append(float(idx + 1))

a = np.array(manufacturer_map)
ax = plt.subplot(111, projection='3d')
ax.view_init(25, -135)
ax.scatter(pca_data[0], pca_data[1], pca_data[2], marker='o',
           c=a, cmap='gist_rainbow')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.show()
