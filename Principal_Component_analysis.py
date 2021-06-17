import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import math
import random

#The amount of the correlation
n = 1

#Generate 1000 samples from a uniform random variable
x = np.random.uniform(1,2,1000)

y = x.copy()*n # y = n * x

#PCA works better if the data is centered
x = x - np.mean(x) #Center x. Remove its mean
y = y - np.mean(y) #Center y. Remove its mean

#Create a data frame with x and y
data = pd.DataFrame({'x': x, 'y':y})
#Plot the original correlated data in blue
plt.scatter(data.x, data.y)

#Instantiate a PCA. Choose to get two output variables
pca = PCA(n_components = 2)

#Create the transformation model for this data. Internally, it gets 
# the rotation matrix and the explained variance

pcaTr = pca.fit(data)

# Transform the database on the rotation of pcaTr
# Create a data frame with the new variables. We call these variables PC1 and PC2
rotatedData = pcaTr.transform(data)
# Transform the database on the rotated matrix
dataPCA = pd.DataFrame(data = rotatedData, columns = ['PC1', 'PC2'])

plt.scatter(dataPCA.PC1, dataPCA.PC2)
plt.savefig('.\pca_transform.png')

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

random.seed(100)

std1 = 1
# The desired standard deviation of our first random variable
std2 = 0.3333
# The desired standard deviation of our second random variable

x = np.random.normal(0, std1, 1000)
y = np.random.normal(0, std2, 1000)

#PCA works better if the data is centered

x = x - np.mean(x)
y = y - np.mean(y)

#Define  a pair of dependent variables with a desired amount
#of covariance

n = 1 #Magnitude of covariance
angle = np.arctan(1/n) # Convert the covariance to and angle
print('angle: ', angle*180/math.pi)

# Create a rotation matrix using the given angle

rotationMatrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

print('rotationMatrix')
print(rotationMatrix)

xy = np.concatenate(([x], [y]), axis = 0)
print(x)
print(y)
print(xy)

# Transform the data using the rotation matrix. It correlates the two variables
data = np.dot(xy.T, rotationMatrix)
plt.scatter(data[:,0], data[:,1])
plt.savefig('./savefig.png')