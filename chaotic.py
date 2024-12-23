import numpy as np
from scipy.ndimage import zoom
np.random.seed(0)
a = np.array([3,1,2,4,5])
row_permutation = np.arange(5)

random_array = np.random.rand(3, 3)
print(random_array)
index1 = np.argsort(a)
row_permutation = row_permutation[index1]

inv_index1 = np.argsort(index1)

chaotic = [1,2,3]
chaotic1 = [1,3,2]
b = a[row_permutation]

for i in range(3):
    random_array[i,:]= np.roll(random_array[i,:], chaotic[i])
    for j in range(3):
        random_array[:,j]= np.roll(random_array[:,j], -chaotic1[j])  
print(random_array)

for i in range(3):
    for j in range(3):
        random_array[:,j]= np.roll(random_array[:,j], chaotic1[j])  
    random_array[2-i,:]= np.roll(random_array[2-i,:], -chaotic[2-i])
print(random_array)