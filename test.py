import torch 
import numpy as np
from sklearn.decomposition import PCA
W = np.random.random((5,3))
W1 = W.copy()
for i in range(1,5):
    W1[i,:] = W1[i,:]-W1[i-1,:]
W1[0,:] = 0
pca = PCA(n_components=2)
x = pca.fit_transform(W)
P0 = np.array(pca.components_)
y = pca.fit_transform(W1)
P1 = np.array(pca.components_)
a = np.linalg.matrix_rank(P0)
b = np.linalg.matrix_rank(P1)
P3 = np.concatenate((P0,P1),axis=0)
c = np.linalg.matrix_rank(P3)
pass