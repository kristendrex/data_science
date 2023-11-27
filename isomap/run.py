import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
import random
import isomap

#load isomap data
file = '../../data/isomap/isomap.mat'
images = loadmat(file)['images']
(d, n) = np.shape(images)

#get adjacency matrix and eigenvectors
A,Z = isomap.isomap(file)

#plot adjacency matrix by intensity
plt.figure()
plt.title('Adjacency Matrix')
plt.imshow(A,cmap=plt.get_cmap('gray'))
plt.savefig('adjacency_matrix.png')

# plot a random sample of images
sample = random.sample(range(n), 40)
isomap.plot_faces(Z,sample,images, 'plot_faces.png')

# compare with PCA implementation 
pca = PCA(n_components = 2)
pca.fit(images.T)
images_new = pca.transform(images.T)
isomap.plot_faces(images_new, sample,images, 'plot_faces_PCA.png')