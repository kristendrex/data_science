import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

# Define Euclidean Distance
def distance(x,y, lp):
    return np.linalg.norm(x-y, lp)

def adjacencyMatrix(images, d, n):
    # Generate Matrix for Similarity Graph
    G = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            G[i,j] = distance(images[:,i], images[:,j], 1)

    for i in range(n):
        p = np.percentile(G[i,:], 10100/n) # Find the threshold for the distrance of 100 nearest neighbor
        for j in range(n):
            if G[i,j] > p:
                G[i,j] = 99999.9 # Assign large value for distance greater than the threshold   
    return G

def Matrix_D(W):
    # Generate graph and obtain Matrix D from weight matrix, W,
    # defining the weight on the edge between each pair of nodes.
    # assign sufficiently large weights to non-existing edges

    n = np.shape(W)[0]
    Graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            Graph.add_weighted_edges_from([(i,j,min(W[i,j], W[j,i]))])

    res = dict(nx.all_pairs_dijkstra_path_length(Graph))
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            D[i,j] = res[i][j]
    np.savetxt('D.csv', D)
    return D

# function to visualize the projected dataset
def plot_faces(z, sample,images):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(z[:, 0], z[:, 1], '.k')
    for i in sample:
        single_image = images[:, i].reshape(64, 64).T

        imagebox = OffsetImage(single_image, zoom=0.6, cmap = 'gray')
        ab = AnnotationBbox(imagebox, z[i], pad=0.1)
        ax.add_artist(ab)
        
def isomap(file):
    #load data
    images = loadmat(file)['images']
    (d, n) = np.shape(images)
    
    # generate adjacency matrix
    A = adjacencyMatrix(images, d, n)
    
    # compute D, the pairwise shortest distance matrix
    D = Matrix_D(A)
    D = (D + D.T)/2
    
    # Use a centering matrix H to get C, the covariance matrix
    ones = np.ones([n,1])
    H = np.eye(n) - 1/n*ones.dot(ones.T)
    C = -H.dot(D**2).dot(H)/(2*n)
    
    # obtain and sort eigen vals (lambdas) and eigen vectors (w vector)
    eig_val, eig_vec = np.linalg.eig(C) 
    index = np.argsort(-eig_val) # Sort eigenvalue from large to small

    #calculate matrix Z
    Z = eig_vec[:,index[0:2]].dot(np.diag(np.sqrt(eig_val[index[0:2]])))
    
    return A, Z