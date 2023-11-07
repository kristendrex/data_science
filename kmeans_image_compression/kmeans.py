import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import time
import imageio

def kmeans(data, k, ct_init):
    
    ct_old = ct_init

    #set dimension variables
    n1,n2,n3 = np.shape(data)
    data = np.reshape(data,(-1, 3))
    dim = np.shape(data)[1]
    
    #set empty array to hold updated centroids
    ct_new = np.full((k,dim), np.nan)
    
    #set looping parameter to ensure stopping condition
    maxIter = 200
    nIter = 1
    cost_old = 1e10
    cost_list = []
    
    while (nIter <=maxIter):
    
        # find the distances of all data points to each centroid, L2 norm
        dist_mtx = cdist(data, ct_old, 'euclidean') 
        
        #reset cost to have a basis of comparison for each iteration
        current_cost = 0
        
        # find the cluster assignment for each data point
        cl = np.argmin(dist_mtx, axis=1)
        
        # update the centroid for each group
        for idx in range(k):
                        
            # find the index of data points in cluster ii
            idx_j = np.where(cl==idx)
            x_j = data[idx_j]
            
            # update new centroid based on mean
            ct_new[idx] = np.mean(x_j, axis=0)
            
            # exit condition if new centroid mean is infinite
            if ~np.isfinite(sum(ct_new[idx])):
                ct_new[idx] = np.full(np.shape(ct_new[idx]), fill_value = np.inf)
            
            # calculate cost for current iteration
            current_cost = current_cost + np.sum(x_j.dot(ct_new[idx]))
        
        # record cost of current iteration
        cost_list.append(current_cost)
        
        # check converge
        if current_cost == cost_old:
            break
        
        # update the variable for next iteration
        cost_old = current_cost
        ct_old = ct_new
        nIter = nIter+1
    
    # assign the new pixel value with new centroid
    dist_all = cdist(data, ct_new, 'euclidean') # L2 norm
    cl_all = np.argmin(dist_all, axis=1)
    
    # prepare to output the result
    img = np.full(np.shape(data), fill_value = np.nan)
    for ii in np.unique(cl_all):
        img[np.where(cl_all == ii)] = ct_new[ii]/255
    
    img_out = np.reshape(img,(n1,n2,n3))
    
    # check empty cluster:
    n_empty = sum(1 - np.isfinite( np.sum(ct_new, axis=1) ))
    
    return img_out, n_empty