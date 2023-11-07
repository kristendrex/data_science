from kmeans import kmeans
import imageio
from matplotlib import pyplot as plt
import time
import numpy as np
import sys

def run_kmeans(path):    
    raw_img = imageio.imread(path)
    k_mesh = [2,4,8, 16, 32]
    run_time = []
    n_empty_all = []

    # prepare fig/ax
    fig, ax = plt.subplots(3,2)
    ax[0,0].imshow(raw_img)
    ax[0,0].set_title('original', fontsize = 8)
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)


    # set random seed
    rseed = 6

    for ii in range(5):
        start_time = time.time()
        np.random.seed(rseed)

        # set initial centroids to be in the range of 100 - 200
        ct_init = np.random.random((k_mesh[ii],3))*100+100

        img, n_empty = kmeans(raw_img, k_mesh[ii], ct_init)
        end_time = time.time()

        # show image on its associated axis
        ax[int((ii+1)/2), np.remainder(ii+1,2)].imshow(img)
        ax[int((ii+1)/2), np.remainder(ii+1,2)].set_title('k='+str(k_mesh[ii]), fontsize = 8)
        ax[int((ii+1)/2), np.remainder(ii+1,2)].get_xaxis().set_visible(False)
        ax[int((ii+1)/2), np.remainder(ii+1,2)].get_yaxis().set_visible(False)

        run_time.append(end_time - start_time)
        n_empty_all.append(n_empty)

    fig.tight_layout(pad=0.5)
    fig.suptitle('Results for Different Values of K:')
    fig.subplots_adjust(top=0.9)

    # save figure 
    plt.savefig('kmeans_export.pdf', dpi = 300)

    print('Running time for each k:')
    for k_ind in range(5):
        out_text = 'k = '+str(k_mesh[k_ind])+':   '+'%.2f'%run_time[k_ind]+ 'sec.    # of empty clusters: '+ str(n_empty_all[k_ind])
        print(out_text)
        with open('result.txt','a') as writer:
            writer.write(out_text)
            writer.write("\n")
            
if __name__ == "__main__":
    path = str(sys.argv[1])
    run_kmeans(path)