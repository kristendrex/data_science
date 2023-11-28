import em_gmm
import numpy as np
import matplotlib.pyplot as plt

#upload data
data  = np.loadtxt("../../../data/mnist/data.dat")
labels = np.loadtxt("../../../data/mnist/label.dat")

#visualize a sample of 2 and 6
fig, (axs1, axs2) = plt.subplots(1, 2)
img1 = data.T[np.random.randint(0, 700)].reshape((28,28))
img2 = data.T[np.random.randint(1000, 1800)].reshape((28,28))
fig.suptitle('Sample of 2 and 6')
axs1.imshow(np.rot90(np.fliplr(img1)), cmap="Greys")
axs2.imshow(np.rot90(np.fliplr(img2)), cmap="Greys") 
fig.savefig('sample.png')

#Implement EM for MNIST dataset, first applying PCA to reduce dimensionality
#to fit a Gaussian Mixure Model where k = 2
#K = 2 since we are modeling 2 outcomes of interest
mnist = em_gmm.em_pca_mnist(data,labels)
mnist.pca(5)
mnist.em_alg(2)

#Graph log-likelihood over each iteration
mnist.graph_log_likelihood()

#Recover the mean vector, visualize what the average 2 and 6 look like
mnist.visualize_avg_outcome('avg_outcome.png')

#write classification results to a text file
acc2, acc6 = mnist.classification_accuracy()
with open('result.txt','a') as writer:
    writer.write('Results for MNIST Accuracy:')
    writer.write("\n")
    writer.write('accuracy rate for #2: {0:.4f}'.format(acc2))
    writer.write("\n")
    writer.write('accuracy rate for #6: {0:.4f}'.format(acc6))
    writer.write("\n")