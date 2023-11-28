import numpy as np
import decision_boundaries

#upload data
images  = np.loadtxt("../../../data/mnist/data.dat").T  
labels = np.loadtxt("../../../data/mnist/label.dat")

#get decision boundaries
cls = decision_boundaries.classifiers(images, labels, 1e-3)
rate_nb, rate_lr, rate_kn = cls.get_classifier_accuracy()

#write results to a file
with open('result.txt','a') as writer:
    writer.write('Results for MNIST Data:')
    writer.write("\n")
    writer.write('test accuracy of NB: {0:.4f}'.format(rate_nb))
    writer.write("\n")
    writer.write('test accuracy of LR: {0:.4f}'.format(rate_lr))
    writer.write("\n")
    writer.write('test accuracy of KN: {0:.4f}'.format(rate_kn))
    writer.write("\n")
    

#plot decision boundaries
cls.fit_reduced()
cls.plot_decision_boundary(cls.nb_red, 'mnist: Naive Bayes','naive_bayes.png')
cls.plot_decision_boundary(cls.lr_red, 'mnist: Logistic Regression','log_reg.png')
cls.plot_decision_boundary(cls.kn_red, 'mnist: KNN','knn.png')