import numpy as np
import scipy.io as sio

# from matplotlib.colors import ListedColormap
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import time

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class classifiers(object):
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.load_data()
        
    def load_data(self):
        data = sio.loadmat(self.file_name)
        
        xtrain = data["xtrain"]
        self.xtrain = xtrain/255;      
        xtest = data["xtest"]
        self.xtest  = xtest/255;      
        self.ytrain = data["ytrain"].reshape(-1,)
        self.ytest = data["ytest"].reshape(-1,)

        self.ntest = self.ytest.shape[0]
        self.sample = 5000
        
    def log_reg(self):
        lr = LogisticRegression(random_state=0).fit(self.xtrain, self.ytrain)
        y_pred = lr.predict(self.xtest)
        return y_pred
    
    def knn(self):
        knn = KNeighborsClassifier(n_neighbors=3).fit(self.xtrain[0:self.sample], self.ytrain[0:self.sample])
        y_pred = knn.predict(self.xtest) 
        return y_pred
  
    def lin_svm(self):
        lsvm = SVC(kernel='linear', random_state=0).fit(self.xtrain[0:self.sample],self.ytrain[0:self.sample])
        y_pred = lsvm.predict(self.xtest)
        return y_pred
    
    def kern_svm(self):
        A = pairwise_distances(self.xtrain[0:self.sample], self.xtrain[0:self.sample])
        ndist = A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)
        bd = 1/(np.median(ndist.ravel()) **2) 

        ksvm = SVC(kernel='rbf', random_state=0, gamma=bd).fit(self.xtrain[0:self.sample],self.ytrain[0:self.sample])
        y_pred = ksvm.predict(self.xtest)
        return y_pred

    def neural_net(self):
        nn = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=0).fit(self.xtrain,self.ytrain)
        y_pred = nn.predict(self.xtest)
        return y_pred 
    
    def summarize_results(self,y_pred,run_time,model):     
        acc = sum(y_pred==self.ytest)/self.ntest
        cr = classification_report(self.ytest, y_pred)
        cm = confusion_matrix(self.ytest, y_pred)

        f = open('{}.txt'.format(model), 'w')
        f.write('{}\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\n'.format(model,cr, cm))
        f.write('accuracy: {}\n'.format(acc))
        f.write('running time: {} seconds'.format(run_time))
        f.close()


    def fit_model(self,model):
        t0 = time.time()
               
        if model == 'logistic_regression':
            y_pred = self.log_reg()            
        elif model == 'knn':
            y_pred = self.knn()
        elif model == 'linear svm':
            y_pred = self.lin_svm()
        elif model == 'kernel svm':
            y_pred = self.kern_svm()
        else: 
            y_pred = self.neural_net()
            
        run_time = round((time.time()-t0),2)
        
        self.summarize_results(y_pred,run_time,model)
        