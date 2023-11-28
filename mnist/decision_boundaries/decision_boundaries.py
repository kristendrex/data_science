import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class classifiers(object):
    
    def __init__(self, tgt_x, tgt_y, noise_level):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(tgt_x, tgt_y, test_size=0.2)
        self.ntest = len(self.y_test)
        self.tgt_x = tgt_x
        self.tgt_y = tgt_y
        self.noise_level = noise_level 
        
    def naive_bayes(self):
        nb = GaussianNB(var_smoothing = self.noise_level)
        y_pred_nb = nb.fit(self.X_train, self.y_train).predict(self.X_test)
        rate_nb = sum(y_pred_nb==self.y_test)/self.ntest
        return rate_nb  
    
    def log_reg(self):
        lr = LogisticRegression(random_state=0).fit(self.X_train, self.y_train)
        y_pred_lr = lr.predict(self.X_test)
        rate_lr = sum(y_pred_lr==self.y_test)/self.ntest
        return rate_lr
    
    def knn(self):
        kn = KNeighborsClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        y_pred_kn = kn.predict(self.X_test)
        rate_kn = sum(y_pred_kn==self.y_test)/self.ntest
        return rate_kn
    
    def get_classifier_accuracy(self):
        rate_nb = self.naive_bayes()
        rate_lr = self.log_reg()
        rate_kn = self.knn()
        return rate_nb, rate_lr, rate_kn

    def pca(self, x_data):
        xc = (x_data - x_data.mean(axis=0))
        u, s, _ = np.linalg.svd(xc.T @ xc /len(xc))
        xt = xc@u[:,0:2]@np.diag(s[0:2]**-1/2)
        return xt, u, s
    
    def reduce_dims(self):
        X_trn, X_tst, y_train, y_test = train_test_split(self.tgt_x, self.tgt_y, test_size=0.2)
        X_train, u, s = self.pca(X_trn)
        X_test = (X_tst-X_trn.mean(axis=0))@ u[:, 0:2] @np.diag(s[0:2]**-1/2)
        self.X_train_red = X_train
        self.X_test_red = X_test
        self.y_train_red = y_train
        return X_train, X_test, y_train
        
    def fit_reduced(self):
        X_train, X_test, y_train = self.reduce_dims()
        self.nb_red = GaussianNB().fit(X_train, y_train)
        self.lr_red = LogisticRegression(random_state=0).fit(X_train, y_train)
        self.kn_red = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)  
        
    # ref: https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
    def plot_decision_boundary(self,model, title,file_name): #, x_train, x_test, y_train):

        h = 0.005
        cmap_light = ListedColormap(['#F8BAAE',  '#D4DFFF'])
        cmap_bold = ListedColormap(['#FF0000',  '#0000FF'])

        x_min, x_max = self.X_train_red[:,0].min(), self.X_train_red[:,0].max() 
        y_min, y_max = self.X_train_red[:,1].min(), self.X_train_red[:,1].max()

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

        # Also plot the training points
        plt.scatter(self.X_train_red[:,0], self.X_train_red[:,1], c=self.y_train_red, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(title)
        plt.savefig(file_name)