import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

class em_pca_mnist(object):
    
    def __init__(self,data,label): 
        np.random.seed(0) # for reproducibility
        self.data = data
        self.label = label
        self.n, self.m = data.shape        
        
    def pca(self, num_dims):
        self.num_dims = num_dims
        
        datamean = np.mean(self.data, axis=1)
        datamean = datamean[:, np.newaxis]
        xc = self.data - datamean
        C = xc@xc.T/self.m
        u, s, _ = np.linalg.svd(C)

        ut = u[:, 0:num_dims]
        ds = np.diag(s[0:num_dims])
        d_inv_half = np.diag(s[0:num_dims]**(-1/2))

        #below vars are referenced later, so make them global
        # reduced-dimensional data, transpose to keep the dimension definition consistent
        self.data_rd = (xc.T @ ut @ d_inv_half).T  
        self.datamean = datamean
        self.ut = ut
        self.ds = ds
                
    def init_posterior(self, k, num_dims):
        mu_old = np.random.randn(k,num_dims)
        mu_new = mu_old + 10

        sig_old = np.empty((k, num_dims, num_dims))
        for ii in range(k):
            tmp = np.random.randn(num_dims, num_dims)
            tmp = tmp@tmp.T +1
            sig_old[ii] = tmp
        sig_new = sig_old + 10

        pi_old = np.random.random(k)
        pi_old = pi_old/pi_old.sum()

        pi_new = np.zeros(2)
        tau = np.zeros((self.m,k), dtype=float) 

        return mu_old, mu_new, sig_old, sig_new, pi_old, pi_new, tau
    
    def expectation_step(self,mu_old,sigma_old,pi_old,tau):
        for ii in range(self.k):
            fpdf = mvn(mu_old[ii], sigma_old[ii]) 
            tau[:,ii] = fpdf.pdf(self.data_rd.T)
        tau = tau*pi_old
        tmp = tau.sum(axis=1)
        tmp = tmp[:, np.newaxis]
        tau = tau / tmp  
        return tau

    def maximization_step(self,tau):
        mu_new = tau.T @ self.data_rd.T
        tmp = tau.sum(axis=0)
        tmp = tmp[:, np.newaxis]
        mu_new = mu_new/tmp  
        return mu_new

    def update_vars(self,sigma_new,mu_new,tau):
        # update cov matrix
            for ii in range(self.k):
                tmp = self.data_rd.T - mu_new[ii]
                sigma_new[ii] = tmp.T @ np.diag(tau[:,ii]) @ tmp/ tau[:,ii].sum()

            # update priors
            pi_new = tau.sum(axis=0)/self.m

            # check log-likelihood
            likelihood = 0  # likelihood
            for ii in range(self.k):
                ftmp = mvn(mu_new[ii], sigma_new[ii])
                ll_tmp = ftmp.pdf(self.data_rd.T) * pi_new[ii]
                likelihood = likelihood+ ll_tmp
            self.ll_all.append(np.sum(np.log(likelihood)))

            return sigma_new, pi_new

    def em_alg(self,k):
        # number of mixture components
        self.k = k 

        # Metrics for log-likelihood
        self.ll_all = []
        i = 1
        maxIter =100
        tol = 1e-2

        mu_old, mu_new, sig_old, sig_new, pi_old, pi_new, tau = self.init_posterior(k, self.num_dims)

        while i < maxIter:
            tau = self.expectation_step(mu_old,sig_old,pi_old,tau)
            mu_new = self.maximization_step(tau)
            sig_new, pi_new = self.update_vars(sig_new,mu_new,tau)

            #exit condition
            if np.linalg.norm(mu_new.ravel() - mu_old.ravel()) < tol:
                break

            #update prev mu, sigma, pi
            mu_old = mu_new
            sig_old = sig_new
            pi_old = pi_new

            #increment iteration
            i = i+1
        
        #globalize variables that will be used later
        self.mu_new = mu_new
        self.sig_new = sig_new
        self.tau = tau
            
    def graph_log_likelihood(self):
        plt.plot(self.ll_all,'-o')
        plt.title('log-likelihood over iteration')
        plt.xlabel('iteration')
        plt.savefig('log_likelihood.png')
        

    def visualize_avg_outcome(self,save_file):
        # ### recover the mean vector
        mu_rec = np.empty((self.k, self.n))
        sig_rec = np.empty((self.k, self.n, self.n))
        fig2, ax2 = plt.subplots(2, 2)
        for ii in range(self.k):
            tmp = self.ut @ np.sqrt(self.ds) @ self.mu_new[ii]
            mu_rec[ii] = self.datamean.ravel() + tmp
            im = mu_rec[ii].reshape(28, 28)
            ax2[ii, 0].imshow(im.T, cmap='gray')
            ax2[ii, 0].set_title('mean vector')
            ax2[ii, 0].set_xticks([])
            ax2[ii, 0].set_yticks([])

            sig_rec[ii] = self.ut @ np.sqrt(self.ds) @ self.sig_new[ii] @ np.sqrt(self.ds) @ self.ut.T
            ax2[ii, 1].imshow(sig_rec[ii], cmap='gray')
            ax2[ii, 1].set_title('covariance matrix')
            ax2[ii, 1].set_xticks([])
            ax2[ii, 1].set_yticks([])
        fig2.savefig(save_file)
            
    def classification_accuracy(self):
        idx2 = np.where(self.label==2)[0]
        idx6 = np.where(self.label==6)[0]

        match2 = self.tau[idx2,0]>=self.tau[idx2,1]
        acc2 = match2.sum()/idx2.size
        acc2 = np.max([acc2, 1-acc2])

        match6 = self.tau[idx6,0]>=self.tau[idx6,1]
        acc6 = match6.sum()/idx6.size
        acc6 = np.max([acc6, 1-acc6])
        
        return acc2, acc6