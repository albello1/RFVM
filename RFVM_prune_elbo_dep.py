import numpy as np
from scipy import linalg
from sklearn.metrics import r2_score, auc, roc_curve
import multiprocessing as mp
import math
from sklearn.preprocessing import StandardScaler
import scipy as sc
from sklearn.metrics import accuracy_score

class LR_ARD(object):
    def __init__(self):
        pass

    def fit(self, z, t, z_tst = None, t_tst = None,  hyper = None, prune = 0, maxit = 500, 
            pruning_crit = 1e-6, tol = 1e-6, prune_a = 0, pruning_crit_a = 1e-6):
        self.z = z  #(NxK)
        self.zv = z
        self.z_tst = z_tst  #(NxK_tst)
        self.t_tst = t_tst
        self.t = t  #(NxD)
        self.Ac = 0
        #self.K_tr = self.center_K(self.z @ self.z.T)
        self.prune = prune
        self.prune_a = prune_a
        self.labs_final = []
        self.num_feat = []
        self.num_sample = []


        self.K = self.z.shape[1] #num dimensiones input
        self.K_dep = self.z.shape[1]
        #Realmente es self.D = self.t.shape[1] pero para no fallar lo ponemos a 0
        self.D = self.t.shape[0] #num dimensiones output
        self.N = self.z.shape[0]
        self.Nv = self.zv.shape[0]# num datos
        self.N_tst = self.t_tst.shape[0]
        self.index = np.arange(self.K)
        self.index_a = np.arange(self.Nv)
        self.fact_sel = np.arange(self.z.shape[1])
        self.fact_sel_a = np.arange(self.zv.shape[0])
        

        self.L = []
        self.mse = []
        self.mse_tst = []        
        self.R2 = []
        self.R2_tst = []
        self.accu = []
        #self.AUC = []
        #self.AUC_tst = []
        self.K_vec = []
        self.labels_pred = []
        self.train_pred = []
        self.input_idx = np.ones(self.K, bool)
        if hyper == None:
            self.hyper = HyperParameters(self.K, self.N)
        else:
            self.hyper = hyper
        self.q_dist = Qdistribution(self.N, self.D, self.K, self.hyper, self.z)

        self.fit_vb(self.prune, self.prune_a, maxit, pruning_crit, pruning_crit_a, tol)
        
        
    def center_K(self, K):
        """Center a kernel matrix K, i.e., removes the data mean in the feature space
        Args:
            K: kernel matrix
        """
           
        size_1,size_2 = K.shape;
        D1 = K.sum(axis=0)/size_1
        D2 = K.sum(axis=1)/size_2
        E = D2.sum(axis=0)/size_1
        return K + np.tile(E,[size_1,size_2]) - np.tile(D1,[size_1,1]) - np.tile(D2,[size_2,1]).T



    def pruning(self, pruning_crit):       
        q = self.q_dist
        
        maximo = np.max(q.V['mean'])
        
        
        self.fact_sel = np.arange(self.z.shape[1])[(q.V['mean'] > maximo*pruning_crit).flatten()].astype(int)
        #fact_sel = np.arange(self.z.shape[1])[(abs(np.diagflat(q.V['mean']) @ self.zv.T @ q.A['mean'].T) > maximo*pruning_crit).flatten()].astype(int)
        
        
        
        aux = self.input_idx[self.input_idx]
        aux[self.fact_sel] = False
        self.input_idx[self.input_idx] = ~aux
        
        # Pruning alpha
        self.z = self.z[:,self.fact_sel]
        self.zv = self.zv[:,self.fact_sel]
        self.z_tst = self.z_tst[:,self.fact_sel]
        q.V['mean'] = q.V['mean'][self.fact_sel]
        #q.V['cov'] = q.V['cov'][fact_sel, fact_sel]
        q.V['cov'] = q.V['cov'][self.fact_sel]
        q.alpha['a'] = q.alpha['a'][self.fact_sel]
        q.alpha['b'] = q.alpha['b'][self.fact_sel]
        self.hyper.alpha_a = self.hyper.alpha_a[self.fact_sel]
        self.hyper.alpha_b = self.hyper.alpha_b[self.fact_sel]
        self.index = self.index[self.fact_sel]
        q.K = len(self.fact_sel)
        self.K = len(self.fact_sel)
        self.num_feat.append(len(self.fact_sel))

    def depruning(self, pruning_crit):
        q = self.q_dist
        
        maximo = np.max(q.V['mean'])

        K_prune = self.K_dep - self.K
        if K_prune >= 1:
            self.z = np.hstack((self.z, maximo*pruning_crit*0.1*np.ones((self.N, K_prune))))
            self.zv = np.hstack((self.zv, maximo*pruning_crit*0.1*np.ones((self.Nv, K_prune))))
            q.V['mean'] = np.concatenate((q.V['mean'],maximo*pruning_crit*0.1*np.ones((K_prune,1))))
            q.V['cov'] = np.concatenate((q.V['cov'],maximo*pruning_crit*0.1*np.ones((K_prune,1))))
            q.alpha['a'] = np.concatenate((q.alpha['a'],maximo*pruning_crit*0.1*np.ones(K_prune,)))
            q.alpha['b'] = np.concatenate((q.alpha['b'],maximo*pruning_crit*0.1*np.ones(K_prune,)))
            self.hyper.alpha_a = np.concatenate((self.hyper.alpha_a,maximo*pruning_crit*0.1*np.ones(K_prune,)))
            self.hyper.alpha_b = np.concatenate((self.hyper.alpha_b,maximo*pruning_crit*0.1*np.ones(K_prune,)))
            self.K = np.shape(self.z)[1]
        
    def pruning_a(self,pruning_crit_a):
        q = self.q_dist
        
        maximo = np.max(abs(q.A['mean']))
        
        self.fact_sel_a = np.arange(self.zv.shape[0])[(abs(q.A['mean']) > maximo*pruning_crit_a).flatten()].astype(int)
        
        
        self.zv = self.zv[self.fact_sel_a,:]
        q.A['mean'] = q.A['mean'][:,self.fact_sel_a]
        #q.V['cov'] = q.V['cov'][fact_sel, fact_sel]
        q.A['cov'] = q.A['cov'][self.fact_sel_a,:][:,self.fact_sel_a]
        q.A['prodT'] = q.A['mean'].T @ q.A['mean'] + q.A['cov']
        q.psi['a'] = q.psi['a'][self.fact_sel_a]
        q.psi['b'] = q.psi['b'][self.fact_sel_a]
        self.hyper.psi_a = self.hyper.psi_a[self.fact_sel_a]
        self.hyper.psi_b = self.hyper.psi_b[self.fact_sel_a]
        self.index_a = self.index_a[self.fact_sel_a]
        self.Nv = len(self.fact_sel_a)
        self.num_sample.append(len(self.fact_sel_a))
    
    def depruning_a(self, pruning_crit_a):
        q = self.q_dist
        
        maximo = np.max(abs(q.A['mean']))

        N_prune = self.N - self.Nv

        if N_prune >= 1:
            self.zv = np.vstack((self.zv, maximo*pruning_crit_a*0.1*np.ones((N_prune, self.K))))
            q.A['mean'] = np.hstack((q.A['mean'], maximo*pruning_crit_a*0.1*np.ones((1, N_prune))))
            q.A['cov'] = np.vstack((np.hstack((q.A['cov'], maximo*pruning_crit_a*0.1*np.ones((self.Nv, N_prune)))), maximo*pruning_crit_a*0.1*np.ones((N_prune, self.N))))
            q.A['prodT'] = np.vstack((np.hstack((q.A['prodT'], maximo*pruning_crit_a*0.1*np.ones((self.Nv, N_prune)))), maximo*pruning_crit_a*0.1*np.ones((N_prune, self.N))))
            q.psi['a'] = np.concatenate((q.psi['a'],maximo*pruning_crit_a*0.1*np.ones(N_prune,)))
            q.psi['b'] = np.concatenate((q.psi['b'],maximo*pruning_crit_a*0.1*np.ones(N_prune,)))
            self.hyper.psi_a = np.concatenate((self.hyper.psi_a,maximo*pruning_crit_a*0.1*np.ones(N_prune,)))
            self.hyper.psi_b = np.concatenate((self.hyper.psi_b,maximo*pruning_crit_a*0.1*np.ones(N_prune,)))
            self.Nv = np.shape(self.zv)[0]

       
    def predict(self, Z_test):
        q = self.q_dist
        
        
        
        med = Z_test @ np.diagflat(q.V['mean']) @ self.zv.T @ q.A['mean'].T + q.b['mean']
        #sig = q.tau_mean()
        sig = q.tau_mean() + q.b['cov'] + q.V['mean'].T * Z_test @ self.zv.T @ q.A['cov'] @ self.zv @ (Z_test.T * q.V['mean']) + q.V['cov'].T * Z_test @ self.zv.T @ q.A['prodT'] @ self.zv @ Z_test.T
        sig = np.diagonal(sig)
        sig = sig[:, np.newaxis]
        sig_2 = sig**2
        results = self.sigmoid(med/np.sqrt(1 + (np.pi/8)*sig_2))
        return results
            
    def predict_binary(self, Z_test):
        q = self.q_dist
        if self.prune == 1:
            Z_test = Z_test[:,self.fact_sel]
        ################
        q.A['cov'] = np.reshape(q.A['cov'], (np.shape(q.A['cov'])[0],np.shape(q.A['cov'])[0]))
        med = Z_test @ np.diagflat(q.V['mean']) @ self.zv.T @ q.A['mean'].T + q.b['mean']
        sig = q.tau_mean() + q.b['cov'] + q.V['mean'].T * Z_test @ self.zv.T @ q.A['cov'] @ self.zv @ (Z_test.T * q.V['mean']) + q.V['cov'].T * Z_test @ self.zv.T @ q.A['prodT'] @ self.zv @ Z_test.T
        sig = np.diagonal(sig)
        sig = sig[:, np.newaxis]
        results = self.sigmoid(med/np.sqrt(1 + (np.pi/8)*sig))
        labs = []
        for i in range(np.shape(self.labels_pred[-1])[0]):
            if not np.isnan(results[i]):
                if results[i] < 0.5:
                    labs.append(0)
                else:
                    labs.append(1)
            #Por si hay Nans
            else:
                labs.append(0)
        labs = np.array(labs)
        labs = labs.T
        return labs
    
    def return_pesos(self):
        q = self.q_dist
        return np.diagflat(q.V['mean']) @ self.zv.T @ q.A['mean'].T
    
    def return_feature(self):
        q = self.q_dist
        return q.V['mean']
    
    def return_vector_relevance(self):
        q = self.q_dist
        return q.A['mean']
    
    def return_alpha(self):
        q = self.q_dist
        return q.alpha_mean()
    
    def return_psi(self):
        q = self.q_dist
        return q.psi_mean()
    
    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))
    
    def gamma(self, X):
        return np.tanh(X/2)/(4*X)
    
    def final(self):
        q = self.q_dist
        return np.diagflat(q.V['mean']) @ self.z.T @ q.A['mean'].T
    
    def ret_final_probabilites(self):
        return self.predict(self.z_tst)

    def ret_train_probabilities(self):
        return self.predict(self.z)
        
    def fit_vb(self, prune, prune_a, maxit=200, pruning_crit = 1e-3, pruning_crit_a = 1e-3, tol = 1e-6):
        q = self.q_dist
        for j in range(maxit):
            self.update()
            self.K_vec.append(q.K)
            #####################
            self.labels_pred.append(self.predict(self.z_tst))
            self.train_pred.append(self.predict(self.z))
            ###############
            labs = []
            ###############
            for i in range(np.shape(self.labels_pred[-1])[0]):
                if self.labels_pred[-1][i,0] < 0.5:
                    labs.append(0)
                else:
                    labs.append(1)
            labs = np.array(labs)
            labs = labs.T
            self.labs_final.append(labs)
            
            labs_tr = []
            for i in range(np.shape(self.train_pred[-1])[0]):
                if self.train_pred[-1][i,0] < 0.5:
                    labs_tr.append(0)
                else:
                    labs_tr.append(1)
            labs_tr = np.array(labs_tr)
            labs_tr = labs_tr.T
            ##################
            self.accu.append(accuracy_score(labs[:,np.newaxis], self.t_tst))
            print('Iteration ', j)
            #print('Acc: ',accuracy_score(labs.flatten(), self.t_tst.flatten()))
            print('Acc train: ', accuracy_score(labs_tr[:,np.newaxis], self.t))
            print('Acc test: ', self.accu[-1])
            
            self.depruning(pruning_crit)
            self.depruning_a(pruning_crit_a)
            self.L.append(self.update_bound()[0][0])
            if prune == 1 and j>5:
                self.pruning(pruning_crit)
            if prune_a == 1 and j>5:
                self.pruning_a(pruning_crit_a)
            if q.K == 0:
                print('\nThere are no representative latent factors, no structure found in the data.')
                self.L.append(self.L[-3])
                self.labs_final.append(self.labs_final[-3])
                self.accu.append(self.accu[-3])
                return
            # print('\rIteration %d Lower Bound %.1f K %4d Nv %4d' %(j+1, self.L[-1], q.K, self.Nv), end='\r', flush=True)
            print('\rIteration %d Lower Bound %.1f K %4d Nv %4d' %(j+1, self.L[-1], q.K, self.Nv))
            if (len(self.L) > 300) and (abs(1 - np.mean(self.L[-200:-1])/self.L[-1]) < tol) or np.isnan(self.L[-1]):
                print('\nModel correctly trained. Convergence achieved')
                #print(np.diagflat(q.V['mean']) @ self.z.T @ q.A['mean'].T)
                return              
        print('')
        labs = []
        ###############
        for i in range(np.shape(self.labels_pred[-1])[0]):
            if self.labels_pred[-1][i,0] < 0.5:
                labs.append(0)
            else:
                labs.append(1)
        labs = np.array(labs)
        #print(labs.T)

    def update(self):
        self.update_a()
        self.update_psi()
        self.update_v()
        self.update_alpha()
        self.update_b()
        self.update_tau()
        self.update_xi()
        self.update_y()

    def myInverse(self,X):
        """Computation of the inverse of a matrix.
        
        This function calculates the inverse of a matrix in an efficient way 
        using the Cholesky decomposition.
        
        Parameters
        ----------
        __A: bool, (default 0). 
            Whether or not to print all the lower bound updates.
            
        """
        
        try:
            L = linalg.pinv(np.linalg.cholesky(X), rcond=1e-10) #np.linalg.cholesky(A)
            return np.dot(L.T,L) #linalg.pinv(L)*linalg.pinv(L.T)
        except:
            return np.nan
        
        
        
    def entrance(self):
        x = self.z
        return x
    
    def update_a(self):
        q = self.q_dist
        # cov
        a_cov = q.tau_mean() * self.zv @ (self.z.T * q.V['mean']) @ (q.V['mean'].T * self.z) @ self.zv.T + q.tau_mean() * self.zv @ (np.sum((self.z.T * q.V['cov'] * self.z.T),1) * self.zv).T
        a_cov_new = a_cov + np.diag(q.psi_mean())
        a_cov_inv = self.myInverse(a_cov_new)
        
        if not np.any(np.isnan(a_cov_inv)):
            q.A['cov'] = a_cov_inv
            # mean
            q.A['mean'] = q.tau_mean() * (q.Y['mean'].T - np.full((1, self.N), q.b['mean'])) @ self.z * q.V['mean'].T @ self.zv.T @ q.A['cov']
            #E[A*A^T]
            q.A['prodT'] = q.A['mean'].T @ q.A['mean'] + q.A['cov']
            
           
        else:
            print ('Cov A is not invertible, not updated')
            
        
    
    def update_v(self):
        q = self.q_dist

        v_var = q.tau_mean() * np.diag(np.dot(np.dot(self.zv.T, q.A['prodT']),self.zv)) * np.diag(self.z.T @ self.z) + q.alpha_mean()
        v_var = np.reshape(v_var, (self.K,1))
        if not np.any(np.isnan(v_var)): 
            q.V['cov'] = 1/v_var
    
            term_loop = -np.sum(((q.tau_mean()) * q.V['mean'] * self.zv.T @ q.A['prodT'] @ self.zv).T * (self.z.T @ self.z),1) + np.sum((q.tau_mean() * q.V['mean'] * self.zv.T @ q.A['prodT'])*self.zv.T,1) * np.sum((self.z**2).T,1)
            total = ((q.tau_mean() * (q.Y['mean'].T - q.b['mean']) @ self.z) * self.zv).T @ q.A['mean'].T + term_loop[:,np.newaxis]
            v_mean = q.V['cov'] * total
            q.V['mean'] = v_mean
            
            v_m, v_s = self.update_abs(q.V['mean'],q.V['cov'])
            q.V['mean'] = v_m
            q.V['cov'] = v_s
           
        else:
            print ('Cov V is not invertible, not updated')
        

     
    def update_b(self):
        q = self.q_dist
        q.b['cov'] = 1/(self.N * q.tau_mean() + 1)
        q.b['mean'] = q.tau_mean()* q.b['cov'] * (-np.full((1, self.N), 1) @ self.z * q.V['mean'].T @ self.zv.T @ q.A['mean'].T + np.sum(q.Y['mean']))
        q.b['prodT'] = q.b['mean']**2 + q.b['cov']
    
        
    def update_psi(self):
        q = self.q_dist
        q.psi['a'] = (self.hyper.psi_a + 0.5)
        q.psi['b'] = (self.hyper.psi_b + 0.5 * np.diag(q.A['prodT']))
    def update_alpha(self):
        q = self.q_dist
        q.alpha['a'] = (self.hyper.alpha_a + 0.5)
        q.alpha['b'] = (self.hyper.alpha_b[0] + 0.5 * (q.V['mean']**2 + q.V['cov'])).flatten()      
    
    def update_tau(self):
        q = self.q_dist
        q.tau['a'] = (self.hyper.tau_a + 0.5 * self.N)
        b_loop = np.sum(((q.V['mean'] @ q.V['mean'].T)*(self.zv.T @ q.A['prodT'] @ self.zv).T)*(self.z.T @ self.z).T + ((q.V['cov'] *(self.zv.T @ q.A['prodT'] @ self.zv).T)*(self.z.T @ self.z)))
        q.tau['b'] = (self.hyper.tau_b + 0.5 *(np.trace(q.Y['prodT']) - (2 * (q.Y['mean'].T - np.full((1, self.N), q.b['mean'])) @ self.z * q.V['mean'].T @ self.zv.T @ q.A['mean'].T) + b_loop + self.N * q.b['prodT'] -2 * q.Y['mean'].T @ np.full((1, self.N),q.b['mean']).T))
        
    
    def update_y(self):
        q = self.q_dist
        y_cov = self.myInverse(q.tau_mean()*np.identity(self.N) + 2*np.diagflat(self.gamma(q.xi['mean'])))
        if not np.any(np.isnan(y_cov)): 
            q.Y['cov'] = y_cov
            q.Y['mean'] = (self.t.T - np.full((1, self.N),1/2) + q.tau_mean() * q.A['mean'] @ self.zv @ np.diagflat(q.V['mean']) @ self.z.T + q.tau_mean()*q.b['mean']) @ q.Y['cov']
            q.Y['mean'] = q.Y['mean'].T
            q.Y['prodT'] = q.Y['mean'] @ q.Y['mean'].T + q.Y['cov']
        else:
            print ('Cov Y is not invertible, not updated')
    
    
    def update_xi(self):
        q = self.q_dist
        q.xi['mean'] = np.sqrt(q.Y['mean']**2 + np.reshape(np.diag(q.Y['cov']),(self.N,1))) 
        
    
    def HGamma(self, a, b):
        """Compute the entropy of a Gamma distribution.

        Parameters
        ----------
        __a: float. 
            The parameter a of a Gamma distribution.
        __b: float. 
            The parameter b of a Gamma distribution.

        """
        
        return -np.log(b)
    
    def HGauss(self, mn, cov, entr):
        """Compute the entropy of a Gamma distribution.
        
        Uses slogdet function to avoid numeric problems. If there is any 
        infinity, doesn't update the entropy.
        
        Parameters
        ----------
        __mean: float. 
            The parameter mean of a Gamma distribution.
        __covariance: float. 
            The parameter covariance of a Gamma distribution.
        __entropy: float.
            The entropy of the previous update. 

        """
        
        H = 0.5*mn.shape[0]*np.linalg.slogdet(cov)[1]
        return self.checkInfinity(H, entr)
        
    def checkInfinity(self, H, entr):
        """Checks if three is any infinity in th entropy.
        
        Goes through the input matrix H and checks if there is any infinity.
        If there is it is not updated, if there isn't it is.
        
        Parameters
        ----------
        __entropy: float.
            The entropy of the previous update. 

        """
        
        if abs(H) == np.inf:
            return entr
        else:
            return H
     
        
    def update_bound(self):
        """Update the Lower Bound.
        
        Uses the learnt variables of the model to update the lower bound.
        
        """
        
        q = self.q_dist
        q.A['LH'] = self.HGauss(q.A['mean'], q.A['cov'], q.A['LH'])
        q.V['LH'] = 0.5*q.V['mean'].shape[0]*np.sum(np.log(q.V['cov']))
        q.Y['LH'] = self.HGauss(q.Y['mean'], q.Y['cov'], q.Y['LH'])
        # Entropy of alpha and tau
        # q.alpha['LH'] = np.sum(self.HGamma(q.alpha['a'], q.alpha['b']))
        # q.tau['LH'] = np.sum(self.HGamma(q.tau['a'], q.tau['b']))
            
        # Total entropy
        # EntropyQ = q.W['LH'] + q.alpha['LH']  + q.tau['LH']
           
        # Calculation of the E[log(p(Theta))]
        q.tau['ElogpWalp'] = -(0.5 *  self.N + self.hyper.tau_a - 2)* np.log(q.tau['b'])
        q.alpha['Elogp'] = -(0.5 + np.mean(self.hyper.alpha_a) - 2)* np.sum(np.log(q.alpha['b']))
        q.psi['Elogp'] = -(0.5 + np.mean(self.hyper.psi_a) - 2)* np.sum(np.log(q.psi['b']))
        
        term = 0
        for n in range(np.shape(self.z)[0]):
            term += np.log(self.sigmoid(q.xi['mean'][n])) + q.Y['mean'][n]*self.t[n] -(1/2)*(q.Y['mean'][n] + q.xi['mean'][n]) - self.gamma(q.xi['mean'][n])*(q.Y['prodT'][n,n] - q.xi['mean'][n]**2)
        
        ElogP = q.tau['ElogpWalp'] + q.alpha['Elogp'] + q.psi['Elogp']
        return ElogP - q.A['LH'] - q.V['LH'] - (1/2)*q.b['prodT'] - (1/2)*q.b['cov']
               
    def update_abs(self, mea, sig):
        mean = np.sqrt((sig*2)/(np.pi))*np.exp(-(mea**2)/(2*(sig))) + mea*(1 - 2*sc.stats.norm.cdf(-(mea/np.sqrt(sig))))
        sigma = sig + mea**2 - mean**2
        return mean, sigma
        

class HyperParameters(object):
    def __init__(self, K, N):
        self.alpha_a = 1e-12 * np.ones((K,))
        self.alpha_b = 1e-14 * np.ones((K,))
        self.tau_a = 1e-14
        self.tau_b = 1e-14
        self.psi_a = 1e-12 * np.ones((N,))
        self.psi_b = 1e-14 * np.ones((N,))

class Qdistribution(object):
    def __init__(self, n, D, K, hyper, z):
        self.n = n
        self.D = D
        self.K = K
        self.z = z
        
        # Initialize gamma disributions
        alpha = self.qGamma(hyper.alpha_a,hyper.alpha_b,self.K)
        self.alpha = alpha 
        psi = self.qGamma(hyper.psi_a,hyper.psi_b,self.n)
        self.psi = psi 
        tau = self.qGamma(hyper.tau_a,hyper.tau_b,1)
        self.tau = tau 

        # The remaning parameters at random
        self.init_rnd()

    def init_rnd(self):
        self.A = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        self.V = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        self.b = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        self.Y = {
                "mean":     None,
                "cov":      None,
                "prodT":    None,
                "LH":       0,
                "Elogp":    0,
            }
        self.xi = {
                "mean":     None,
            }
        
            
        
        self.A["mean"] = np.random.normal(0.0, 1.0, self.n).reshape(self.n,1)
        self.A["cov"] = np.eye(self.n)
        self.A["prodT"] = np.dot(self.A["mean"].T, self.A["mean"])+self.n*self.A["cov"]
        
        self.V["mean"] = np.random.normal(0.0, 1.0, self.K).reshape(self.K,1)
        
        self.V["cov"] = np.ones((self.K,1))
        ###########
        self.b["mean"] = np.random.normal(0.0, 1.0)

        self.b["cov"] = 1
    
        self.b["prodT"] = self.b["mean"]**2 + self.b["cov"]
        #############
        self.Y['mean'] = self.z @ np.diagflat(self.V["mean"]) @ self.z.T @ self.A["mean"] + np.full((np.shape(self.z)[0], 1), self.b["mean"])
        
        self.Y['cov'] = self.tau_mean() * np.identity(np.shape(self.z)[0])
        
        self.Y['prodT'] = np.dot(self.Y["mean"].T, self.Y["mean"])+self.Y["cov"]
        ###############
        self.xi['mean'] = np.random.normal(0.0, 1.0, self.n).reshape(self.n,1)

    def qGamma(self,a,b,K):
        """ Initialisation of variables with Gamma distribution..
    
        Parameters
        ----------
        __a : array (shape = [1, 1]).
            Initialistaion of the parameter a.        
        __b : array (shape = [K, 1]).
            Initialistaion of the parameter b.
        __m_i: int.
            Number of views. 
        __K: array (shape = [K, 1]).
            dimension of the parameter b for each view.
            
        """
        
        param = {                
                "a":         a,
                "b":         b,
                "LH":         None,
                "ElogpWalp":  None,
            }
        return param
        
    def alpha_mean(self):
        return self.alpha['a'] / self.alpha['b'] 
    def tau_mean(self):
        return self.tau['a'] / self.tau['b']
    def psi_mean(self):
        return self.psi['a'] / self.psi['b']
