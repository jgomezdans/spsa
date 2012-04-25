#!/usr/bin/env python
import numpy as np


class SimpleSPSA ( object ):
    """Simultaneous Perturbation Stochastic Approximation. 
    """
    # These constants are used throughout
    alpha = 0.602
    gamma = 0.101
    a = 1e-6
    

    def __init__ ( self, loss_function, noise_var=0.01, args=(), min_vals=None, max_vals=None, param_tolerance=None, function_tolerance=None ):
        """The constructor requires a loss function and any required extra 
        arguments. Optionally, boundaries as well as tolerance thresholds can
        be specified"""
        self.args = args
        self.loss = loss_function
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.param_tolerance = param_tolerance
        self.function_tolerance = function_tolerance
        self.c = noise_var
        self.max_iter = 5000000
        self.A = self.max_iter/10.
        
    def calc_loss ( self, theta ):
        """Evalute the cost/loss function with a value of theta"""
        retval = self.loss ( theta, *(self.args ) )
        return retval

    def minimise ( self, theta_0, ens_size=3 ):
        """The main minimisation looop"""
        n_iter = 0
        p = theta_0.shape[0]
        print "Starting theta=", theta_0
        theta = theta_0
        j_old = self.calc_loss ( theta )
        theta_saved = theta_0*100
        while  (np.linalg.norm(theta_saved-theta)/np.linalg.norm(theta_saved) > \
                1e-8) and (n_iter < self.max_iter):
            theta_saved = theta
            ak = self.a/( n_iter + 1 + self.A)**self.alpha
            ck = self.c/( n_iter + 1 )**self.gamma  
            ghat = 0.
            for j in np.arange ( ens_size ):
                delta = (np.random.randint(0, 2,p) * 2 - 1) 
                theta_plus = theta + ck*delta
                theta_minus = theta - ck*delta
                j_plus = self.calc_loss ( theta_plus )
                j_minus = self.calc_loss ( theta_minus )
                ghat = ghat + ( j_plus - j_minus)/(2.*ck*delta)
            ###fprime = optimize.approx_fprime ( theta, self.calc_loss, 1e-15)
            ###fprime[0] = np.sum ( 2*self.args[0]**2*(self.args[1]-theta[0]*self.args[0]**2 - theta[1]*self.args[0] - theta[2]))/self.args[2]**2
            ###fprime[1] = np.sum ( 2*self.args[0]*(self.args[1]-theta[0]*self.args[0]**2 - theta[1]*self.args[0] - theta[2]))/self.args[2]**2
            ###fprime[2] = np.sum ( 2*(self.args[1]-theta[0]*self.args[0]**2 - theta[1]*self.args[0] - theta[2]))/self.args[2]**2
            ghat = ghat/float(ens_size)
            theta = theta - ak*ghat
            #theta = theta - ak*fprime
            
            j_new = self.calc_loss ( theta )
            if n_iter % 500 == 0:
                print "\tIter %05d"%n_iter, j_new, theta, ak, ck
            if self.function_tolerance is not None:    
                if np.abs ( j_new - j_old ) > self.function_tolerance:
                    print "\t No function tolerance!", np.abs ( j_new - j_old )
                    theta = theta_saved
                    continue
                else:
                    j_old = j_new
            if self.param_tolerance is not None:
                theta_dif = np.abs ( theta - theta_saved ) 
                if not np.all ( theta_dif < self.param_tolerance ):
                    print "\t No param tolerance!",theta_dif < self.param_tolerance
                    theta = theta_saved
                    continue
                    
            if (self.min_vals is not None) and (self.max_vals is not None):
                theta = np.minimum ( theta, self.max_vals )
                theta = np.maximum ( theta, self.min_vals ) 
             
            
            n_iter += 1
        return ( theta, j_new, n_iter)

if __name__ == "__main__":
    from scipy import optimize
    
    fitfunc = lambda p, x: p[0]*x*x + p[1]*x + p[2]
    errfunc = lambda p, x, y, noise_var: np.sum ( (fitfunc(p,x)-y)**2/noise_var**2 )
    errfunc2 = lambda p, x, y: fitfunc(p,x)-y
    # make some data
    x = np.arange(100) * 0.3
    obs = 0.1 * x**2 - 2.6 * x - 1.5
    np.random.seed(76523654)
    noise = np.random.normal(size=100) * 0.3     # add some noise to the obs
    obs += noise
    
    opti = SimpleSPSA ( errfunc, args=( x, obs, 0.3), noise_var=0.3, \
        min_vals=np.ones(3)*(-5), max_vals = np.ones(3)*5 )
    theta_0 = np.random.rand(3)
    opti.minimise (theta_0 )