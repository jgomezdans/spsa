#!/usr/bin/env python

"""
A class to implement Simultaneous Perturbation Stochastic Approximation.
"""
import pdb
import numpy as np


class SimpleSPSA ( object ):
    """Simultaneous Perturbation Stochastic Approximation. 
    """
    # These constants are used throughout
    alpha = 0.602
    gamma = 0.101
    
    

    def __init__ ( self, loss_function, a_par = 1e-6, noise_var=0.01, args=(), \
            min_vals=None, max_vals=None, param_tolerance=None, \
            function_tolerance=None ):
        """The constructor requires a loss function and any required extra 
        arguments. Optionally, boundaries as well as tolerance thresholds can
        be specified.
        
        :param loss_function: The loss (or cost) function that will be minimised.
            Note that this function will have to return a scalar value, not a 
            vector.
        :param a_par: This is the ``a`` parameter, which controls the scaling of
            the gradient. It's value will have to be guesstimated heuristically.
        :param noise_var: The noise variance is used to scale the approximation
            to the gradient. It needs to be >0.
        :param args: Any additional arguments to ``loss_function``.
        :param min_vals: A vector with minimum bounds for parameters
        :param max_vals: A vector with maximum bounds for parameters
        :param param_tolerance: A vector stating the maximum parameter change
            per iteration.
        :param function_tolerance: A scalar stating the maximum change in 
            ``loss_function`` per iteration.
        :return: None
        """
        self.args = args
        self.loss = loss_function
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.param_tolerance = param_tolerance
        self.function_tolerance = function_tolerance
        self.c_par = noise_var
        self.max_iter = 5000000
        self.big_a_par = self.max_iter/10.
        self.a_par = a_par
        
    def calc_loss ( self, theta ):
        """Evalute the cost/loss function with a value of theta"""
        retval = self.loss ( theta, *(self.args ) )
        return retval

    def minimise ( self, theta_0, ens_size=2, report=500 ):
        """The main minimisation loop. Requires a starting value, and optionally
        a number of ensemble realisations to estimate the gradient. It appears
        that you only need two of these, but the more the merrier, I guess.
        
        :param theta_0: The starting value for the minimiser
        :param ens_size: Number of relaisations to approximate the gradient.
        :return: A tuple containing the parameters that optimise the function,
            the function value, and the number of iterations used.
        """
        n_iter = 0
        num_p = theta_0.shape[0]
        print "Starting theta=", theta_0
        theta = theta_0
        j_old = self.calc_loss ( theta )
        # Calculate the initial cost function
        theta_saved = theta_0*100
        while  (np.linalg.norm(theta_saved-theta)/np.linalg.norm(theta_saved) >\
                1e-8) and (n_iter < self.max_iter):
            # The optimisation carried out until the solution has converged, or
            # the maximum number of itertions has been reached.
            theta_saved = theta # Store theta at the start of the iteration
                                # as we may well be restoring it later on.
            # Calculate the ak and ck scalars. Note that these require
            # a degree of tweaking
            ak = self.a_par/( n_iter + 1 + self.big_a_par)**self.alpha
            ck = self.c_par/( n_iter + 1 )**self.gamma  
            ghat = 0.  # Initialise gradient estimate
            for j in np.arange ( ens_size ):
                # This loop produces ``ens_size`` realisations of the gradient
                # which will be averaged. Each has a cost of two function runs.
                # Bernoulli distribution with p=0.5
                delta = (np.random.randint(0, 2, num_p) * 2 - 1)
                # Stochastic perturbation, innit
                theta_plus = theta + ck*delta
                theta_plus = np.minimum ( theta_plus, self.max_vals )
                theta_minus = theta - ck*delta
                theta_minus = np.maximum ( theta_minus, self.min_vals )
                # Funcion values associated with ``theta_plus`` and 
                #``theta_minus``
                j_plus = self.calc_loss ( theta_plus )
                j_minus = self.calc_loss ( theta_minus )
                # Estimate the gradient
                ghat = ghat + ( j_plus - j_minus)/(2.*ck*delta)
            # Average gradient...
            ghat = ghat/float(ens_size)
            # The new parameter is the old parameter plus a scaled displacement
            # along the gradient.
            not_all_pass = True
            this_ak = ( theta*0 + 1 )*ak
            theta_new = theta
            while not_all_pass:
                out_of_bounds = np.where ( np.logical_or ( \
                    theta_new - this_ak*ghat > self.max_vals, 
                    theta_new - this_ak*ghat < self.min_vals ) )[0]
                theta_new = theta - this_ak*ghat
                if len ( out_of_bounds ) == 0:
                    theta = theta - this_ak*ghat
                    not_all_pass = False
                else:
                    this_ak[out_of_bounds] = this_ak[out_of_bounds]/2.
            
            # The new value of the gradient.
            j_new = self.calc_loss ( theta )
            # Be chatty to the user, tell him/her how it's going...
            if n_iter % report == 0:
                print "\tIter %05d" % n_iter, j_new, ak, ck
            # Functional tolerance: you can specify to ignore new theta values
            # that result in large shifts in the function value. Not a great
            # way to keep the results sane, though, as ak and ck decrease
            # slowly.
            if self.function_tolerance is not None:    
                if np.abs ( j_new - j_old ) > self.function_tolerance:
                    print "\t No function tolerance!", np.abs ( j_new - j_old )
                    theta = theta_saved
                    continue
                else:
                    j_old = j_new
            # You can also specify the maximum amount you want your parameters
            # to change in one iteration.
            if self.param_tolerance is not None:
                theta_dif = np.abs ( theta - theta_saved ) 
                if not np.all ( theta_dif < self.param_tolerance ):
                    print "\t No param tolerance!", theta_dif < \
                        self.param_tolerance
                    theta = theta_saved
                    continue
            # Ignore results that are outside the boundaries
            if (self.min_vals is not None) and (self.max_vals is not None):      
                i_max = np.where ( theta >= self.max_vals )[0]
                i_min = np.where ( theta <= self.min_vals )[0]
                if len( i_max ) > 0:
                    theta[i_max] = self.max_vals[i_max]*0.9
                if len ( i_min ) > 0:
                    theta[i_min] = self.min_vals[i_min]*1.1
            n_iter += 1
        return ( theta, j_new, n_iter)

def test_spsa ( p_in, noise_var ):
    
    
    fitfunc = lambda p, x: p[0]*x*x + p[1]*x + p[2]
    errfunc = lambda p, x, y, noise_var: np.sum ( (fitfunc( p, x ) - y)**2/ \
        noise_var**2 )
    # The following error function can be used by leastsq in scipy.optimize
    #errfunc2 = lambda p, x, y: fitfunc( p, x ) - y
    # make some data
    x_arr = np.arange(100) * 0.3
    obs = p_in[0] * x_arr**2 + p_in[1] * x_arr + p_in[2]
    np.random.seed(76523654)
    noise = np.random.normal(size=100) * noise_var  # add some noise to the obs
    obs += noise
    
    opti = SimpleSPSA ( errfunc, args=( x_arr, obs, noise_var), \
        noise_var=noise_var, min_vals=np.ones(3)*(-5), max_vals = np.ones(3)*5 )
    theta0 = np.random.rand(3)
    ( xsol, j_opt, niter ) = opti.minimise (theta0 )
    print xsol, j_opt, niter
if __name__ == "__main__":
    test_spsa ( [0.1, -2.6, -1.5], 0.3 )