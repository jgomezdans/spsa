

class SimpleSPSA(FiniteDifferences):
    """Simultaneous Perturbation Stochastic Approximation. 
    """
    # These constants are used throughout
    alpha = 0.602
    gamma = 0.101
    a = .0017
    

    def __init__ ( self, loss_function, noise_var=0.01, args=(), min_vals=None, max_vals=None, param_tolerance=None, function_tolerance=None,  n_grads=10 ):
        """The constructor requires a loss function and any required extra 
        arguments. Optionally, boundaries as well as tolerance thresholds can
        be specified"""
        self.loss = loss_function
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.param_tolerance = param_tolerance
        self.function_tolerance = function_tolerance
        self.c = noise_var
        self.max_iter = 10000
        self.A = self.max_iter*0.1
        
    def calc_loss ( self, theta ):
        """Evalute the cost/loss function with a value of theta"""
        retval = self.loss ( theta, *(self.args ) )
        return retval

    def minimise ( self, theta_0, ens_size=10 ):
        """The main minimisation looop"""
        n_iter = 0
        p = theta_0.shape[0]
        print "Starting theta= %f" % theta_0
        theta = theta_0
        j_old = self.calc_loss ( theta )
        while ( ( ghat > 1.0e-6 ) and (n_iter < self.max_iter)):
            an = self.a/( i + 1 + self.A)**self.alpha
            cn = self.c/(i+1)**self.gamma    
            for j in np.arange ( ens_size ):
                delta = 2*np.round ( np.rand(p)) - 1
                theta_plus = theta + ck*delta
                theta_minus = theta - ck*delta
                j_plus = self.calc_loss ( theta_plus )
                j_minus = self.calc_loss ( theta_minus )
                ghat += ( j_plus - j_minus)/(2.*ck*delta)
            theta_saved = theta
            theta -= ak*(ghat/float(ens_size))
            j_new = self.calc_loss ( theta )
            if self.function_tolerance is not None:    
                if np.abs ( j_new - j_old ) > self.function_tolerance:
                    theta = theta_saved
                else:
                    j_old = j_new
            if self.min_vals is not None and self.max_vals is not None:
                theta = np.minimum ( theta, self.min_vals )
                theta = np.maximum ( theta, self.max_vals ) 

            n_iter += 1
        return ( theta, j_new, n_iter)

if __name__ == "__main__":
    num_points = 150
    Tx = np.linspace(5., 8., num_points)
    Ty = Tx
    tX = 11.86*np.cos(2*np.pi/0.81*Tx-1.32) + \
        0.64*Tx+4*((0.5-np.rand(num_points))*np.exp(2*np.rand(num_points)**2))
    tY = -32.14*np.cos(2*pi/0.8*Ty-1.94) + \
        0.15*Ty+7*((0.5-np.rand(num_points))*np.exp(2*np.rand(num_points)**2))
    # Fit the first set
    fitfunc = lambda p, x: p[0]*np.cos(2*pi/p[1]*x+p[2]) + p[3]*x # Target function
    errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
    opti = SimpleSPSA ( errfunc, args=(Tx, tx), noise_var=2, param_tolerance=0.1*np.ones(3) )
    theta_0 = np.array([-15., 0.8, 0., -1.])
    opti.minimise ( theta_0, ens_size = 20)