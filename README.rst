==============
SPSA
==============
:Info: A Simultaneous Perturbation Stochastic Approximation optimisation code in python
:Author: J Gomez-Dans <j.gomez-dans@ucl.ac.uk>
:Date: $Date: 2012-04-25 16:00:00 +0000  $
:Description: README file


Simultaneous perturbation stochastic approximation
---------------------------------------------------

The code here is a function optimiser that uses ideas of stochastic approximation to estimate the function gradient, and feeds them into an steepest descent algorithm. In theory, only two extra function evaluations are required to approximate the gradient (although you could obviously use more), resulting in a fairly economic iteration. The main issues are to do with the scheduling update. These require the specification of how much to follow down the gradient, and therefore require tweaking. In the code, this parameter is ``a``. And lower values ought to be preferred, but not too low as otherwise, exploration of the functional space is very slow.
