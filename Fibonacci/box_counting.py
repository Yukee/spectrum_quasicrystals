
# coding: utf-8

# # Box-counting
# 
# ## Box-counting for 1D sets
# 
# We consider a set $S = \{ ..., \{ p_i, \mu_i \}, ... \}$ covering an interval $I$ on the real line. We cover $I$ with boxes of equal size. 
# We count the point $p$ in the box $b$ if 
# $$ b \epsilon \leq p  < (b+1)\epsilon, $$ 
# where $\epsilon$ in the size of the boxes. By convention the last point is in the last box.
# This is equivalent to saying that point $p$ is in the $b^\text{th}$ box (labelling from small to large absissas) iff
# $$ \text{floor}\left( \frac{p}{\epsilon} \right) = b. $$
# 
# We can either keep $\epsilon$ fixed, say $\epsilon = 1$, and vary the length of $I$, or keep the length of $I$ fixed and vary $\epsilon$. We are going to adopt this latter point of view.
# 
# We define the $q$-weight, or partition function, at length scale $\epsilon$:
# $$ \chi_q(\epsilon) = \sum_b \left( \mu_\epsilon(b) \right)^q $$
# where $\mu_\epsilon(b)$ is the weight of the box $b$:
# $$ \mu_\epsilon(b) = \sum_{i \text{ in } b} \mu_i  $$.
# 
# Multifractal thermodynamic theory tells us that the $q$-weight is related to the fractal dimensions in the same way than the partition function of a thermodynamic system is related to its free energy:
# $$ \tau_q \sim \frac{\log \chi_q(\epsilon)}{\log \epsilon}$$
# where we take $\epsilon \rightarrow 0$ limit, equivalent to the thermodynamic limit of a thermodynamic system.
# 
# ## Box size specification
# 
# The box size can be written as $\epsilon = \delta^{-n} L$. If $I$ is of length $L$, then if either $\delta$ or $n$ are not integers, the box size will not divide 1, ie part of the last box will fall out of $I$.
# This generally means that the $q$-weight will be badly estimated. 
# 
# ## Accuracy
# 
# With double floats (default python on the computer I wrote this!), the fractal dimensions are typically accurate up to 2 digits.


import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool

# rescale S between m and M
def rescale(s,m,M):
    return (M-m)*(s-s[0])/(s[-1]-s[0])+m

""" compute the q-weigth (or partition function) of S """

# s: set of absissas, w: set of corresponding weigths
def qWeight(s, w, epsilon, q):
    # checks
    if len(s) != len(w): return("the sets of absissas and weights are of unequal lenths")
    #if sum(w) != 1.: return("the weights are not normalized to 1")
    
    # rescale the set between 0 and 1
    s = rescale(s, 0., 1.)
    # total q-weight
    qw = 0
    # weight in the current box
    curw = w[0]
    # label of the current box
    b = 0
    
    for i in range(1,len(s)-1):
        bnew = math.floor(s[i]/epsilon)
        # if the current point is in the same box as the one juste at its left, add its weigth to the total weight in the current box
        if bnew == b:
            curw += w[i]
        # else the current point is in a new box, add the last box weight to the total q-weight, set the current label to this box
        else:
            qw += curw**q
            curw = w[i]
            b = bnew
   
    # the last point is in the last box
    nb = math.floor(1./epsilon)
    # the last point is alone in the last box
    if b < nb-1:
        qw += curw**q
        qw += w[-1]**q
    # there is at least 2 points in the last box
    elif b == nb-1:
        curw += w[-1]
        qw += curw**q
    # some part of the last box falls outside of I
    else: 
        print('Box size '+str(epsilon)+' does not divide 1! Still adding data point')
    
    return qw


# ## Application to the Fibonacci Hamiltonian
# 
# We consider Fibonacci tight-binding Hamiltonian with pure hopping.
# 
# We can compute the fractal dimensions of the wavefunctions. We then take the positions and the weights to be:
# $$ p_i = x_i, \text{ and } w_i =|\psi_E(x_i)|^2  $$
# where $\psi_E(x_i)$ is the coefficient at site $i$ of position $x_i$ of the wavefunction associated to the energy $E$.
# 
# We can also compute the fractal dimensions of the spectrum. This time $p_i$ is the position of the band $i$, and $w_i$ is the width of this band.
# 
# We can as well compute the fractal dimensions of the local spectral measure $\mu_{x,x}(E)$. $p_i$ is still the position of the band $i$, and $w_i$ is the local density of states at position $x$ and energy in the band $i$.


""" build the Fibonacci tight-binding hamiltonian """

# compute Fibonacci numbers
def fib(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

# inverse golden ratio
om = 2./(1.+math.sqrt(5))     

# jump amplitudes
def jump(n, i, tw, ts):
    p = fib(n-2)    
    q = fib(n-1)
    L = fib(n)    
    if(i*q % L < p):return ts
    else:return tw

# build the hamiltonian (free boundary conditions)
def h(n, rho):
    L = fib(n)
    h = np.zeros((L,L))
    for i in range(L-1): 
        h[i,i+1] = jump(n, i, rho, 1.)
        h[i+1,i] = h[i,i+1]
    return h
    
# build the hamiltonian (periodic boundary conditions)
def hp(n, rho):
    L = fib(n)    
    hp = np.zeros((L,L))
    for i in range(L-1): 
        hp[i,i+1] = jump(n, i, rho, 1.)
        hp[i+1,i] = hp[i,i+1]
        hp[0,L-1] = jump(n, L-1, rho, 1.)
        hp[L-1,0] = hp[0,L-1]
    return hp

""" Pass to conumbering """

# takes a label (a position along the chain), return the associated colabel (position in perp space)
def co(label, n):
    q = fib(n-1)
    L = fib(n)
    return q*label % L

# takes a colabel (position in perp space), return the associated label (position in para space)
def label(co, n):
    q = fib(n-1)
    L = fib(n)
    sgn = -2*(n % 2)+1
    return sgn*q*co % L

# takes a matrix written in position basis, return the same matrix in the conumbered basis
def conum(vec, n):
    q = fib(n-1)
    L = fib(n)
    # compute the permutation matrix associated to approximant n
    perm_mat = np.fromfunction(lambda i,j: i == q*j % L, (L, L))
    # permute elements in vec
    perm_vec = np.dot(perm_mat,np.dot(vec,np.transpose(perm_mat)))
    return perm_vec

# ## Local fractal dimensions of the wavefunctions

""" compute the eigenvalues """

# diagonalize
rho = .1
n = 15
val, vec = linalg.eigh(hp(n, rho))

# in conumbering!
covec = conum(vec,n)

""" Averaged fractal dimensions of the wavefunctions """
q = 2.

# position set
s = np.arange(0,fib(n),1.)

# wavefunction index 
a = label(306,n)
# in conumbering, wavefunction indexes are also relabelled
coa = co(a, n)

# weights are presence probabilities
w = abs(vec[:,a])**2
cow = abs(covec[:,coa])**2
#cow = np.copy(w)
# test whether permuting randomly the positions has an impact on the behaviour of the q norm
np.random.shuffle(cow)
    
# smallest box size (in the form smallest_espilon = 10**(-t))
t = 3.5
# for some reason, there are problems with 3, so we exclude it from the list of primes
primes = [2,5,6,7,10]
epsilonRange = sorted([1/n**delta for n in primes for delta in range(1,int(t*math.log(10.)/math.log(n))+1)])
#epsilonRange = 2**np.arange(-21.,0.,1.)

qwList = [qWeight(s, w, epsilon, q) for epsilon in epsilonRange]
coqwList = [qWeight(s, cow, epsilon, q) for epsilon in epsilonRange]

""" data plot and linear regression """

def loglogplot(min, max):
    
    # plot
    epsilonRange2 = epsilonRange[min:max]
    qwList2 = qwList[min:max]

    plt.title('The q-weigth for the wavefunction '+str(a)+' for q = ' + str(q))
    plt.xlabel('log epsilon')
    plt.ylabel('log chi')
    plt.loglog(epsilonRange, qwList,'-,r',markersize=3., linewidth=2.)
    plt.loglog(epsilonRange2, qwList2,'o',markersize=3., linewidth=2.)
    plt.show()
    
    # regression
    logQW = [math.log(qw) for qw in qwList2]
    logEps = [math.log(eps) for eps in epsilonRange2]

    slope, intercept, r_value, p_value, std_err = stats.linregress(logEps,logQW)
    print(slope, r_value)

def cologlogplot(min, max):
    
    # plot
    epsilonRange2 = epsilonRange[min:max]
    qwList2 = coqwList[min:max]

    plt.title('The q-weigth for the wavefunction '+str(a)+' for q = ' + str(q))
    plt.xlabel('log epsilon')
    plt.ylabel('log chi')
    plt.loglog(epsilonRange, coqwList,'-,r',markersize=3., linewidth=2.)
    plt.loglog(epsilonRange2, qwList2,'o',markersize=3., linewidth=2.)
    plt.show()
    
    # regression
    logQW = [math.log(qw) for qw in qwList2]
    logEps = [math.log(eps) for eps in epsilonRange2]

    slope, intercept, r_value, p_value, std_err = stats.linregress(logEps,logQW)
    print(slope, r_value)
    
def comp(min, max):
    
    # plot
    epsilonRange2 = epsilonRange[min:max]
    coqwList2 = coqwList[min:max]
    qwList2 = qwList[min:max]

    plt.title('The q-weigth for the wavefunction '+str(a)+' for q = ' + str(q))
    plt.xlabel('log epsilon')
    plt.ylabel('log chi')
    plt.loglog(epsilonRange, qwList,'-',markersize=3., linewidth=2.)
    plt.loglog(epsilonRange, coqwList,'-',markersize=3., linewidth=2.)
    plt.loglog(epsilonRange2, qwList2,'o',markersize=3., linewidth=2.)
    plt.loglog(epsilonRange2, coqwList2,'o',markersize=3., linewidth=2.)
    plt.show()
    
    # regression
    logQW = [math.log(qw) for qw in qwList2]
    cologQW = [math.log(qw) for qw in coqwList2]
    logEps = [math.log(eps) for eps in epsilonRange2]

    slope, intercept, r_value, p_value, std_err = stats.linregress(logEps,logQW)
    coslope, intercept, r_value, p_value, std_err = stats.linregress(logEps,cologQW)    
    print(slope, coslope)