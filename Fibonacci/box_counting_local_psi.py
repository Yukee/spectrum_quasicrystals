
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
import cmath
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
    
# build the hamiltonian (k flux boundary conditions)
def hk(n, rho, k):
    L = fib(n)    
    hp = np.zeros((L,L), dtype=complex)
    for i in range(L-1): 
        hp[i,i+1] = jump(n, i, rho, 1.)
        hp[i+1,i] = hp[i,i+1]
        hp[0,L-1] = jump(n, L-1, rho, 1.)*cmath.exp(1j*k)
        hp[L-1,0] = jump(n, L-1, rho, 1.)*cmath.exp(-1j*k)
    return hp
    
# build the hamiltonian (conumbering, periodic boundary conditions)
def hpco(n, rho):
    p = fib(n-2)
    q = fib(n-1)
    h = np.zeros((p+q,p+q))
    for i in range(p):
        h[i,i+q] = 1.
        h[i+q,i] = 1.
    for i in range(q):
        h[i,i+p] = rho
        h[i+p,i] = rho
    return h


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
    
""" Compute x recursively using conumbering """

# return the x associated to a given conumber co
def x(co, n, ren):
    if(co >= fib(n-1)):
        ren += 1
        co -= fib(n-1)
        n -= 2
    elif(co >= fib(n-2)):
        co -= fib(n-2)
        n -= 3
    else:
        ren += 1
        n -= 2
        
    if(n<3): return ren
    else: return x(co,n,ren)

# return the renormalization path
def path(co, n, pth):
    if(co >= fib(n-1)):
        pth += '+'
        co -= fib(n-1)
        n -= 2
    elif(co >= fib(n-2)):
        pth += '0'
        co -= fib(n-2)
        n -= 3
    else:
        pth += '-'
        n -= 2
        
    if(n<2): return pth
    else: return path(co,n,pth)

#n = 17
#xlist = [x(label(co,n),n,0) for co in range(fib(n))]
#plt.plot(xlist)
#plt.show()

# ## Local fractal dimensions of the wavefunctions

""" compute the eigenvalues and write them to a file """

# diagonalize
#rho = .1
#n = 16
#valco, vecco = linalg.eigh(hpco(n,rho))
#np.save("data/wf_conum_n" + str(n) +"_rho" + str(rho), vecco)
#val0, vec0 = linalg.eigh(hk(n, rho, 0))
#valpi2, vecpi2 = linalg.eigh(hk(n, rho, 0.5*cmath.pi))
#valpi, vecpi = linalg.eigh(hk(n, rho, cmath.pi))
#int0 = abs(vec0)**2
#intpi2 = abs(vecpi2)**2
#intpi = abs(vecpi)**2
## average over boundary conditions
#vec = (int0 + 4*intpi2 + intpi)/6
#
## write to file
#np.save("data/wf_averaged_n" + str(n) +"_rho" + str(rho), vec)

""" Averaged fractal dimensions of the wavefunctions """
#q = 2.
#
## position set
#s = np.arange(0,fib(n),1.)
#    
# smallest box size (in the form smallest_espilon = 10**(-t))
t = 3.5
# for some reason, there are problems with 3, so we exclude it from the list of primes
primes = [2,6,10]
epsilonRange = sorted([1/n**delta for n in primes for delta in range(0,int(t*math.log(10.)/math.log(n))+1)])
# n=16: choosing 4;-4 is good!
realRange = epsilonRange[4:-7]
#epsilonRange = 2**np.arange(-21.,0.,1.)

#a = label(304,n)
#w = abs(vec[:,a])**2
#qwList = [qWeight(s, w, epsilon, q) for epsilon in epsilonRange]
#coqwList = [qWeight(s, cow, epsilon, q) for epsilon in epsilonRange]

""" linear regression for every energy """
## loading wavefunctions
#n = 16
#vec = np.load("data/wf_conum_n16_rho0.1.npy")
#
## position set
#s = np.arange(0,fib(n),1.)
#
#q = 2
#w = abs(vec)**2
#    
#def linreg(a):
#    logWeights = [math.log(qWeight(s, w[:,a], epsilon, q)) for epsilon in realRange]
#    logEps = [math.log(eps) for eps in realRange]
#    
#    slope, intercept, r_value, p_value, std_err = stats.linregress(logEps,logWeights)
#    return slope
#
#""" computations """
#pool = Pool(4)
#dim = list(pool.map(linreg, range(fib(n))))
#
#""" plot dimension vs a at fixed q """
#plt.plot(dim,'o', markersize=3.)
## x(a)
#xList = np.array([x(a,n,0) for a in range(fib(n))])
#plt.plot(0.8*xList/7,'+',markersize=3.)
#plt.show()

""" box counting using Fibonacci boxes """

n = 16
vec = np.load("data/wf_conum_n16_rho0.1.npy")

# p: step of the subdivision, w: weight list, n: size
def chi(w, n, q, p):
    if(p == 0):
        return sum(w)**q
    else:
        # divide w for the subsequent summations
        w1 = w[:fib(n-2)]
        w2 = w[fib(n-2):fib(n-1)]
        w3 = w[fib(n-1):]
        
        p -= 1
        return chi(w1,n-2,q,p) + chi(w2,n-3,q,p) + chi(w3,n-2,q,p)

def gamma(w, n, q, tau, p):
    if(p == 0):
        return sum(w)**q/fib(n)**tau
    else:
        # divide w for the subsequent summations
        w1 = w[:fib(n-2)]
        w2 = w[fib(n-2):fib(n-1)]
        w3 = w[fib(n-1):]
        
        p -= 1
        return gamma(w1,n-2,q,p) + gamma(w2,n-3,q,p) + gamma(w3,n-2,q,p)

# compute the q-weight as a function of the number of boxes        
q = 2.
# weights
w = abs(vec)**2
# energy label
a = 23
# associated path
pa = path(a,n,'')
# list of steps
steps = range(0,len(pa)+1)
# list of q-weights
chiList = [chi(w[:,a],n,q,p) for p in steps]
logChi = [math.log(chi) for chi in chiList]

# evaluate the slope of log chi(n)
slope, intercept, r_value, p_value, std_err = stats.linregress(steps, logChi)
# evaluate the fit function
values = np.arange(steps[0], steps[-1]+1, .5)
logFit = [slope*i + intercept for i in values]
# plots!
plt.plot(values, logFit)
plt.plot(logChi,'o')
plt.title('The q-weight for the wavefunction ' + pa)

""" fit for every individual wf """

# the max number of steps is n/2 (every decimation is molecular)
steps = range(int(n/2)+1)
# for every wf, compute chi for the max number of steps possible
chis = np.ma.array([chi(w,n,q,p) for p in steps])
# compute all path lengthes
pths = np.array([len(path(a,n,'')) for a in range(fib(n))])
# mask the extra steps
for en in range(fib(n)):
    chis[pths[en]+1:,en] = np.ma.masked
# goin' log scale baby
log = np.vectorize(math.log)
logChis = log(chis)

# evaluate the slope of log chi(n)
fits = []
for en in range(fib(n)):
    fits.append(list(stats.linregress(range(0,pths[en]+1), logChis[:,en].compressed())[0:2]))
fits = np.array(fits)
# evaluate the fit function
values = np.arange(steps[0], steps[-1]+1, .5)
logFit = [fits[a,0]*i + fits[a,1] for i in values]
# plots!
plt.plot(values, logFit, '+')


""" data plot and linear regression """

def loglogplot(min, max):
    
    # plot
    epsilonRange2 = epsilonRange[min:max]
    qwList2 = qwList[min:max]

    plt.title('The q-weight for the wavefunction '+str(a)+' for q = ' + str(q))
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

def simple_plot(q, a, min, max):
    # plot
    epsilonRange2 = epsilonRange[min:max]
    qwList = [qWeight(s, w[:,a], epsilon, q) for epsilon in epsilonRange]
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