# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:07:16 2015

@author: nicolas
"""

import numpy as np
from scipy import linalg, stats
import matplotlib.pyplot as plt
import math
from cmath import exp
from timeit import default_timer as timer

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

""" compute the eigenvalues """

# diagonalize
rho = 1.
n = 12
L = fib(n)
val, vec = linalg.eigh(hp(n, rho))

# in conumbering!
#covec = conum(vec,n)

""" compute the propagator """

# list of on-site presence probabilities at time t, starting localized at site x
def I(t, x):
    psi0 = vec[x]
    expH = np.array([exp(- 1j*t*e) for e in val])
    return abs(vec.dot(expH*psi0))**2
    
""" compute the q-average at time t, starting site orig """

def P(orig,t,q):
    dist = np.array([(min(abs(x-orig),L-abs(x-orig))/float(L))**q for x in range(L)])
    return dist.dot(I(t,orig))
    
tRange = 10**np.arange(-1,4,.2)
#tRange = np.arange(0,30,.2)
orig = 0
q = 2.
#plist = [P(orig,t,q) for t in tRange]
#p2 = [P(0,t,q) for t in tRange]

""" average over all starting sites """

# matrix of the q-distances: D_{ij}(q) = |x_i-x_j|^q
q_dists = np.fromfunction(lambda j,i: (np.minimum(abs(i-j),L-abs(i-j))/float(L))**q, (L, L))
# average at time t
def avP(t,dists):
    # construct the matrix of intensities at time t
    Is = np.array([I(t,j) for j in range(L)])
    
    return np.trace(np.dot(Is, dists))/L
    
# brute force averaging (for testing purposes)
def brute_avP(t,q):
    avp = 0
    for x in range(L):
        avp += P(x,t,q)
    return avp/L
    
""" timing tests """

start = timer()
p1 = [avP(t,q_dists) for t in tRange]
elapsed = timer() - start
print("averaged diffuction (matrix multiplications): ",elapsed)

start = timer()
p2 = [brute_avP(t,2) for t in tRange]
elapsed = timer() - start
print("averaged diffuction (simple iteration): ",elapsed)

# TODO: average over some positions (randomly picked?) instead of averaging over all positions

def fit(pmin,pmax):
    tRangeFit = tRange[pmin:pmax]
    plistFit = plist[pmin:pmax]
    
    plt.loglog(tRange, plist, '-,b')
    plt.loglog(tRangeFit, plistFit, '-,r')
    plt.show()
    
    # data in log-log scale
    logP = [math.log(p) for p in plistFit]
    logT = [math.log(t) for t in tRangeFit]
    slope, intercept, r_value, p_value, std_err = stats.linregress(logT,logP)
    
    return slope/q, r_value