#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.linalg as lg
import warnings
from scipy.linalg import null_space,pinv,inv
from scipy.optimize import newton
from ProgressBar import ProgressBar


# # Overview
#
# Here we will have $p(x,y)$, $p(x|y)$, and $d(x,y)$ as matrices, of the form
# $$
# P_{xy} =
# \left[\begin{array}{cccc}
# p(x=0,y=0) & p(x=1,y=0) & \cdots & p(x=n_x-1,y=0) \\
# p(x=0,y=1) & p(x=1,y=1) & \cdots & p(x=n_x-1,y=1) \\
# \vdots & \vdots & \ddots & \vdots \\
# p(x=0,y=n_y-1) & p(x=1,y=n_y-1) & \cdots & p(x=n_x-1,y=n_y-1)
# \end{array}\right]
# $$
# where $n_x$ and $n_y$ are, respectively, the alphabet sizes of $x$ and $y$. We will specify $p(x|y)\rightarrow P_{x|y}$ and $d(x,y)\rightarrow D_{xy}$ in the same order as above.
#
# This will require matrix stacking and destacking, and methods will be written to accomplish this.


# this will return 0 if p < 0 or p > 1, and will print a warning if the value exceeds a tolerance
def phi(p,bits=False,debug=False):
    tol = 1e-14
    if p <= 0:
        if (np.abs(p) > tol) and (debug is True):
            warnings.warn('in phi: Exceeds input tolerance (' + str(np.abs(p)) + ' vs. ' + str(tol) +')')
        return 0
    if p >= 1:
        if (p-1 > tol) and (debug is True):
            warnings.warn('in phi: Exceeds input tolerance (' + str(p-1) + ' vs. ' + str(tol) +')')
        return 0
    if bits is True:
        return p*np.log2(1/p)
    else:
        return p*np.log(1/p)


def stack(M):
    (ny,nx) = np.shape(M)
    result = np.zeros(nx*ny)
    for i in range(ny):
        result[(i*nx):((i+1)*nx)] = M[i,:]
    return result

def unstack(v,ny,nx):
    M = np.zeros((ny,nx))
    for i in range(ny):
        M[i,:] = v[(i*nx):((i+1)*nx)]
    return M


# Calculates the mutual information in the joint distribution Pxy
def MI(Pxy,bits=False):
    (ny,nx) = np.shape(Pxy)
    px = np.ones(ny) @ Pxy
    Hx = np.sum([phi(i,bits=bits) for i in px])
    py = Pxy @ np.ones(nx)
    Hy = np.sum([phi(i,bits=bits) for i in py])
    Hxy = np.sum([phi(float(i),bits=bits) for i in np.nditer(Pxy)])
    return Hx + Hy - Hxy

def h_x_given_y(Pxy):
    (ny,nx) = np.shape(Pxy)
    py = Pxy @ np.ones(nx)
    result = np.zeros((ny,nx))
    for i in range(ny):
        result[i,:] = Pxy[i,:] / py[i]
    return result

# here we will pass the diagonal equivalent reward matrix Q to the function
def getAvgLambdaStar(Q,px):
    return np.sum(np.log(np.diag(Q)) * px)

def hx(px):
    return np.sum([phi(i) for i in px])

# Implements the EM-based algorithm from:
# M. Hayashi, "Bregman divergence based EM algorithm and its application to
# classical and quantum rate distortion theory," IEEE Trans. Info. Thy., 2023.
def getRD(px,dxy,max_iter=1000,epsilon=1e-8,numPoints=1000,show_pb=False,py0=None):
    (minD,maxD) = minmaxAverageDistortion(px,dxy)
    D_v = np.linspace(minD,maxD,numPoints)
    D_v = D_v[1:-2]

    result = np.zeros(len(D_v))
    p = []
    zero_p_index = -1

    if py0 is None:
        (ny,nx) = np.shape(dxy)
        py0 = np.ones(ny)/ny

    if show_pb is True:
        pb = ProgressBar(len(D_v),40)
    for zzz in range(len(D_v)):
        py = py0
        qqq = 0
        keep_going = True
        while keep_going is True:
            tbar = newton(dfunc,0,fprime=dfunc_prime,args=(px,py,dxy,D_v[zzz]))
            pxy = np.zeros((len(py),len(px)))
            for i in range(len(px)):
                for j in range(len(py)):
                    num = px[i] * py[j] * np.exp(tbar * dxy[j,i])
                    den = np.sum(py*np.exp(tbar * dxy[:,i]))
                    pxy[j,i] = num/den
            py = np.sum(pxy,axis=1) # next py

            if qqq >= 1:
                last_mi = mi

            mi = MI(pxy)

            if (qqq >= 2) and (np.abs(last_mi - mi) < epsilon):
                keep_going = False
            qqq += 1
            if qqq >= max_iter:
                keep_going = False

        if (zzz > 0) and (result[zzz-1] < mi):
            result[zzz] = 0
            if (zero_p_index == -1):
                zero_p_index = zzz-1
            p.append(np.array(p[zero_p_index])) # copy
        else:
            #print((zzz > 1,result[zzz-1] < mi,result[zzz-1],mi))
            #print(result[zzz-1])
            result[zzz] = mi
            p.append(stack(pxy))

        if show_pb is True:
            pb.iterate()

    rd = {}
    rd['Dmax_v'] = D_v
    rd['r_v'] = result
    rd['p'] = p

    if show_pb is True:
        pb.hide()

    return rd

# the following three functions (down to getRD_BA) are used by the EM-based R(D) method above
def dfunc(x,px,py,dxy,d):
    dresult = 0
    for i in range(len(px)):
        num = 0
        den = 0
        for j in range(len(py)):
            num += py[j]*dxy[j,i]*np.exp(x*dxy[j,i])
            den += py[j]*np.exp(x*dxy[j,i])
        dresult += px[i] * (num/den)
    return dresult-d

def dfunc_prime(t,px,py,dxy,d):
    result = 0
    for i in range(len(px)):
        a = 0
        da = 0
        b = 0
        db = 0
        for j in range(len(py)):
            q = py[j] * np.exp(t * dxy[j,i])
            a += q * dxy[j,i]
            da += q * dxy[j,i] * dxy[j,i]
            b += q
            db += q * dxy[j,i]
        result += px[i] * (b * da - a * db)/(b*b)
    return result

def minmaxAverageDistortion(px,dxy):
    (ny,nx) = np.shape(dxy)
    resultMin = 0
    resultMax = 0
    for i in range(nx):
        resultMin += px[i]*np.min(dxy[:,i])
        resultMax += px[i]*np.max(dxy[:,i])
    return (resultMin,resultMax)

# Gets the rate-distortion function using the Blahut-Arimoto method from Blahut's original paper
# Faster than the EM-based method but doesn't work with all the results, particularly where the
# slope of R(D) is constant
def getRD_BA(px,dxy,smin=-10,smax=0,q0=None,num_iter=1000000,ba_tol=1e-8,numPoints=100,show_pb=False):
    s_v = np.linspace(smin,smax,numPoints)
    rate_v = np.zeros(len(s_v))
    dist_v = np.zeros(len(s_v))
    pxy_v = []
    (ny,nx) = np.shape(dxy)
    if q0 is None:
        q = np.ones(ny)/ny
    else:
        q = q0
        
    if show_pb is True:
        pb = ProgressBar(len(s_v),40)

    for i in range(len(s_v)):

        s = s_v[i]

        A = np.exp(s * dxy)

        converge = False
        last_rr = -1*np.inf
        last_dd = np.inf
        j = 0

        while j < num_iter and converge == False:
            den = np.sum(np.diag(q) @ A,axis=0)
            c = np.sum(A @ np.diag(px/den),axis=1)
            q = q * c

            den = np.sum(np.diag(q) @ A,axis=0)
            Qxy = np.diag(q) @ A @ np.diag(1/den)
            pxy = Qxy @ np.diag(px)

            # test convergence
            rr = MI(pxy)
            dd = np.sum(dxy*pxy)
            if (abs(rr-last_rr) < ba_tol) or (abs(dd-last_dd) < ba_tol):
                converge = True
            else:
                last_rr = rr
                last_dd = dd
                j += 1

        rate_v[i] = rr
        dist_v[i] = dd
        pxy_v.append(stack(pxy))
        
        if show_pb is True:
            pb.iterate()

    r = {}
    r['Dmax_v'] = dist_v
    r['r_v'] = rate_v
    r['p'] = pxy_v
    
    if show_pb is True:
        pb.hide()

    return r

# gets the equivalent strategy matrix t from the reward matrix R and the actual strategy s
# R is a (number of phenotypes) x (number of outcomes) matrix
# s is a (cardinality of side information) x (number of phenotypes) matrix
# Dxy = s @ R gives the growth rate (actual, not log), where D[i,j] = reward for side info i and outcome j
# note: this is the same order as given at the top (but CHECK TO MAKE SURE!!)
def getT(R,s):
    (rows,cols) = np.shape(R)
    q = pinv(R) @ np.ones(rows)
    Qinv = np.diag(q)
    B = R @ Qinv
    return s @ B


def getDistortionFunction(R,s):
    return -1*np.log(getT(R,s))

# returns the xth diagonal element of Q
def getLambdaStarX(R,x=None):
    (rows,cols) = np.shape(R)
    if rows == cols:
        q = inv(R) @ np.ones(rows)
    else:
        q = pinv(R) @ np.ones(rows)
    Qinv = np.diag(q)
    Q = lg.inv(Qinv)
    if x is None:
        return np.log(np.diag(Q))
    return np.log(Q[x,x])
    #Qinv = np.diag((np.linalg.inv(R) @ (np.array([np.ones(cols)]).T))[:,0])
    #Q = np.linalg.inv(Qinv)
    #if x is None:
    #  return np.log(np.diag(Q))
    #return np.log(Q[x,x])

def getFullDistortionFunction(R,s):
    (rows,cols) = np.shape(R)
    Dxy = getDistortionFunction(R,s)
    for i in range(cols):
        Dxy[:,i] -= getLambdaStarX(R,i)
    return Dxy