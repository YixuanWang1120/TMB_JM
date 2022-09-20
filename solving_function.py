# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:58:41 2022

@author: wyx
"""

from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import math
from scipy.optimize import fsolve
from sklearn import metrics
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Datasets simulation
# ------------------------------------------------------------------------------
alpha_simu = -1.8
alpham_simu = 0.3
lamda_simu = 1.0
beta_simu = 2.2
betam_simu = -0.4
omega_simu = -1.0
sigma_simu = 1.0
sigma_err = 0.5
n = 200
num = 7
z = np.identity(n)
TMB = np.random.normal(3.0, 1.0, (n, 1))
err = np.random.normal(0, sigma_err, (n, 1))
TMB_E = TMB + err
X = np.random.rand(n).reshape(-1, 1)
b_simu = np.random.normal(0, sigma_simu, (n, 1))
yetaR_simu = alpha_simu*X + alpham_simu*TMB + b_simu
yetaT_simu = beta_simu*X + betam_simu*TMB + omega_simu*b_simu
P0 = 1 / (1 + np.exp(yetaR_simu))
R = np.zeros((n, 1))
for i in range(n):
    if np.random.rand(1) > P0[i]:
        R[i] = 1
T_simu = np.zeros((n, 1))
C = np.random.uniform(0, 3, n)
for i in range(n):
    T_simu[i] = (-np.log(np.random.rand(1)) /
                 (np.exp(yetaT_simu[i]))**(1/lamda_simu))
D = np.zeros((n, 1))
T = np.zeros((n, 1))
for i in range(n):
    D[i] = 1 if T_simu[i] <= C[i] else 0
    T[i] = T_simu[i] if D[i] == 1 else C[i]
    
# ------------------------------------------------------------------------------
# Monte Carlo
# ------------------------------------------------------------------------------
K = 10
TMB_C = np.zeros((n, K), dtype=complex)
for k in range(K):
    TMB_C[:, k] = (TMB_E + np.random.normal(0,
                   sigma_err, (n, 1))*1j).reshape(-1)
#------------------------------------------------------------------------------
# Joint modeling 
# ------------------------------------------------------------------------------
def p1(alpha, alpha_m, b, TMB):
    p1 = ((np.exp(np.dot(X, alpha) + alpha_m*TMB + b))**R) / \
        (1+np.exp(np.dot(X, alpha) + alpha_m*TMB + b))
    return p1

def p2(lamda, beta, beta_m, b, TMB):
    p2 = ((lamda*(T**(lamda-1))*np.exp(np.dot(X, beta) + beta_m*TMB + b))**D)\
        * (np.exp(-T**lamda*np.exp(np.dot(X, beta) + beta_m*TMB + b)))
    return p2

def p3(sigma, b):
    p3 = ((2*math.pi)**0.5*sigma)**(-1)*np.exp(-(1/2)*(b/sigma)**2)
    return p3

def k_b(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB):
    k = np.log(p1(alpha,alpha_m,b,TMB))+np.log(p2(lamda,beta,beta_m,b,TMB))+np.log(p3(sigma,b))
    return k

def F(x):
    return ((1+np.exp(-x))**(-1))

def d_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB):
    d_k = R-F(np.dot(X, alpha)+alpha_m*TMB+b)\
         + D-T**lamda*np.exp(np.dot(X, beta)+beta_m*TMB+b)\
         - b/(sigma**2)
    return d_k

def Re_F(alpha, alpha_m, b, TMB_C):
    sc = 0
    for i in range(K):
        sc += F(np.dot(X, alpha)+alpha_m*(TMB_C[:, i].reshape(-1, 1))+b).real
    return sc/K

def Re_F_db(alpha, alpha_m, b, TMB_C):
    df = np.zeros((n, 1))
    db = 1.0e-4
    b1 = np.copy(b)
    b1 = b1+db  # x+dx
    df = (Re_F(alpha, alpha_m, b1, TMB_C)
          - Re_F(alpha, alpha_m, b, TMB_C))/db  # f(x+dx)-f(x)/dx
    return df

def Re(alpha, alpha_m, b, TMB_C):
    sc = 0
    for i in range(K):
        sc += (F(np.dot(X, alpha)+alpha_m *
               (TMB_C[:, i].reshape(-1, 1))+b)*(TMB_C[:, i].reshape(-1, 1))).real
    return sc/K

def dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB):
    dd_k = -np.exp(np.dot(X, alpha)+alpha_m*TMB+b)/(1+np.exp(np.dot(X, alpha)+alpha_m*TMB+b))**2\
        - T**lamda*np.exp(np.dot(X, beta)+beta_m*TMB+b)\
        - 1/(sigma**2)
    return dd_k

def d_k_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB_E, TMB_C):
    d_k = R-Re_F(alpha, alpha_m, b, TMB_C)\
        + D-T**lamda*np.exp(np.dot(X, beta)+beta_m*TMB_E+b)*np.exp(-(sigma_err**2)*(beta_m**2)/2)\
        - b/(sigma**2)
    return d_k

def dd_k_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB_E, TMB_C):
    dd_k = -Re_F_db(alpha, alpha_m, b, TMB_C)\
        - T**lamda*np.exp(np.dot(X, beta)+beta_m*TMB_E+b)*np.exp(-(sigma_err**2)*(beta_m**2)/2)\
        - 1/(sigma**2)
    return dd_k

def kk_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB_E, TMB_C):
    dd_k = -1/2*np.log(-dd_k_c(alpha,alpha_m,lamda,beta,beta_m,sigma,b,TMB_E,TMB_C))
    return np.sum(dd_k)

def dfun_kk_c(x,b,TMB_E,TMB_C):
    df = np.zeros(num, dtype=float)
    dx = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):              # differential
        x1 = np.copy(x)
        x1[i] = x1[i]+dx  # x+dx
        df[i] = (kk_c(x1[0],x1[1],x1[2],x1[3],x1[4],x1[5],b,TMB_E,TMB_C)-
                 kk_c(x[0],x[1],x[2],x[3],x[4],x[5],b,TMB_E,TMB_C))/dx  # f(x+dx)-f(x)/dx
    return df

def ddfun_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB_E, TMB_C):
    df = np.zeros((n, 1))
    db = 1.0e-4
    b1 = np.copy(b)
    b1 = b1+db  # x+dx
    df = (d_k_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b1, TMB_E, TMB_C)
          - d_k_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB_E, TMB_C))/db  # f(x+dx)-f(x)/dx
    return df

# ------------------------------------------------------------------------------
# Newton-Raphson
# ------------------------------------------------------------------------------
def Newton_b(theta, b, TMB):
    b1 = np.copy(b)
    j = 0
    delta = 5
    while(np.sum(abs(delta)) > 1.e-3 and j < 500):
        b1 = b-d_k(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], b, TMB)\
            /dd_k(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], b, TMB)
        delta = b1-b
        b = b1
        j = j+1
    return b

def Newton_c(theta, b, TMB_E, TMB_C):
    b1 = np.copy(b)
    j = 0
    delta = 5
    while(np.sum(abs(delta)) > 1.e-3 and j < 500):
        b1 = b-d_k_c(theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],b,TMB_E,TMB_C)\
            /dd_k_c(theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],b,TMB_E,TMB_C)
        delta = b1-b
        b = b1
        j = j+1
    return b

def ML(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB):
    l = 1/2*np.log(2*math.pi)+np.log(p1(alpha, alpha_m, b, TMB))+np.log(p2(lamda, beta, beta_m, b,TMB))\
        + np.log(p3(sigma, b))-1/2*np.log(abs(dd_k(alpha,alpha_m,lamda,beta,beta_m,sigma,b,TMB)))
    L = np.sum(l)
    return L

def dfun2(x, b, TMB):
    df = np.zeros(num, dtype=float)
    dx = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):              # differential
        x1 = np.copy(x)
        x1[i] = x1[i]+dx  # x+dx
        df[i] = (ML(x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], b, TMB) -
                 ML(x[0], x[1], x[2], x[3], x[4], x[5], b, TMB))/dx  # f(x+dx)-f(x)/dx
    return df

def Score(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB):
    Score_alpha = (R-F(np.dot(X, alpha)+alpha_m*TMB+b))*X\
        - 1/2*((np.exp(np.dot(X, alpha)+alpha_m*TMB+b)*(1-np.exp(np.dot(X, alpha)+alpha_m*TMB+b))
                /(1+np.exp(np.dot(X, alpha)+alpha_m*TMB+b))**3)*X
               /abs(dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB)))

    Score_alpham = (R-F(np.dot(X, alpha)+alpha_m*TMB+b))*TMB\
        - 1/2*((np.exp(np.dot(X, alpha)+alpha_m*TMB+b)*(1-np.exp(np.dot(X, alpha)+alpha_m*TMB+b))
                /(1+np.exp(np.dot(X, alpha)+alpha_m*TMB+b))**3)*TMB
               /abs(dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB)))

    Score_lamda = (D*(1/lamda+np.log(T))-T**lamda*np.log(T)*np.exp(beta*X+beta_m*TMB+b))\
        - 1/2*(T**lamda*np.log(T)*np.exp(beta*X+beta_m*TMB+b)
               / abs(dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB)))

    Score_beta = (D-T**lamda*np.exp(beta*X+beta_m*TMB+b))*X\
        - 1/2*(T**lamda*np.exp(beta*X+beta_m*TMB+b)*X
               /abs(dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB)))

    Score_betam = (D-T**lamda*np.exp(beta*X+beta_m*TMB+b))*TMB\
        - 1/2*(T**lamda*np.exp(beta*X+beta_m*TMB+b)*TMB
               / abs(dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB)))

    Score_thetab = -sigma**(-1)+(b**2)*(sigma**(-3))+(sigma**(-3)) / \
        abs(dd_k(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB))

    Score = np.array((sum(Score_alpha), sum(Score_alpham), sum(Score_lamda), sum(
        Score_beta), sum(Score_betam), sum(Score_thetab)))
    return (Score.reshape(-1))

def Score_c(alpha, alpha_m, lamda, beta, beta_m, sigma, b, TMB_E, TMB_C):
    
    Score_alpha = R*X-Re_F(alpha,alpha_m,b,TMB_C)*X

    Score_alpham = R*TMB_E-Re(alpha,alpha_m,b,TMB_C)

    Score_lamda = D*(1/lamda+np.log(T))-T**lamda*np.log(T)\
                   *np.exp(beta*X+beta_m*TMB_E+b)*np.exp(-(sigma_err**2)*(beta_m**2)/2)

    Score_beta = (D-T**lamda*np.exp(beta*X+beta_m*TMB_E+b)
                  *np.exp(-(sigma_err**2)*(beta_m**2)/2))*X
        
    Score_betam = D*TMB_E-T**lamda*np.exp(beta*X+beta_m*TMB_E+b)*np.exp(-(sigma_err**2)*(beta_m**2)/2)\
                    *(TMB_E-(sigma_err**2)*beta_m)
        
    Score_thetab = -sigma**(-1)+(b**2)*(sigma**(-3))

    Score1 = np.array((sum(Score_alpha), sum(Score_alpham), sum(Score_lamda), sum(Score_beta), sum(Score_betam), sum(Score_thetab)))
    Score2 = dfun_kk_c([alpha, alpha_m, lamda, beta, beta_m, sigma],b,TMB_E,TMB_C)
    Score = Score1.reshape(-1)+Score2
    return (Score.reshape(-1))

def ddfun2(x, b, TMB):
    df = np.zeros((num, num), dtype=float)
    h = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):
        for j in range(i, num):
            x1 = np.copy(x)
            x1[j] = x1[j]+h
            df[i, j] = (Score(x1[0],x1[1],x1[2],x1[3],x1[4],x1[5],b,TMB)[i]
                        - Score(x[0],x[1],x[2],x[3],x[4],x[5],b,TMB)[i])/h
    df += df.T - np.diag(df.diagonal())
    return df

def ddfun2_c(x, b, TMB_E, TMB_C):
    df = np.zeros((num, num), dtype=float)
    h = 1.0e-4
    x1 = np.copy(x)
    for i in range(num):
        for j in range(i, num):
            x1 = np.copy(x)
            x1[j] = x1[j]+h
            df[i, j] = (Score_c(x1[0],x1[1],x1[2],x1[3],x1[4],x1[5],b,TMB_E,TMB_C)[i]
                        - Score_c(x[0],x[1],x[2],x[3],x[4],x[5],b,TMB_E,TMB_C)[i])/h
    df += df.T - np.diag(df.diagonal())
    return df

def solve_function(t):
    return (Score(t[0],t[1],t[2],t[3],t[4],t[5],b_it,TMB))

def solve_function_c(t):
    return (Score_c(t[0], t[1], t[2], t[3], t[4], t[5], b_it_err, TMB_E, TMB_C))

#%% ------------------------------------------------------------------------------
# Solving score functions
# ------------------------------------------------------------------------------
theta = np.array([-1.8,0.3,1.0,2.2,-0.4,1.0])
theta0 = np.array([-1.8,0.3,1.0,2.2,-0.4,1.0])
delta = theta
j = 0
while(sum(abs(delta)) > 1.e-4 and j < 1000):
    b_it = Newton_b(theta, b_simu, TMB)
    solve_t = fsolve(solve_function,theta0,full_output=1)
    if solve_t[2] == 1:
        theta_it = solve_t[0]
    else:
        print('fail')
        break    
    delta = theta_it-theta
    theta = theta_it
    #print(ML(theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],b_it,TMB))
    j += 1

theta_var = np.linalg.inv(-ddfun2(theta, b_it, TMB))
var = [(theta_var[i][i])**0.5 for i in range(num)]

theta_err = theta
delta_err = theta
j = 0
while(sum(abs(delta_err)) > 1.e-4 and j < 1000):
    b_it_err = Newton_c(theta_err, b_simu, TMB_E, TMB_C)
    solve_t_c = fsolve(solve_function_c,theta, full_output=1, xtol=1.0e-02)
    if solve_t_c[2] == 1:
        theta_err_it = solve_t_c[0]
    else:
        print('fail')
        break
    #print(ML(theta_err[0],theta_err[1],theta_err[2],theta_err[3],theta_err[4],theta_err[5],b_it_err,TMB_E))
    delta_err = theta_err_it-theta_err
    theta_err = theta_err_it
    j += 1

theta_var_err = np.linalg.inv(-ddfun2(theta_err, b_it_err, TMB))
var_err = [(theta_var_err[i][i])**0.5 for i in range(num)]

#%% ------------------------------------------------------------------------------
# Determining TMB threshold
# ------------------------------------------------------------------------------
cph = CoxPHFitter()
clf_lg = LogisticRegression(fit_intercept=0)
T0 = np.median(T)

def P_R(alpha, alpha_m, b, TMB):
    p1 = ((1+np.exp(-np.dot(X, alpha)-alpha_m*TMB-b))**(-1))
    return p1

def P_T(lamda, beta, beta_m, b, TMB):
    p2 = (np.exp(-T0**lamda*np.exp(np.dot(X, beta) + beta_m*TMB + b)))
    return p2

def P_J(alpha, alpha_m, lamda, beta, beta_m, sigma, b,TMB):
    f = P_R(alpha, alpha_m, b_it_err, TMB)*P_T(lamda, beta,
                                               beta_m, b_it_err, TMB)*p3(sigma, b_it_err)
    return f

pro = P_J(theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],b_it,TMB)
pro_err = P_J(theta_err[0],theta_err[1],theta_err[2],theta_err[3],theta_err[4],theta_err[5],b_it_err,TMB_E)

label = np.zeros(n,dtype=int)
label_err = np.zeros(n,dtype=int)
k1 = 30
k2 = 30
for i in range(n):
    if pro[i] > np.percentile(pro,k1):
        label[i] = 1
    else:
        label[i] = 0
for i in range(n):
    if pro_err[i] > np.percentile(pro_err,k2):
        label_err[i] = 1
    else:
        label_err[i] = 0

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = np.zeros(len(TPR))
    for i in range(len(TPR)):
        y[i] = TPR[i] - FPR[i]
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC(label, y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return roc_auc, optimal_th, optimal_point
# ROC curves
#------------------------------------------------------------------------------ 
print('------------preform the dichotomy------------')
fpr, tpr, thresholds = metrics.roc_curve(label,TMB,drop_intermediate=False)
roc_auc, cutoff_JM, optimal_point = ROC(label,TMB)
print('the dichotomy threshold is: ', cutoff_JM)
print('------------preform the dichotomy for TMB with measurement error------------')
fpr_err, tpr_err, thresholds_err = metrics.roc_curve(label_err,TMB_E,drop_intermediate=False)
roc_auc_err, cutoff_JM_err, optimal_point_err = ROC(label_err,TMB_E)
print('the dichotomy threshold is: ', cutoff_JM_err)
