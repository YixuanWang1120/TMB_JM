# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:55:39 2022

@author: wyx
"""
import numpy as np
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