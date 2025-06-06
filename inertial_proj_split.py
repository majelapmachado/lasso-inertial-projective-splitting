# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:08:27 2025

@author: majela.penton
"""

import numpy as np
import time

def InertialProjSpl_LASSO(f_opt, A, At, f, nu, z, w, alpha, lambda_, mu, eta, beta, maxIter, tol):
    """
    Inertial Projective Splitting for LASSO:
    min 1/2 * ||Ax - b||^2 + nu * |x|_1
    
    INPUT:
        f_opt - optimal value
        A, At - matrix and its transpose
        f - ovserved vector (b)
        nu - regularization L1
        z, w - initial points
        alpha - sequential parameter
        lambda_ - step size for f-subproblem
        mu - step size for g-subproblem
        eta - inertial parameter
        beta - relaxation parameter
        maxIter - maximum number of iterations
        tol - error tolerance
    
    OUTPUT:
        restore - approximation of optimal solution 
        fun_value - vector of functional values
        tic_psm - total time
        err_psm - vector of errors
        res_psm - vector of residuals
    """
    # Inicialization
    start_time = time.time()
    fun_value = np.zeros(maxIter)
    err_psm = np.zeros(maxIter)
    res_psm = np.zeros(maxIter)
    
    err = tol + 1
    k = 1
    z_old = np.copy(z)
    w_old = np.copy(w)

    while k <= maxIter and err > tol:
        # Inertial Step
        bar_z = z + eta * (z - z_old)
        bar_w = w + eta * (w - w_old)

        # Subproblem f
        u = At @ (A @ bar_z - f)  # Gradient f(bar_z)
        x = bar_z - lambda_ * (u - bar_w)
        b = At @ (A @ x - f)  # Gradient f(x)

        # Subproblem g
        aux = (1 - alpha) * bar_z + alpha * x + mu * bar_w
        y = np.maximum(np.abs(aux) - nu * mu, 0) * np.sign(aux)  # Proxy mu * g
        a = (1 / mu) * (aux - y)

        # Theta
        aux_1 = a + b
        aux_2 = y - x
        phi = np.sum((bar_z - x) * aux_1) + np.sum(aux_2 * (bar_w - a))
        pi = np.linalg.norm(aux_1, 2)**2 + np.linalg.norm(aux_2, 2)**2
        theta = phi / pi

        # Update
        z_old = np.copy(z)
        w_old = np.copy(w)
        z = bar_z - beta * theta * aux_1
        w = bar_w - beta * theta * aux_2

        # Loss, errors and residual
        fun_value[k - 1] = 0.5 * np.linalg.norm(A @ z - f, 2)**2 + nu * np.linalg.norm(z, 1)
        err = abs((f_opt - fun_value[k - 1]) / f_opt)
        err_psm[k - 1] = err
        res_psm[k - 1] = max([np.linalg.norm(aux_1), np.linalg.norm(aux_2)])

        k += 1

    # Reduce vectors to relevant values
    restore = z
    fun_value = fun_value[:k - 1]
    err_psm = err_psm[:k - 1]
    res_psm = res_psm[:k - 1]
    tic_psm = time.time() - start_time

    return restore, fun_value, tic_psm, err_psm, res_psm
