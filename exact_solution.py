# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:05:50 2025

@author: majela.penton
"""

import numpy as np
from inertial_proj_split import InertialProjSpl_LASSO

def ExactSolution(A, At, b, x0, nu, maxIter):
    """
    Determina o valor ótimo do problema de Lasso associado usando 
    projective splitting paralelo.
    """
    # Dimensões da matriz
    m, n = A.shape
    
    # Inicialização de vetores
    z = np.zeros((n, 1))
    w = np.zeros((n, 1))
    
    # Constantes
    L = np.linalg.norm(At @ A, ord=np.inf)  # Norma infinita
    sigma = 0.24
    lambda_ = sigma / L
    mu = lambda_
    alpha = 0
    tol = 10**-10
    eta = 0
    beta = 1

    # Cálculo do valor ótimo inicial
    f_opt = 0.5 * np.linalg.norm(A @ x0 - b, 2)**2 + nu * np.linalg.norm(x0, 1)

    # Chamada da função InertialProjSpl_LASSO
    restore_prj_spl, fun_value_prj_spl, tic_prj_spl, err_prj_spl, whitoutuse = InertialProjSpl_LASSO(
        f_opt, A, At, b, nu, z, w, alpha, lambda_, mu, eta, beta, maxIter, tol
    )
    
    # Determinação do menor valor encontrado
    fOpt = min(fun_value_prj_spl)
    
    return fOpt
