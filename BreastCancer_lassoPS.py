# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:57:10 2025

@author: Majela
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from exact_solution import ExactSolution
from inertial_proj_split import InertialProjSpl_LASSO

def load_data(filename='wdbc.data'):
    A = pd.read_csv(filename, header=None).values[:, 2:].astype(float)
    m, n = A.shape
    idx = np.arange(n)
    coef = (-1)**idx * np.exp(-idx / 10)
    coef[10:] = 0
    x0 = coef.reshape(-1, 1)
    A = A @ np.diag(1.0 / np.sqrt(np.sum(A**2, axis=0)))
    b = A @ x0 + np.sqrt(0.001) * np.random.randn(m, 1)
    b /= np.linalg.norm(b)
    return A, A.T, b, x0

def run_experiment(alphas, A, At, b, x0, lambda_, mu, eta, beta, tol, max_iter, nu):
    f_opt = ExactSolution(A, At, b, x0, nu, 1000)
    results = []

    for alpha in alphas:
        for name, inertial_eta in zip(['IR-PS', 'PS'], [eta, 0]):
            z = np.zeros((A.shape[1], 1))
            w = np.zeros((A.shape[1], 1))
            z_out, fval, time_elapsed, err, res = InertialProjSpl_LASSO(
                f_opt, A, At, b, nu, z, w, alpha, lambda_, mu, inertial_eta, beta, max_iter, tol
            )
            results.append({
                'method': name,
                'alpha': alpha,
                'time': time_elapsed,
                'iters': len(fval),
                'final_error': err[-1],
                'final_obj': fval[-1],
                'final_res': min(res)
            })
    return pd.DataFrame(results)

def plot_bars(df):
    pivot_time = df.pivot(index='alpha', columns='method', values='time')
    pivot_iters = df.pivot(index='alpha', columns='method', values='iters')

    for metric, ylabel in zip([pivot_time, pivot_iters], ['Execution Time (s)', 'Number of Iterations']):
        metric.plot(kind='bar', figsize=(10, 5))
        plt.title(ylabel + ' per Alpha')
        plt.ylabel(ylabel)
        plt.xlabel('Alpha')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

def main():
    #np.random.seed(0)
    A, At, b, x0 = load_data()
    L = np.linalg.norm(At @ A)
    sigma = 0.24
    lambda_ = sigma / L
    mu = lambda_
    eta = 0.5
    eta_bar = eta + 0.07
    beta = 2 * (eta_bar - 1)**2 / (2 * (eta_bar - 1)**2 + 3 * eta_bar - 1)
    tol = 1e-4
    max_iter = 5000
    nu = 0.1 * np.linalg.norm(At @ b, ord=np.inf)
    
    # Alphas from the paper
    alphas = [1, -1, 0, 0.8147, 0.1270, 0.6324, -0.2785, -0.5469, -0.9575, -0.3584]
    
    # Random alphas
    #alphas = np.random.uniform(-1, 1, size=12)
    
    
    df_results = run_experiment(alphas, A, At, b, x0, lambda_, mu, eta, beta, tol, max_iter, nu)
    df_results.to_csv('performance.csv', index=False)
    plot_bars(df_results)

if __name__ == "__main__":
    main()
