"""
微素粒子結合の最小トイモデルシミュレーション
"""
import numpy as np
try:
    from scipy.optimize import minimize
    use_scipy = True
except ImportError:
    use_scipy = False
import matplotlib.pyplot as plt
import os

def total_energy(x, params):
    N = params['N']
    thetas = x[:N]
    phis = x[N:2*N]
    k_theta = params['k_theta']
    k_phi = params['k_phi']
    k_I = params['k_I']
    theta0 = params['theta0']
    sigma_I = params['sigma_I']
    
    Is = np.zeros(N)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dth = thetas[i] - thetas[j]
            dth = (dth + np.pi) % (2*np.pi) - np.pi
            dphi = phis[i] - phis[j]
            dphi = (dphi + np.pi) % (2*np.pi) - np.pi
            E += k_theta * (-np.cos(dth - theta0))
            E += k_phi * (-np.cos(dphi))
            E += k_I * (-np.exp(- (Is[i]-Is[j])**2 / (sigma_I**2 + 1e-12)))
    return E

def optimize_energy(params, n_restarts=30):
    N = params['N']
    best = None
    best_x = None
    for seed in range(n_restarts):
        rng = np.random.RandomState(seed * 9973 + 13)
        x0 = np.concatenate([rng.uniform(0, 2*np.pi, N), rng.uniform(0, 2*np.pi, N)])
        
        if use_scipy:
            res = minimize(lambda x: total_energy(x, params), x0, method='Nelder-Mead',
                           options={'maxiter':2000, 'xatol':1e-8, 'fatol':1e-8, 'disp': False})
            x_opt = res.x
            E = res.fun
        else:
            x = x0.copy()
            curE = total_energy(x, params)
            step = 0.5
            for it in range(2000):
                i = rng.randint(0, 2*N)
                cand = x.copy()
                cand[i] += rng.normal(scale=step)
                candE = total_energy(cand, params)
                if candE < curE:
                    x = cand
                    curE = candE
                    step *= 0.9995
            x_opt = x
            E = curE
            
        if best is None or E < best:
            best = E
            best_x = x_opt.copy()
    return best_x, best

def plot_configuration(x_opt, E_opt, params, save_path):
    N = params['N']
    thetas_opt = x_opt[:N] % (2*np.pi)
    phis_opt = x_opt[N:2*N] % (2*np.pi)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title(f"Toy-model stable configuration (N={N})\nTotal energy = {E_opt:.6f}")
    
    r = np.ones(N)
    ax.scatter(thetas_opt, r, s=100)
    for i in range(N):
        j = (i+1)%N
        ax.plot([thetas_opt[i], thetas_opt[j]], [1,1], linestyle='-', linewidth=1)
    for i in range(N):
        ax.text(thetas_opt[i], 1.1, f"φ={phis_opt[i]:.2f}", ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
