import numpy as np
from src.simulation import optimize_energy

def test_optimize_energy_stable_configuration():
    params = {
        "N": 3, "k_theta": 1.0, "k_phi": 1.0, "k_I": 1.0,
        "theta0": 2.0943951023931953, "sigma_I": 0.5
    }
    x_opt, E_opt = optimize_energy(params, n_restarts=5)
    
    assert x_opt is not None
    assert isinstance(E_opt, float)
    assert E_opt < 0  # 安定構造は負のエネルギーを持つべき
