import numpy as np
from src.acim_cosmology import ACIM_v14_Cosmology, ACIM_v15_CMB_Fitter

def test_v14_cosmology_engine():
    alpha = 9.5785e-6
    engine = ACIM_v14_Cosmology(alpha=alpha)
    
    # 現在 (a=1) のハッブル係数比は近似的に1であるべき
    E_today = engine.get_E(1.0)
    assert np.isclose(E_today, 1.0, atol=0.1)
    
    # 過去 (a < 1) では O_t が0より大きいべき
    O_t_past = engine._get_O_t(0.5)
    assert O_t_past > 0.0

def test_v15_cmb_fitter_initialization():
    # ダミーのCMBデータ
    dummy_data = {
        'L': np.array([2, 10, 100, 500, 1000]),
        'TT': np.array([1000.0, 5000.0, 6000.0, 2000.0, 500.0])
    }
    alpha = 9.5785e-6
    fitter = ACIM_v15_CMB_Fitter(dummy_data, alpha_v10b=alpha)
    
    assert fitter.baseline_spline is not None
    assert fitter.Cl_info_template is not None
    assert len(fitter.Cl_info_template) == len(dummy_data['L'])
