"""
ACIM v14/v15 宇宙論エンジンおよびCMBフィッター
"""
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
import sys

class ACIM_v14_Cosmology:
    c = 2.9979e8 
    H0_seconds = 2.2e-18 
    Omega_m0 = 0.31 
    Omega_r0 = 9.2e-5 
    Omega_L0 = 0.69 
    epsilon = 1e-10

    def __init__(self, alpha: float):
        if alpha < 0:
            print(f"警告: v14エンジンが負のalpha={alpha}で初期化されました｡", file=sys.stderr)
        self.alpha = alpha

    def _get_O_t(self, a: float) -> float:
        if a < 1e-100:
            a = 1e-100
        delta_obs = self.alpha / a
        return delta_obs / (1.0 + delta_obs)

    def calculate_E_squared(self, a: float) -> float:
        O_t = self._get_O_t(a)
        omega_m_current = self.Omega_m0 * (a ** (-3.0))
        omega_r_current = self.Omega_r0 * (a ** (-(4.0 - O_t)))
        return omega_r_current + omega_m_current + self.Omega_L0

    def get_E(self, a: float) -> float:
        E_sq = self.calculate_E_squared(a)
        if E_sq <= 0 or not np.isfinite(E_sq):
            return 0.0 
        return np.sqrt(E_sq)

class ACIM_v15_CMB_Fitter:
    def __init__(self, cmb_data_dict: dict, alpha_v10b: float):
        self.alpha_v10b = alpha_v10b
        self.cmb_data = cmb_data_dict
        self.v14_engine = ACIM_v14_Cosmology(alpha=self.alpha_v10b)
        self.std_engine = ACIM_v14_Cosmology(alpha=0.0)
        self.baseline_spline = self._create_baseline_spline()
        self.Cl_info_template = self._calculate_Cl_info_template_v14()
        self.optimized_beta = 0.0
        self.baseline_chi2 = np.inf
        self.v15_chi2 = np.inf

    def _create_baseline_spline(self):
        if 'TT' not in self.cmb_data or len(self.cmb_data['L']) == 0:
            return None
        l_obs = self.cmb_data['L']
        Cl_obs = self.cmb_data['TT']
        mask = l_obs > 1
        if np.sum(mask) < 5:
            return None
        log_l = np.log10(l_obs[mask])
        log_Cl = np.log10(Cl_obs[mask])
        return UnivariateSpline(log_l, log_Cl, s=0.5)

    def _calculate_Cl_info_template_v14(self) -> np.ndarray:
        if self.baseline_spline is None:
            return None
        l_values = self.cmb_data['L']
        l_safe = np.where(l_values < 2, 2.0, l_values.astype(float))
        a_proxy = 1.0 / l_safe
        
        E_v14_vec = np.array([self.v14_engine.get_E(a) for a in a_proxy])
        E_std_vec = np.array([self.std_engine.get_E(a) for a in a_proxy])
        E_std_vec[E_std_vec == 0] = 1.0
        
        deviation = (E_v14_vec / E_std_vec) - 1.0
        Cl_std_at_l = np.zeros_like(l_values, dtype=float)
        mask = l_values > 1
        if np.sum(mask) > 0:
            Cl_std_at_l[mask] = 10**self.baseline_spline(np.log10(l_values[mask]))
        
        Cl_info = deviation * Cl_std_at_l
        Cl_info[~np.isfinite(Cl_info)] = 0.0
        return Cl_info
