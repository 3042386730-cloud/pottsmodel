import os
import numpy as onp
from .config import PottsConfig

class DataAnalyzer:
    """Handles the analysis of simulation results, specifically Binder cumulants."""
    
    def __init__(self, config: PottsConfig):
        self.config = config
        self.output_base = config.output_base

    def analyze(self):
        print("\n=== Calculating Binder cumulants ===", flush=True)
        self._calculate_standard_binder()
        if self.config.q == 2:
            self._calculate_paper_binder_u2()

    @staticmethod
    def _calculate_binder_with_error(mags, num_bins=32):
        n_samples = len(mags)
        if n_samples < num_bins:
            num_bins = max(1, n_samples // 2)
        
        bin_size = n_samples // num_bins
        binned_data = mags[:num_bins*bin_size].reshape(num_bins, bin_size)
        
        bin_binders = []
        for bin_data in binned_data:
            M2 = onp.mean(bin_data**2)
            M4 = onp.mean(bin_data**4)
            binder = 1 - M4/(3*M2**2) if M2 > 0 else 0.0
            bin_binders.append(binder)
        
        binder_mean = onp.mean(bin_binders)
        binder_std = onp.std(bin_binders, ddof=1) if len(bin_binders) > 1 else 0.0
        binder_error = binder_std / onp.sqrt(num_bins) if num_bins > 1 else binder_std
        return binder_mean, binder_error

    def _calculate_standard_binder(self):
        binder_results = []
        error_results = []
        
        for i_temp in range(self.config.n_temp):
            temp_val = float(self.config.temperatures[i_temp])
            all_mags = self._load_data(temp_val, "magnetization_run_{}.txt")
            
            if len(all_mags) == 0:
                print(f"  Warning: No data found for temperature {temp_val:.4f}", flush=True)
                continue
                
            all_mags = onp.concatenate(all_mags)
            binder_mean, binder_error = self._calculate_binder_with_error(all_mags)
            
            binder_results.append((temp_val, binder_mean))
            error_results.append((temp_val, binder_error))
            print(f"  Temperature {temp_val:.4f}: Standard Binder U4 = {binder_mean:.6f} ± {binder_error:.6f}", flush=True)
            
        self._save_results("binder_cumulants.txt", "BinderCumulant", binder_results, error_results)

    def _calculate_paper_binder_u2(self):
        print("\n=== Calculating Binder U2 (paper method: U2 = <m^2>/<|m|>^2) ===", flush=True)
        binder_results = []
        error_results = []
        
        for i_temp in range(self.config.n_temp):
            temp_val = float(self.config.temperatures[i_temp])
            all_signed_mags = self._load_data(temp_val, "signed_magnetization_run_{}.txt")
            
            if len(all_signed_mags) == 0:
                print(f"  Warning: No signed mag data found for temperature {temp_val:.4f}", flush=True)
                continue
                
            all_signed_mags = onp.concatenate(all_signed_mags)
            U2_mean, U2_error = self._calculate_u2_with_error(all_signed_mags)
            
            binder_results.append((temp_val, U2_mean))
            error_results.append((temp_val, U2_error))
            print(f"  Temperature {temp_val:.4f}: Paper-style Binder U2 = {U2_mean:.6f} ± {U2_error:.6f}", flush=True)
            
        self._save_results("binder_U2.txt", "BinderU2", binder_results, error_results)

    def _load_data(self, temp_val, filename_pattern):
        data = []
        for j_run in range(self.config.n_run):
            file_path = os.path.join(
                self.output_base, f"temp_{temp_val:.4f}", 
                filename_pattern.format(j_run)
            )
            if os.path.exists(file_path):
                data.append(onp.loadtxt(file_path))
        return data

    @staticmethod
    def _calculate_u2_with_error(signed_mags, num_bins=78):
        n_samples = len(signed_mags)
        if n_samples < num_bins:
            num_bins = max(2, n_samples // 2)
        
        bin_size = n_samples // num_bins
        binned_data = signed_mags[:num_bins*bin_size].reshape(num_bins, bin_size)
        
        U2_vals = []
        for bin_data in binned_data:
            m2 = onp.mean(bin_data**2)
            abs_m = onp.mean(onp.abs(bin_data))
            U2 = m2 / (abs_m**2) if abs_m > 1e-10 else 0.0
            U2_vals.append(U2)
        
        U2_mean = onp.mean(U2_vals)
        U2_std = onp.std(U2_vals, ddof=1) if len(U2_vals) > 1 else 0.0
        U2_error = U2_std / onp.sqrt(num_bins) if num_bins > 1 else U2_std
        return U2_mean, U2_error

    def _save_results(self, filename, metric_name, results, errors):
        path = os.path.join(self.output_base, filename)
        with open(path, 'w') as f:
            f.write(f"Temperature {metric_name} Error\n")
            for (temp, val), (_, err) in zip(results, errors):
                f.write(f"{temp:.6f} {val:.6f} {err:.6f}\n")
        print(f"\n{metric_name} saved to: {path}", flush=True)
