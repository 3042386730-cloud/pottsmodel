import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import t

# ================== Configuration Parameters ==================
# File list - replace with your actual filenames
data_files = [
    "binder_cumulants_8.txt", "binder_cumulants_16.txt", "binder_cumulants_32.txt"
]

# Extract system sizes from filenames
system_sizes = []
for filename in data_files:
    size_str = os.path.splitext(os.path.basename(filename))[0]
    try:
        # Extract size from filename (e.g., "8.txt" -> 8)
        size = int(''.join(filter(str.isdigit, size_str)))
        system_sizes.append(size)
    except:
        # Default estimation if parsing fails
        size = 8 * (2 ** len(system_sizes))
        system_sizes.append(size)

# Output filenames
output_png = "binder_old_analysis.png"
output_pdf = "binder_old_analysis.pdf"
fit_results_file = "fitting_old_results.txt"

# Plot parameters
fig_size = (12, 8)  # Larger figure size to accommodate more information
dpi = 300
line_width = 2.5
marker_size = 10
error_width = 1.5
cap_size = 4
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# ================== Load Data ==================
all_data = []
for i, filename in enumerate(data_files):
    if not os.path.exists(filename):
        print(f"Warning: File {filename} does not exist, skipping.")
        continue

    try:
        # Skip possible header lines, auto-detect
        with open(filename, 'r') as f:
            first_line = f.readline()
            if 'Temperature' in first_line or 'Binder' in first_line:
                skip_rows = 1
            else:
                skip_rows = 0

        data = np.loadtxt(filename, skiprows=skip_rows)
        temps = data[:, 0]
        binders = data[:, 1]
        errors = data[:, 2] if data.shape[1] > 2 else np.zeros_like(binders)

        all_data.append({
            'size': system_sizes[i],
            'temps': temps,
            'binders': binders,
            'errors': errors,
            'filename': filename
        })
        print(f"Loaded {filename} (L={system_sizes[i]}): {len(temps)} data points")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

if len(all_data) < 2:
    print("Error: At least two datasets with different system sizes are required for crossing point analysis.")
    exit()

# ================== Find Crossing Points ==================
print("\n=== Calculating crossing points of Binder cumulant curves ===")
crossing_points = []

# Sort by system size
all_data.sort(key=lambda x: x['size'])

for i in range(len(all_data) - 1):
    data1 = all_data[i]
    data2 = all_data[i + 1]

    # Create cubic spline interpolation functions
    interp1 = interp1d(data1['temps'], data1['binders'], kind='cubic',
                       bounds_error=False, fill_value="extrapolate")
    interp2 = interp1d(data2['temps'], data2['binders'], kind='cubic',
                       bounds_error=False, fill_value="extrapolate")

    # Find temperature overlap region
    temp_min = max(data1['temps'].min(), data2['temps'].min())
    temp_max = min(data1['temps'].max(), data2['temps'].max())

    if temp_min >= temp_max:
        print(f"Warning: System sizes {data1['size']} and {data2['size']} have no temperature overlap region")
        continue


    # Define difference function and solve
    def diff_func(t):
        return interp1(t) - interp2(t)


    # Use midpoint of overlap region as initial guess
    temp_guess = (temp_min + temp_max) / 2.0
    try:
        crossing_temp = fsolve(diff_func, temp_guess)[0]

        # Check if crossing point is within overlap region
        if temp_min <= crossing_temp <= temp_max:
            # Estimate crossing point error (simplified: error propagation)
            error1 = np.interp(crossing_temp, data1['temps'], data1['errors'])
            error2 = np.interp(crossing_temp, data2['temps'], data2['errors'])
            crossing_error = np.sqrt(error1 ** 2 + error2 ** 2) / 2.0  # Average error

            crossing_points.append({
                'L': data2['size'],  # Larger size
                'L_small': data1['size'],
                'T_cross': crossing_temp,
                'U_cross': interp1(crossing_temp),
                'error': crossing_error
            })
            print(
                f"System sizes {data1['size']} and {data2['size']} cross at T = {crossing_temp:.6f} ± {crossing_error:.6f}")
        else:
            print(
                f"Warning: Crossing point {crossing_temp:.6f} for system sizes {data1['size']} and {data2['size']} is outside the temperature overlap range [{temp_min:.4f}, {temp_max:.4f}]")
    except Exception as e:
        print(f"Error calculating crossing point for system sizes {data1['size']} and {data2['size']}: {e}")

if not crossing_points:
    print("Error: No valid crossing points found.")
    exit()

# ================== Finite-Size Scaling Fit ==================
print("\n=== Performing finite-size scaling fit ===")
# Prepare fit data
L_values = np.array([cp['L'] for cp in crossing_points])
T_cross = np.array([cp['T_cross'] for cp in crossing_points])
errors = np.array([cp['error'] for cp in crossing_points])

# Paper formula: T*(L) = T* + A·L^(-1/ν), with ν=1 for 2D
# i.e.: T*(L) = T* + A·L^(-1)
x_values = 1.0 / L_values  # 1/L

# Weighted least squares fit (weights = 1/error^2)
weights = 1.0 / (errors ** 2 + 1e-10)  # Avoid division by zero
weights = weights / np.sum(weights)  # Normalize weights

# Linear fit: y = a + b*x
coeffs = np.polyfit(x_values, T_cross, 1, w=np.sqrt(weights))
A = coeffs[0]  # Slope
Tc = coeffs[1]  # Intercept, the critical temperature

# Calculate fit standard errors
T_fit = np.polyval(coeffs, x_values)
residuals = T_cross - T_fit
chi2 = np.sum((residuals ** 2) * weights)
dof = len(x_values) - 2  # Degrees of freedom
if dof > 0:
    chi2_reduced = chi2 / dof
    # Standard error of critical temperature
    x_mean = np.mean(x_values)
    x_var = np.sum(weights * (x_values - x_mean) ** 2)
    Tc_error = np.sqrt(1.0 / (np.sum(weights) * x_var)) * np.sqrt(chi2_reduced)
    # Standard error of slope
    A_error = np.sqrt(np.sum(weights) / x_var) * np.sqrt(chi2_reduced)
else:
    chi2_reduced = 0
    Tc_error = 0.0
    A_error = 0.0

# Calculate fit curve for plotting
x_fit = np.linspace(0, max(x_values) * 1.1, 100)
T_fit_curve = A * x_fit + Tc

print(f"\nFit Results:")
print(f"Critical temperature Tc = {Tc:.6f} ± {Tc_error:.6f}")
print(f"Slope coefficient A = {A:.6f} ± {A_error:.6f}")
print(f"Chi-squared/DOF = {chi2:.3f}/{dof} = {chi2_reduced:.3f}")

# ================== Plotting ==================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, height_ratios=[2, 1])

# -- Figure 1: Binder cumulant curves --
for i, data in enumerate(all_data):
    color = colors[i % len(colors)]
    label = f'L = {data["size"]}'

    # Plot original data points and error bars
    ax1.errorbar(data['temps'], data['binders'], yerr=data['errors'],
                 fmt='o', color=color, alpha=0.6, capsize=cap_size,
                 markersize=marker_size / 2, label='_nolegend_')

    # Plot interpolation curve
    temp_fine = np.linspace(data['temps'].min(), data['temps'].max(), 200)
    binder_interp = interp1d(data['temps'], data['binders'], kind='cubic',
                             bounds_error=False, fill_value="extrapolate")(temp_fine)
    ax1.plot(temp_fine, binder_interp, '-', linewidth=line_width, color=color, label=label)

# Mark crossing points
for cp in crossing_points:
    ax1.plot(cp['T_cross'], cp['U_cross'], 'k*', markersize=12,
             label=f'L={cp["L_small"]}&{cp["L"]}\nT={cp["T_cross"]:.5f}±{cp["error"]:.5f}')

# Add fitted critical temperature line
ax1.axvline(x=Tc, color='k', linestyle='--', alpha=0.7, linewidth=2,
            label=f'Fitted Tc = {Tc:.5f}±{Tc_error:.5f}')

# Figure 1 settings
ax1.set_xlabel('Temperature $T/J$', fontsize=14)
ax1.set_ylabel('Binder Cumulant $U_4$', fontsize=14)
ax1.set_title('Binder Cumulant Analysis for 2D Ising Model', fontsize=16)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(title='System Size', fontsize=10, title_fontsize=11, loc='upper right')

# Set x-axis range, focus on crossing region
all_temps = np.concatenate([data['temps'] for data in all_data])
temp_range = all_temps.max() - all_temps.min()
temp_buffer = 0.1 * temp_range
ax1.set_xlim(all_temps.min() - temp_buffer, all_temps.max() + temp_buffer)

# -- Figure 2: Finite-size scaling analysis --
# Plot data points
ax2.errorbar(x_values, T_cross, yerr=errors, fmt='o', color='blue',
             capsize=cap_size, markersize=marker_size, alpha=0.8,
             label='Crossing point data')

# Plot fit curve
ax2.plot(x_fit, T_fit_curve, 'r-', linewidth=line_width + 1,
         label=f'Fit: $T^*(L) = T_c + A \\cdot L^{{-1}}$\n$T_c = {Tc:.5f} \\pm {Tc_error:.5f}$')

# Figure 2 settings
ax2.set_xlabel('$1/L$', fontsize=14)
ax2.set_ylabel('Crossing temperature $T^*(L)$', fontsize=14)
ax2.set_title('Finite-Size Scaling of Crossing Points', fontsize=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)

# Add fit equation text box
fit_eq = (f'Fit equation: $T^*(L) = T_c + A \\cdot L^{{-1}}$\n'
          f'$T_c = {Tc:.5f} \\pm {Tc_error:.5f}$\n'
          f'$A = {A:.5f} \\pm {A_error:.5f}$\n'
          f'$\\chi^2/dof = {chi2:.2f}/{dof} = {chi2_reduced:.2f}$')
ax2.text(0.98, 0.05, fit_eq, transform=ax2.transAxes,
         ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8),
         fontsize=11)

# Adjust layout
plt.tight_layout()

# ================== Save and Display ==================
plt.savefig(output_png, dpi=dpi, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"\nPlots saved to:")
print(f"  PNG: {os.path.abspath(output_png)}")
print(f"  PDF: {os.path.abspath(output_pdf)}")

plt.show()

# ================== Save Fit Results ==================
with open(fit_results_file, 'w') as f:
    f.write("=== Binder Cumulant Crossing Analysis Results ===\n\n")
    f.write("Reference: Dolfi et al., arXiv:1401.2000v1 (2014)\n")
    f.write("Method: Finite-size scaling analysis, T*(L) = T* + A·L^(-1/ν), with ν=1\n")
    f.write("      Weighted least squares fit with weights = 1/σ²\n\n")

    f.write("Crossing point data:\n")
    f.write("L_small\tL_large\tT_cross\tError\n")
    for cp in crossing_points:
        f.write(f"{cp['L_small']}\t{cp['L']}\t{cp['T_cross']:.6f}\t{cp['error']:.6f}\n")

    f.write("\nFitting results:\n")
    f.write(f"Critical temperature Tc/J = {Tc:.6f} ± {Tc_error:.6f}\n")
    f.write(f"Slope coefficient A = {A:.6f} ± {A_error:.6f}\n")
    f.write(f"Chi-squared/DOF = {chi2:.3f}/{dof} = {chi2_reduced:.3f}\n")
    f.write(f"Exact value (Onsager): Tc/J = 2.269185\n")
    f.write(f"Relative error = {(Tc - 2.269185) / 2.269185 * 100:.4f}%\n")

print(f"\nFitting results saved to: {fit_results_file}")
print(f"Critical temperature Tc/J = {Tc:.6f} ± {Tc_error:.6f}")

# ================== Data Summary ==================
print("\n===== Data Summary =====")
for data in all_data:
    print(f"\nFile: {data['filename']} (L={data['size']})")
    print(f"  Temperature range: {data['temps'].min():.4f} to {data['temps'].max():.4f}")
    print(f"  Binder range: {data['binders'].min():.6f} to {data['binders'].max():.6f}")
    print(f"  Number of data points: {len(data['temps'])}")