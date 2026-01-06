import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import os

# MODIFY THIS LIST WITH YOUR ACTUAL FILENAMES
file_list = [
    "binder_cumulants_8.txt", "binder_cumulants_16.txt", "binder_cumulants_32.txt", "binder_cumulants_64.txt"
]

# Read data from files
datasets = []
for filename in file_list:
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found. Please check the filename.")
        continue

    try:
        data = np.loadtxt(filename, skiprows=1)
        if data.shape[1] < 3:
            print(f"Error: File {filename} does not have enough columns. Expected 3 columns.")
            continue
        datasets.append(data)
        print(f"Successfully loaded {filename} with {data.shape[0]} data points")
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")

if len(datasets) < 2:
    raise ValueError("At least two valid datasets are required for intersection analysis")

# Determine overlapping temperature range
T_min = max(data[0, 0] for data in datasets)
T_max = min(data[-1, 0] for data in datasets)
T_grid = np.linspace(T_min, T_max, 1000)

# Create figure
plt.figure(figsize=(10, 6))

# Store interpolants for intersection calculation
interpolants = []

# Process each dataset
for i, data in enumerate(datasets):
    T = data[:, 0]
    U = data[:, 1]
    err = data[:, 2]

    # Filter data to overlapping temperature range
    mask = (T >= T_min) & (T <= T_max)
    T = T[mask]
    U = U[mask]
    err = err[mask]

    # Cubic spline interpolation
    cs = CubicSpline(T, U, bc_type='natural')
    interpolants.append(cs)

    # Plot data points and interpolation
    plt.errorbar(T, U, yerr=err, fmt='o', markersize=6,
                 label=f'Dataset {i + 1}', capsize=3, alpha=0.8)
    plt.plot(T_grid, cs(T_grid), '-', linewidth=2, label=f'Fit {i + 1}')

# Calculate intersections between all pairs of curves
intersections = []
colors = ['green', 'purple', 'brown', 'orange', 'cyan']

pair_idx = 0
for i in range(len(interpolants)):
    for j in range(i + 1, len(interpolants)):
        # Define difference function
        diff_func = lambda T: interpolants[i](T) - interpolants[j](T)

        # Find sign changes in dense grid
        diff_values = diff_func(T_grid)
        sign_changes = np.where(np.diff(np.sign(diff_values)))[0]

        # Find roots in sign change intervals
        for idx in sign_changes:
            T_left = T_grid[idx]
            T_right = T_grid[idx + 1]
            try:
                T_cross = brentq(diff_func, T_left, T_right)
                U_cross = interpolants[i](T_cross)
                intersections.append((T_cross, U_cross))

                # Plot intersection point
                plt.plot(T_cross, U_cross, 's', markersize=8,
                         color=colors[pair_idx % len(colors)],
                         label=f'Intersection {i + 1}-{j + 1}' if pair_idx == 0 else "")
                pair_idx += 1
            except ValueError:
                continue
            except Exception as e:
                print(
                    f"Warning: Could not find intersection for pair ({i + 1},{j + 1}) in interval [{T_left:.4f}, {T_right:.4f}]: {str(e)}")

# Calculate and plot average intersection
if intersections:
    avg_T = np.mean([pt[0] for pt in intersections])
    avg_U = np.mean([pt[1] for pt in intersections])
    plt.plot(avg_T, avg_U, '*', markersize=15, color='red', markeredgewidth=1.5,
             label=f'Average Intersection\nTc = {avg_T:.5f}\nU = {avg_U:.5f}')
    print(f"\nAnalysis Results:")
    print(f"Number of intersections found: {len(intersections)}")
    print(f"Average critical temperature: Tc = {avg_T:.5f}")
    print(f"Average Binder cumulant at Tc: U = {avg_U:.5f}")

    # Print individual intersections
    print("\nIndividual intersection points:")
    for k, (T_val, U_val) in enumerate(intersections):
        print(f"  Intersection {k + 1}: T = {T_val:.5f}, U = {U_val:.5f}")
else:
    print("Warning: No intersections found between curves. Cannot determine critical point.")

# Finalize plot
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Binder Cumulant', fontsize=12)
plt.title('Binder Cumulant Curves and Intersections Analysis', fontsize=14)
plt.legend(loc='best', fontsize=9, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show plot
output_filename = 'binder_intersections_analysis.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_filename}")
plt.show()

print("\nAnalysis complete. Check the plot for visual verification of results.")