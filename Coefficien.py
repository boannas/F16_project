import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# **Step 1: Data Input**
# CX Data (5x12: EL vs Alpha)
CX_updated_data_full = np.array([
    [-0.099, -0.081, -0.081, -0.063, -0.025, 0.044, 0.097, 0.113, 0.145, 0.167, 0.174, 0.166],
    [-0.048, -0.038, -0.040, -0.021, 0.016, 0.083, 0.127, 0.137, 0.162, 0.177, 0.179, 0.167],
    [-0.022, -0.020, -0.021, -0.004, 0.032, 0.094, 0.128, 0.130, 0.154, 0.161, 0.155, 0.138],
    [-0.040, -0.038, -0.039, -0.025, 0.006, 0.062, 0.087, 0.085, 0.100, 0.110, 0.104, 0.091],
    [-0.083, -0.073, -0.076, -0.072, -0.046, 0.012, 0.024, 0.025, 0.043, 0.053, 0.047, 0.040]
])

# CZ Data (1x12: Alpha)
CZ_updated_data_full = np.array([
     0.770, 0.241, -0.100, -0.416, -0.731, -1.053,
    -1.366, -1.646, -1.917, -2.120, -2.248, -2.229
])

# Updated CM (Pitching Moment Coefficient) data (5x12)
CM_updated_data_full = np.array([
    [0.205, 0.168, 0.186, 0.196, 0.213, 0.251, 0.245, 0.238, 0.252, 0.231, 0.198, 0.192],
    [0.081, 0.077, 0.107, 0.110, 0.110, 0.141, 0.127, 0.119, 0.133, 0.108, 0.081, 0.093],
    [-0.046, -0.020, -0.009, -0.005, -0.006, 0.010, 0.006, -0.001, 0.014, 0.000, -0.013, 0.032],
    [-0.174, -0.145, -0.121, -0.127, -0.129, -0.102, -0.097, -0.113, -0.087, -0.084, -0.069, -0.006],
    [-0.259, -0.202, -0.184, -0.193, -0.199, -0.150, -0.160, -0.167, -0.104, -0.076, -0.041, -0.005]
])

# Updated CL (Rolling Moment Coefficient) data (6x12)
CL_updated_data_full = np.array([
    [-0.001, -0.004, -0.008, -0.012, -0.016, -0.019, -0.020, -0.020, -0.015, -0.008, -0.013, -0.015],
    [-0.003, -0.009, -0.017, -0.024, -0.030, -0.034, -0.040, -0.037, -0.016, -0.002, -0.010, -0.019],
    [-0.001, -0.010, -0.020, -0.030, -0.039, -0.044, -0.050, -0.049, -0.023, -0.006, -0.014, -0.027],
    [0.000, -0.010, -0.022, -0.034, -0.047, -0.046, -0.059, -0.061, -0.033, -0.036, -0.035, -0.035],
    [0.007, -0.010, -0.023, -0.034, -0.049, -0.046, -0.068, -0.071, -0.060, -0.058, -0.062, -0.059],
    [0.009, -0.011, -0.023, -0.037, -0.050, -0.047, -0.074, -0.079, -0.091, -0.076, -0.077, -0.076]
])

# Updated CN (Yawing Moment Coefficient) data (6x12)
CN_updated_data_full = np.array([
    [0.018, 0.019, 0.018, 0.019, 0.019, 0.018, 0.013, 0.007, 0.004, -0.014, -0.017, -0.033],
    [0.038, 0.042, 0.042, 0.042, 0.043, 0.039, 0.030, 0.017, 0.004, -0.035, -0.047, -0.057],
    [0.056, 0.057, 0.059, 0.058, 0.058, 0.053, 0.032, 0.012, 0.002, -0.046, -0.071, -0.073],
    [0.064, 0.077, 0.076, 0.074, 0.073, 0.057, 0.029, 0.007, 0.012, -0.034, -0.065, -0.041],
    [0.074, 0.086, 0.093, 0.089, 0.080, 0.062, 0.049, 0.022, 0.028, -0.012, -0.002, -0.013],
    [0.079, 0.090, 0.106, 0.106, 0.096, 0.080, 0.068, 0.030, 0.064, 0.015, 0.011, -0.001]
])

# **Step 2: Data Preparation**

# Labels for CX (EL vs Alpha)
alpha_scaled_cx = np.linspace(-10, 45, CX_updated_data_full.shape[1])  # Alpha from -10 to 45
el_scaled_cx = np.linspace(-25, 25, CX_updated_data_full.shape[0])    # EL from -25 to 25

alpha_scaled_cz = np.linspace(-10, 45, len(CZ_updated_data_full))

# Labels for CM (EL vs Alpha)
cm_el_labels = np.linspace(-25, 25, CM_updated_data_full.shape[0])    # EL from -25 to 25
cm_alpha_labels = np.linspace(-10, 45, CM_updated_data_full.shape[1])  # Alpha from -10 to 45

# Labels for CL and CN (Beta vs Alpha)
cl_cn_beta_labels = np.linspace(-30, 30, CL_updated_data_full.shape[0])  # Beta from -30 to 30
cl_cn_alpha_labels = np.linspace(-10, 45, CL_updated_data_full.shape[1])  # Alpha from -10 to 45

# **Step 3: Equation Derivation**

# Function definitions for fitting
def compute_cx_fixed(alpha, el, data=CX_updated_data_full):
    # Map Alpha and EL into indices and interpolate
    scaled_alpha = 0.2 * alpha
    k = int(np.floor(scaled_alpha))
    k = max(-2, min(k, 8))  # Constrain within bounds
    da = scaled_alpha - k
    l = k + int(np.sign(1.1 * da))
    l = max(-2, min(l, 8))  # Constrain within bounds

    k_index = k + 2  # Adjust index for array (0-based)
    l_index = l + 2  # Adjust index for array (0-based)

    scaled_el = el / 12.5
    m = int(np.floor(scaled_el))
    m = max(-2, min(m, 2))  # Constrain within bounds
    de = scaled_el - m
    n = m + int(np.sign(1.1 * de))
    n = max(-2, min(n, 2))  # Constrain within bounds

    m_index = m + 2  # Adjust index for array (0-based)
    n_index = n + 2  # Adjust index for array (0-based)

    # Ensure indices are within valid ranges

    # Interpolate values
    t = data[m_index, k_index]
    u = data[n_index, k_index]
    v = t + abs(da) * (data[m_index, l_index] - t)
    w = u + abs(da) * (data[n_index, l_index] - u)

    return v + (w - v) * abs(de)

def compute_cz(alpha, beta, el, data=CZ_updated_data_full):
    # Map Alpha into indices and interpolate
    scaled_alpha = 0.2 * alpha
    k = int(np.floor(scaled_alpha))
    k = max(-2, min(k, 8))  # Constrain within bounds
    da = scaled_alpha - k
    l = k + int(np.sign(1.1 * da))
    l = max(-2, min(l, 8))  # Constrain within bounds
    
    k_index = k + 2  # Adjust index for array (0-based)
    l_index = l + 2  # Adjust index for array (0-based)
    
    s = data[k_index] + abs(da) * (data[l_index] - data[k_index])
    return s * (1 - (beta / 57.3) ** 2) - 0.19 * (el / 25.0)

def compute_cm_fixed(alpha, el, data=CM_updated_data_full):
    # Map Alpha and EL into indices and interpolate
    scaled_alpha = 0.2 * alpha
    k = int(np.floor(scaled_alpha))
    k = max(-2, min(k, 8))  # Constrain within bounds
    da = scaled_alpha - k
    l = k + int(np.sign(1.1 * da))
    l = max(-2, min(l, 8))  # Constrain within bounds

    k_index = k + 2  # Adjust index for array (0-based)
    l_index = l + 2  # Adjust index for array (0-based)

    scaled_el = el / 12.5
    m = int(np.floor(scaled_el))
    m = max(-2, min(m, 2))  # Constrain within bounds
    de = scaled_el - m
    n = m + int(np.sign(1.1 * de))
    n = max(-2, min(n, 2))  # Constrain within bounds

    m_index = m + 2  # Adjust index for array (0-based)
    n_index = n + 2  # Adjust index for array (0-based)

    # Ensure indices are within valid ranges

    # Interpolate values
    t = data[m_index, k_index]
    u = data[n_index, k_index]
    v = t + abs(da) * (data[m_index, l_index] - t)
    w = u + abs(da) * (data[n_index, l_index] - u)

    return v + (w - v) * abs(de)

def compute_cl_fixed(alpha, beta, data=CL_updated_data_full):
    # Map Alpha and EL into indices and interpolate
    scaled_alpha = 0.2 * alpha
    k = int(np.floor(scaled_alpha))
    k = max(-2, min(k, 8))  # Constrain within bounds
    da = scaled_alpha - k
    l = k + int(np.sign(1.1 * da))
    l = max(-2, min(l, 8))  # Constrain within bounds

    k_index = k + 2  # Adjust index for array (0-based)
    l_index = l + 2  # Adjust index for array (0-based)

    scaled_beta = 0.2 * abs(beta)
    m = int(np.floor(scaled_beta))
    m = max(0, min(m, 5))  # Constrain within bounds
    db = scaled_beta - m
    n = m + int(np.sign(1.1 * db))
    n = max(0, min(n, 5))  # Constrain within bounds

    m_index = m  # Adjust index for array (0-based)
    n_index = n # Adjust index for array (0-based)

    # Ensure indices are within valid ranges

    # Interpolate values
    t = data[m_index, k_index]
    u = data[n_index, k_index]
    v = t + abs(da) * (data[m_index, l_index] - t)
    w = u + abs(da) * (data[n_index, l_index] - u)
    dum = v + (w - v) * abs(db)
    
    return dum 

def compute_cn_fixed(alpha, beta, data=CN_updated_data_full):
    # Map Alpha and EL into indices and interpolate
    scaled_alpha = 0.2 * alpha
    k = int(np.floor(scaled_alpha))
    k = max(-2, min(k, 8))  # Constrain within bounds
    da = scaled_alpha - k
    l = k + int(np.sign(1.1 * da))
    l = max(-2, min(l, 8))  # Constrain within bounds

    k_index = k + 2  # Adjust index for array (0-based)
    l_index = l + 2  # Adjust index for array (0-based)

    scaled_beta = 0.2 * abs(beta)
    m = int(np.floor(scaled_beta))
    m = max(0, min(m, 5))  # Constrain within bounds
    db = scaled_beta - m
    n = m + int(np.sign(1.1 * db))
    n = max(0, min(n, 5))  # Constrain within bounds

    m_index = m  # Adjust index for array (0-based)
    n_index = n # Adjust index for array (0-based)

    # Ensure indices are within valid ranges

    # Interpolate values
    t = data[m_index, k_index]
    u = data[n_index, k_index]
    v = t + abs(da) * (data[m_index, l_index] - t)
    w = u + abs(da) * (data[n_index, l_index] - u)
    dum = v + (w - v) * abs(db)
    
    return dum 

def bilinear_func(xy, a, b, c):
    x, y = xy
    return a * x + b * y + c

# Preparing data for CX equation (Alpha vs EL)
cx_alpha_mesh, cx_el_mesh = np.meshgrid(alpha_scaled_cx, el_scaled_cx)
cx_data = CX_updated_data_full.ravel()
cx_fit_params, _ = curve_fit(
    bilinear_func,
    (cx_alpha_mesh.ravel(), cx_el_mesh.ravel()),
    cx_data
)


# Prepare data for CM equation (Alpha vs EL)
cm_alpha_mesh, cm_el_mesh = np.meshgrid(cm_alpha_labels, cm_el_labels)
cm_data = CM_updated_data_full.ravel()
cm_fit_params, _ = curve_fit(
    bilinear_func,
    (cm_alpha_mesh.ravel(), cm_el_mesh.ravel()),
    cm_data
)

# Prepare data for CL equation (Alpha vs Beta)
cl_alpha_mesh, cl_beta_mesh = np.meshgrid(cl_cn_alpha_labels, cl_cn_beta_labels)
cl_data = CL_updated_data_full.ravel()
cl_fit_params, _ = curve_fit(
    bilinear_func,
    (cl_alpha_mesh.ravel(), cl_beta_mesh.ravel()),
    cl_data
)

# Prepare data for CN equation (Alpha vs Beta)
cn_alpha_mesh, cn_beta_mesh = np.meshgrid(cl_cn_alpha_labels, cl_cn_beta_labels)
cn_data = CN_updated_data_full.ravel()
cn_fit_params, _ = curve_fit(
    bilinear_func,
    (cn_alpha_mesh.ravel(), cn_beta_mesh.ravel()),
    cn_data
)

# Display the resulting equations
cx_eq = f"C_X = {cx_fit_params[0]:.5f} * Alpha + {cx_fit_params[1]:.5f} * EL + {cx_fit_params[2]:.5f}"
cy_eq = f"C_Y = {-0.02:.5f} * Beta + {0.021:.5f} * (AIL/20.0) + {0.086:.5f} * (RDR/30.0)" #From Aircraft_control_and_simulation(nasa)
cz_eq = f"C_Z = s * (1 - (beta / 57.3) ** 2) - 0.19 * (el / 25.0)"
cm_eq = f"C_M = {cm_fit_params[0]:.5f} * Alpha + {cm_fit_params[1]:.5f} * EL + {cm_fit_params[2]:.5f}"
cl_eq = f"C_L = {cl_fit_params[0]:.5f} * Alpha + {cl_fit_params[1]:.5f} * Beta + {cl_fit_params[2]:.5f}"
cn_eq = f"C_N = {cn_fit_params[0]:.5f} * Alpha + {cn_fit_params[1]:.5f} * Beta + {cn_fit_params[2]:.5f}"

print("Derived Equations:")
print(cx_eq)
print(cy_eq)
print(cz_eq)
print(cm_eq)
print(cl_eq)
print(cn_eq)

# **Step 4: Plotting**

# Generating data for plotting
alpha_range = np.linspace(-10, 45, 100)
el_range = np.linspace(-25, 25, 100)
beta_range = np.linspace(-30, 30, 100)

# For C_M: varying Alpha and EL
alpha_mesh_cm, el_mesh_cm = np.meshgrid(alpha_range, el_range)
cm_values = cm_fit_params[0] * alpha_mesh_cm + cm_fit_params[1] * el_mesh_cm + cm_fit_params[2]

# For C_L: varying Alpha and Beta
alpha_mesh_cl, beta_mesh_cl = np.meshgrid(alpha_range, beta_range)
cl_values = cl_fit_params[0] * alpha_mesh_cl + cl_fit_params[1] * beta_mesh_cl + cl_fit_params[2]

# For C_N: varying Alpha and Beta
cn_values = cn_fit_params[0] * alpha_mesh_cl + cn_fit_params[1] * beta_mesh_cl + cn_fit_params[2]

# # Plotting C_M
# plt.figure(figsize=(8, 6))
# plt.contourf(alpha_mesh_cm, el_mesh_cm, cm_values, levels=50, cmap='viridis')
# plt.colorbar(label='$C_M$')
# plt.title("$C_M$ as a function of Alpha and EL")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("Elevator Deflection (degrees)")
# plt.grid(True)
# plt.show()

# # Plotting C_L
# plt.figure(figsize=(8, 6))
# plt.contourf(alpha_mesh_cl, beta_mesh_cl, cl_values, levels=50, cmap='viridis')
# plt.colorbar(label='$C_L$')
# plt.title("$C_L$ as a function of Alpha and Beta")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("Beta (degrees)")
# plt.grid(True)
# plt.show()

# # Plotting C_N
# plt.figure(figsize=(8, 6))
# plt.contourf(alpha_mesh_cl, beta_mesh_cl, cn_values, levels=50, cmap='viridis')
# plt.colorbar(label='$C_N$')
# plt.title("$C_N$ as a function of Alpha and Beta")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("Beta (degrees)")
# plt.grid(True)
# plt.show()

# Generating specific Alpha, EL, and Beta values for comparison
alpha_check = np.linspace(-10, 45, 12)  # Alpha: -10, -5, ..., 45
el_check = np.linspace(-25, 25, 5)      # EL: -25, -12.5, ..., 25
beta_check = np.linspace(-30, 30, 6)    # Beta: -30, -20, ..., 30

# Create meshes for the checks
alpha_mesh_cm, el_mesh_cm = np.meshgrid(alpha_check, el_check)  # CM and CX
alpha_mesh_cl, beta_mesh_cl = np.meshgrid(alpha_check, beta_check)  # CL and CN

# Calculate coefficients using derived equations
# CM and CX
cm_values_check = cm_fit_params[0] * alpha_mesh_cm + cm_fit_params[1] * el_mesh_cm + cm_fit_params[2]
cx_values_check = cx_fit_params[0] * alpha_mesh_cm + cx_fit_params[1] * el_mesh_cm + cx_fit_params[2]

# CL and CN
cl_values_check = cl_fit_params[0] * alpha_mesh_cl + cl_fit_params[1] * beta_mesh_cl + cl_fit_params[2]
cn_values_check = cn_fit_params[0] * alpha_mesh_cl + cn_fit_params[1] * beta_mesh_cl + cn_fit_params[2]

# Extract original data for comparison
original_cm_values = CM_updated_data_full
original_cx_values = CX_updated_data_full
original_cl_values = CL_updated_data_full
original_cn_values = CN_updated_data_full

# Plot comparison for CM
# plt.figure(figsize=(10, 6))
# for i, el in enumerate(el_check):
#     plt.plot(alpha_check, cm_values_check[i, :], 'o-', label=f"Derived CM (EL={el:.1f})")
#     plt.plot(alpha_check, original_cm_values[i, :], 'x--', label=f"Original CM (EL={el:.1f})")
# plt.title("Comparison of Derived and Original $C_M$ Values")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("$C_M$")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot comparison for CX
# plt.figure(figsize=(10, 6))
# for i, el in enumerate(el_check):
#     plt.plot(alpha_check, cx_values_check[i, :], 'o-', label=f"Derived CX (EL={el:.1f})")
#     plt.plot(alpha_check, original_cx_values[i, :], 'x--', label=f"Original CX (EL={el:.1f})")
# plt.title("Comparison of Derived and Original $C_X$ Values")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("$C_X$")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot comparison for CL
# plt.figure(figsize=(10, 6))
# for i, beta in enumerate(beta_check):
#     plt.plot(alpha_check, cl_values_check[i, :], 'o-', label=f"Derived CL (Beta={beta:.1f})")
#     plt.plot(alpha_check, original_cl_values[i, :], 'x--', label=f"Original CL (Beta={beta:.1f})")
# plt.title("Comparison of Derived and Original $C_L$ Values")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("$C_L$")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot comparison for CN
# plt.figure(figsize=(10, 6))
# for i, beta in enumerate(beta_check):
#     plt.plot(alpha_check, cn_values_check[i, :], 'o-', label=f"Derived CN (Beta={beta:.1f})")
#     plt.plot(alpha_check, original_cn_values[i, :], 'x--', label=f"Original CN (Beta={beta:.1f})")
# plt.title("Comparison of Derived and Original $C_N$ Values")
# plt.xlabel("Alpha (degrees)")
# plt.ylabel("$C_N$")
# plt.legend()
# plt.grid(True)
# plt.show()

# Calculate CZ values from the derived equation for Alpha range (-10 to 45)
alpha_check = np.linspace(-10, 45, len(CZ_updated_data_full))  # Alpha corresponding to original data
beta_check = 0  # Beta set to 0 for direct comparison
el_check = 0    # EL set to 0 for direct comparison

# Compute CZ using the derived formula
cz_computed = [compute_cz(a, beta_check, el_check) for a in alpha_check]

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(alpha_check, CZ_updated_data_full, 'x--', label="Original CZ Data")
plt.plot(alpha_check, cz_computed, 'o-', label="Derived CZ (Alpha, Beta=0, EL=0)")
plt.title("Comparison of Derived and Original $C_Z$ Values")
plt.xlabel("Alpha (degrees)")
plt.ylabel("$C_Z$")
plt.legend()
plt.grid(True)
plt.show()


alpha_check = np.linspace(-10, 45, CX_updated_data_full.shape[1])  # Alpha corresponding to original data
el_check = np.linspace(-25, 25, CX_updated_data_full.shape[0])    # EL corresponding to original data


alpha_mesh_check, el_mesh_check = np.meshgrid(alpha_check, el_check)

# Compute CX using the derived interpolation function
cx_computed = np.array([
    compute_cx_fixed(a, e) for a, e in zip(alpha_mesh_check.ravel(), el_mesh_check.ravel())
]).reshape(alpha_mesh_check.shape)

# Plot comparison
plt.figure(figsize=(10, 6))
for i, el in enumerate(el_check):
    plt.plot(alpha_check, cx_computed[i, :], 'o-', label=f"Derived CX (EL={el:.1f})")
    plt.plot(alpha_check, CX_updated_data_full[i, :], 'x--', label=f"Original CX (EL={el:.1f})")

plt.title("Comparison of Derived and Original $C_X$ Values")
plt.xlabel("Alpha (degrees)")
plt.ylabel("$C_X$")
plt.legend()
plt.grid(True)
plt.show()

# Compute CX using the derived interpolation function
cm_computed = np.array([
    compute_cm_fixed(a, e) for a, e in zip(alpha_mesh_check.ravel(), el_mesh_check.ravel())
]).reshape(alpha_mesh_check.shape)

# Plot comparison
plt.figure(figsize=(10, 6))
for i, el in enumerate(el_check):
    plt.plot(alpha_check, cm_computed[i, :], 'o-', label=f"Derived CM (EL={el:.1f})")
    plt.plot(alpha_check, CM_updated_data_full[i, :], 'x--', label=f"Original CM (EL={el:.1f})")

plt.title("Comparison of Derived and Original $C_M$ Values")
plt.xlabel("Alpha (degrees)")
plt.ylabel("$C_M$")
plt.legend()
plt.grid(True)
plt.show()


cl_alpha_check = np.linspace(-10, 45, CL_updated_data_full.shape[1])  # Alpha corresponding to original data
cl_beta_check = np.linspace(0, 30, CL_updated_data_full.shape[0])    # EL corresponding to original data


cl_alpha_mesh_check, cl_beta_mesh_check = np.meshgrid(cl_alpha_check, cl_beta_check)

cl_computed = np.array([
    compute_cl_fixed(a, e) for a, e in zip(cl_alpha_mesh_check.ravel(), cl_beta_mesh_check.ravel())
]).reshape(cl_alpha_mesh_check.shape)

# Plot comparison
plt.figure(figsize=(10, 6))
for i, beta in enumerate(cl_beta_check):
    plt.plot(cl_alpha_check, cl_computed[i, :], 'o-', label=f"Derived CL (Beta={beta:.1f})")
    plt.plot(cl_alpha_check, CL_updated_data_full[i, :], 'x--', label=f"Original CL (Beta={beta:.1f})")

plt.title("Comparison of Derived and Original $C_L$ Values")
plt.xlabel("Alpha (degrees)")
plt.ylabel("$C_L$")
plt.legend()
plt.grid(True)
plt.show()

cn_alpha_check = np.linspace(-10, 45, CN_updated_data_full.shape[1])  # Alpha corresponding to original data
cn_beta_check = np.linspace(0, 30, CN_updated_data_full.shape[0])    # EL corresponding to original data


cn_alpha_mesh_check, cn_beta_mesh_check = np.meshgrid(cn_alpha_check, cn_beta_check)

cn_computed = np.array([
    compute_cn_fixed(a, e) for a, e in zip(cn_alpha_mesh_check.ravel(), cn_beta_mesh_check.ravel())
]).reshape(cn_alpha_mesh_check.shape)

# Plot comparison
plt.figure(figsize=(10, 6))
for i, beta in enumerate(cn_beta_check):
    plt.plot(cn_alpha_check, cn_computed[i, :], 'o-', label=f"Derived CN (Beta={beta:.1f})")
    plt.plot(cn_alpha_check, CN_updated_data_full[i, :], 'x--', label=f"Original CN (Beta={beta:.1f})")

plt.title("Comparison of Derived and Original $C_N$ Values")
plt.xlabel("Alpha (degrees)")
plt.ylabel("$C_N$")
plt.legend()
plt.grid(True)
plt.show()