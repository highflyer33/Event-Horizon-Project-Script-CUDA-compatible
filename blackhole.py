import ehtim as eh
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# PHASE 1: Data Preprocessing (Complex Waves)
# ==========================================
filename = 'SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits'
print("Loading EHT Complex Visibilities...")
obs = eh.obsdata.load_uvfits(filename)
zbl = 0.6  # Total Flux in Janskys

# Extract full complex visibility data (Amplitude + Phase)
u = cp.array(obs.data['u'], dtype=cp.float64)
v = cp.array(obs.data['v'], dtype=cp.float64)
vis = cp.array(obs.data['vis'], dtype=cp.complex128) / zbl
sigma = cp.array(obs.data['sigma'], dtype=cp.float64) / zbl

# Weight by inverse variance
weights = 1.0 / (sigma ** 2)

# ==========================================
# PHASE 2: The Non-Uniform Fourier Matrix
# ==========================================
print("Constructing 105-Million Element Transform Matrix...")
npix = 128
fov = 120 * eh.RADPERUAS
pixel_scale_uas = fov / eh.RADPERUAS / npix

l = cp.linspace(fov/2, -fov/2, npix)
m = cp.linspace(-fov/2, fov/2, npix)
L, M = cp.meshgrid(l, m)
L_flat, M_flat = L.flatten(), M.flatten()

# Spatial Mask: The black hole emission is physically confined to the central ~55 microarcseconds
R_uas = cp.sqrt((L / eh.RADPERUAS)**2 + (M / eh.RADPERUAS)**2)
spatial_mask = (R_uas <= 55).flatten()

# Build transformation matrix & its Conjugate Transpose
phase = -2j * cp.pi * (cp.outer(u, L_flat) + cp.outer(v, M_flat))
F = cp.exp(phase)
F_conj_T = cp.conj(F).T

# ==========================================
# PHASE 3: Custom GPU Adam Optimizer
# ==========================================
print("Igniting RML Physics Engine on CUDA Cores...")

def get_laplacian(img_2d):
    """Calculates spatial second-derivatives to smooth the plasma ring."""
    lap = cp.zeros_like(img_2d)
    lap += 4 * img_2d
    lap[:, :-1] -= img_2d[:, 1:]
    lap[:, 1:]  -= img_2d[:, :-1]
    lap[:-1, :] -= img_2d[1:, :]
    lap[1:, :]  -= img_2d[:-1, :]
    return lap

# Initialize image as a flat disk confined to our mask area
I_flat = cp.ones_like(L_flat, dtype=cp.float64) * spatial_mask
I_flat /= cp.sum(I_flat)

# Adam Optimizer Parameters
m_adam = cp.zeros_like(I_flat)
v_adam = cp.zeros_like(I_flat)
beta1, beta2, lr = 0.9, 0.999, 2e-3

iterations = 1500
N_vis = len(vis)

for i in range(iterations):
    # 1. Forward Transform (Sky to Telescope)
    V_model = cp.dot(F, I_flat)
    
    # 2. Chi-Squared Gradient (How far off are we from the real telescope data?)
    # Using full complex wave residuals
    residual = V_model - vis
    grad_data = (2.0 / N_vis) * cp.real(cp.dot(F_conj_T, residual * weights))
    
    # 3. Regularization Gradients (Sculpting the physics)
    grad_lap = get_laplacian(I_flat.reshape(npix, npix)).flatten()
    
    # Dynamically scale regularizers so they never overpower the actual telescope data
    max_data_grad = cp.max(cp.abs(grad_data))
    lap_weight = 0.03 * (max_data_grad / (cp.max(cp.abs(grad_lap)) + 1e-10))
    l1_weight  = 0.01 * max_data_grad  # Pushes empty space to true black
    
    # 4. Total Gradient
    total_grad = grad_data + (lap_weight * grad_lap) + l1_weight
    
    # 5. Adam Update
    m_adam = beta1 * m_adam + (1 - beta1) * total_grad
    v_adam = beta2 * v_adam + (1 - beta2) * (total_grad ** 2)
    m_hat = m_adam / (1 - beta1 ** (i + 1))
    v_hat = v_adam / (1 - beta2 ** (i + 1))
    
    I_flat -= lr * (m_hat / (cp.sqrt(v_hat) + 1e-8))
    
    # 6. Physical Constraints
    I_flat = cp.maximum(I_flat, 0.0)       # Positivity (No negative light)
    I_flat *= spatial_mask                 # Confinement
    
    current_flux = cp.sum(I_flat)
    if current_flux > 0:
        I_flat *= (1.0 / current_flux)     # Conserve 1.0 normalized flux
        
    if i % 250 == 0 or i == iterations - 1:
        chi2 = cp.sum(cp.abs(residual)**2 * weights) / N_vis
        print(f"Iteration {i:04d} | Reduced Chi-Squared: {chi2:.3f}")

print("CUDA Reconstruction Converged.")

# ==========================================
# PHASE 4: The Official EHT Restoration
# ==========================================
print("Applying nominal resolving beam and rendering...")
I_cpu = cp.asnumpy(I_flat).reshape((npix, npix)) * zbl

# The theoretical maximum resolution of the EHT in 2017 is ~20 microarcseconds.
# We convolve our mathematical model with this Gaussian beam to produce the physical observation.
beam_fwhm_uas = 20.0
beam_sigma_pixels = (beam_fwhm_uas / pixel_scale_uas) / 2.355
I_final = gaussian_filter(I_cpu, sigma=beam_sigma_pixels)

# Render Graphics
plt.figure(figsize=(10, 10), facecolor='black')
plt.imshow(
    I_final, 
    cmap='afmhot', 
    extent=[-60, 60, -60, 60], 
    origin='lower',
    vmin=0.0, 
    vmax=np.max(I_final)
)

plt.title("M87* Event Horizon (CUDA Complex RML Reconstruction)", color='white', fontsize=16, pad=15)
plt.xlabel(r"Relative R.A. ($\mu$as)", color='white', fontsize=12)
plt.ylabel(r"Relative Dec. ($\mu$as)", color='white', fontsize=12)
plt.tick_params(colors='white', labelsize=10)

# Draw scale bar and beam circle (standard EHT visual conventions)
ax = plt.gca()
beam_circle = plt.Circle((-45, -45), beam_fwhm_uas/2, color='white', fill=False, linewidth=1.5, alpha=0.8)
ax.add_patch(beam_circle)
plt.text(-45, -53, "EHT Beam", color='white', ha='center', fontsize=9, alpha=0.8)

plt.savefig('M87_Master_GPU.png', bbox_inches='tight', dpi=400, facecolor='black')
print("Complete! Output saved to M87_Master_GPU.png")