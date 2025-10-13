import numpy as np
import scipy

def calculate_kuramoto_sivashinsky_residual(u_solution, dt, dx, mean_residual=True):
    """
    Calculates the residual of the 1D Kuramoto-Sivashinsky equation:
    PDE: u_t + u * u_x + u_xx + u_xxxx = 0
    This function uses second-order central finite differences for approximations.

    Args:
        u_solution (np.ndarray): Numerical solution of the KS equation.
                                 Expected shape is (batch_size, num_time_points, num_spatial_points).
                                 Should be a floating-point array.
        dt (float): Time step size (Δt).
        dx (float): Spatial step size (Δx).

    Returns:
        np.ndarray: The residual of the KS equation.
                    Shape: (batch_size, num_time_points-2, num_spatial_points-4).
                    The residual is computed for interior points where all
                    finite difference stencils are well-defined.
    """
    if not isinstance(u_solution, np.ndarray):
        u = np.asarray(u_solution, dtype=np.float64)
    else:
        u = u_solution

    if not np.issubdtype(u.dtype, np.floating):
        u = u.astype(np.float64)

    if u.ndim != 3:
        raise ValueError(f"Input array u_solution must be 3-dimensional (batch, time, space), got {u.ndim} dimensions.")

    N_batch, N_time, N_space = u.shape

    if N_time < 3:
        raise ValueError(f"Time dimension must be at least 3 for temporal derivative, got {N_time}.")
    if N_space < 5:
        raise ValueError(f"Space dimension must be at least 5 for fourth-order spatial derivative, got {N_space}.")
    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}.")
    if dx <= 0:
        raise ValueError(f"Spatial step dx must be positive, got {dx}.")

    # Temporal derivative: u_t ≈ (u(t+Δt) - u(t-Δt)) / (2*Δt)
    # This derivative is evaluated at grid points (batch, t_center, x_center_spatial_derivs)
    # Slicing u[:, 2:, ...] corresponds to u(t+Δt)
    # Slicing u[:, :-2, ...] corresponds to u(t-Δt)
    # The spatial slicing 2:-2 ensures all terms are aligned on the same spatial grid points
    # after considering the widest spatial stencil (u_xxxx).
    u_t = (u[:, 2:, 2:-2] - u[:, :-2, 2:-2]) / (2 * dt)

    # u term for the non-linear part (u * u_x). This is u at the center of all stencils.
    # Corresponds to time index t_center and spatial indices for the reduced grid.
    u_center = u[:, 1:-1, 2:-2]

    # First spatial derivative: u_x ≈ (u(x+Δx) - u(x-Δx)) / (2*Δx)
    # Evaluated at (batch, t_center, x_center_spatial_derivs)
    # u[:, 1:-1, 3:-1] corresponds to u(x+Δx) at t_center.
    # u[:, 1:-1, 1:-3] corresponds to u(x-Δx) at t_center.
    u_x = (u[:, 1:-1, 3:-1] - u[:, 1:-1, 1:-3]) / (2 * dx)

    # Second spatial derivative: u_xx ≈ (u(x+Δx) - 2u(x) + u(x-Δx)) / (Δx^2)
    # Evaluated at (batch, t_center, x_center_spatial_derivs)
    u_xx = (u[:, 1:-1, 3:-1] - 2 * u[:, 1:-1, 2:-2] + u[:, 1:-1, 1:-3]) / (dx**2)

    # Fourth spatial derivative: u_xxxx ≈ (u(x+2Δx) - 4u(x+Δx) + 6u(x) - 4u(x-Δx) + u(x-2Δx)) / (Δx^4)
    # Evaluated at (batch, t_center, x_center_spatial_derivs)
    u_xxxx = (u[:, 1:-1, 4:] - \
              4 * u[:, 1:-1, 3:-1] + \
              6 * u[:, 1:-1, 2:-2] - \
              4 * u[:, 1:-1, 1:-3] + \
              u[:, 1:-1, :-4]) / (dx**4)

    # All derivative terms (u_t, u_x, u_xx, u_xxxx) and u_center now have the
    # shape (N_batch, N_time-2, N_space-4).

    # Calculate the residual: R = u_t + u * u_x + u_xx + u_xxxx
    residual = u_t + u_center * u_x + u_xx + u_xxxx

    if mean_residual:
        return np.abs(residual).reshape(residual.shape[0], -1).mean(axis=-1)
    else:
        return residual

def two_point_corr(data, x, ens,axis):
    """
    data = [nx, ny]
    axis= 0
    ens = 1

    """
    r = np.diff(x).mean()
    f, tke = scipy.signal.welch(data, fs=1/r, nperseg=256,scaling='spectrum', axis=axis, return_onesided=False)
    R = np.fft.ifft(tke,axis=axis).real
    Rtotal = R.mean(axis=ens)
    R_half = Rtotal[:Rtotal.shape[0]//2]
    distance = np.arange(R_half.shape[0])*r
    return distance, R_half/R_half[0] 