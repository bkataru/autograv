"""
autograv: Automatic differentiation for numerical relativity using JAX

This library bridges numerical relativity and automatic differentiation by using JAX
to compute various tensors and quantities from general relativity given a metric function.
"""

from typing import Callable

import jax
import jax.numpy as jnp

# Configure JAX to use 64-bit precision for higher accuracy
jax.config.update("jax_enable_x64", True)

TOLERANCE = 1e-8


def close_to_zero(func):
    """Decorator to round off values close to zero in tensors to reduce floating point errors."""
    def wrapper(*args, **kwargs) -> jnp.ndarray:
        result: jnp.ndarray = func(*args, **kwargs)
        return jnp.where(jnp.abs(result) < TOLERANCE, 0.0, result)
    return wrapper


# ============================================================================
# Metric definitions
# ============================================================================

def minkowski_metric(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Returns the Minkowski metric in float64 precision with the (-1, 1, 1, 1) metric signature.
    
    Args:
        coordinates: Dummy coordinates (not used, but needed for JAX autodiff compatibility)
    
    Returns:
        Minkowski metric tensor
    """
    return jnp.diag(jnp.array([-1, 1, 1, 1], dtype=jnp.float64))


@close_to_zero
def spherical_polar_metric(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Standard metric for a 2-sphere in spherical polar coordinates.
    
    Args:
        coordinates: [r, theta, phi] in spherical polar coordinates
    
    Returns:
        Metric tensor for 2-sphere
    """
    r, theta, phi = coordinates
    return jnp.diag(jnp.array([1, r**2, r**2 * jnp.sin(theta)**2], dtype=jnp.float64))


# ============================================================================
# Christoffel symbols and related quantities
# ============================================================================

@close_to_zero
def christoffel_symbols(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Compute the Christoffel symbols of the second kind (affine connection coefficients).
    
    The Christoffel symbols are defined as:
    Γ^j_kl = (1/2) g^jm (∂g_mk/∂x^l + ∂g_lm/∂x^k - ∂g_kl/∂x^m)
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Christoffel symbols tensor of shape (n, n, n)
    """
    # Evaluate the metric tensor at the coordinates
    g = metric(coordinates)
    # Compute the inverse metric tensor
    g_inv = jnp.linalg.inv(g)
    # Obtain the "jacobian" of the metric tensor at the coordinates
    jacobian = jax.jacfwd(metric)(coordinates)  # this is kl;m
    
    return 0.5 * jnp.einsum('jm, klm -> jkl', g_inv, 
                            jnp.einsum('klm -> mkl', jacobian) + 
                            jnp.einsum('klm -> lmk', jacobian) - jacobian)


@close_to_zero
def torsion_tensor(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Compute the torsion tensor (antisymmetric part of the affine connection).
    
    For Christoffel symbols (symmetric connection), this should always be zero.
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Torsion tensor (should be zero for Christoffel symbols)
    """
    christoffels = christoffel_symbols(coordinates, metric)
    return christoffels - jnp.einsum('ijk -> ikj', christoffels)


# ============================================================================
# Curvature tensors
# ============================================================================

@close_to_zero
def riemann_tensor(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Compute the Riemann curvature tensor.
    
    The Riemann tensor encodes intrinsic curvature and is defined as:
    R^j_klm = ∂_m Γ^j_kl - ∂_l Γ^j_km + Γ^j_rm Γ^r_kl - Γ^j_rl Γ^r_km
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Riemann curvature tensor of shape (n, n, n, n)
    """
    christoffels = christoffel_symbols(coordinates, metric)
    jacobian = jax.jacfwd(christoffel_symbols)(coordinates, metric)  # computes jkl;m
    
    return (jacobian - jnp.einsum('jklm -> jkml', jacobian) + 
            jnp.einsum('jrm, rkl -> jklm', christoffels, christoffels) - 
            jnp.einsum('jrl, rkm -> jklm', christoffels, christoffels))


@close_to_zero
def ricci_tensor(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Compute the Ricci tensor (trace component of Riemann tensor).
    
    The Ricci tensor is defined as: R_kl = R^j_klj
    It encodes information about volume change in the presence of tidal forces.
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Ricci tensor of shape (n, n)
    """
    riemann = riemann_tensor(coordinates, metric)
    return jnp.einsum('jklj -> kl', riemann)  # contracting first and last indices


@close_to_zero
def ricci_scalar(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.float64:
    """Compute the Ricci scalar curvature.
    
    The Ricci scalar is defined as: R = g^kl R_kl
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Ricci scalar (a single float64 value)
    """
    g = metric(coordinates)
    g_inv = jnp.linalg.inv(g)
    ricci = ricci_tensor(coordinates, metric)
    
    return jnp.einsum('kl, kl -> ', g_inv, ricci)  # trace of the Ricci tensor


@close_to_zero
def kretschmann_invariant(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.float64:
    """Compute the Kretschmann invariant (scalar curvature invariant).
    
    The Kretschmann invariant is defined as: R^jklm R_jklm
    It is used to detect true physical singularities independent of coordinate choice.
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Kretschmann invariant (a single float64 value)
    """
    riemann = riemann_tensor(coordinates, metric)
    
    g = metric(coordinates)
    g_inv = jnp.linalg.inv(g)
    
    # Compute R^jklm by contracting with three inverse metric tensors
    riemann_upper = jnp.einsum('pj, qk, rl, ijkl -> ipqr', g_inv, g_inv, g_inv, riemann)
    # Compute R_jklm by contracting with one metric tensor
    riemann_lower = jnp.einsum('pi, ijkl -> pjkl', g, riemann)
    
    return jnp.einsum('ijkl, ijkl ->', riemann_upper, riemann_lower)


# ============================================================================
# Einstein field equations
# ============================================================================

@close_to_zero
def einstein_tensor(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Compute the Einstein tensor.
    
    The Einstein tensor is the crown jewel of general relativity, defined as:
    G_ij = R_ij - (1/2) g_ij R
    
    It forms the left-hand side of the Einstein field equations.
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Einstein tensor of shape (n, n)
    """
    g = metric(coordinates)
    rt = ricci_tensor(coordinates, metric)
    rs = ricci_scalar(coordinates, metric)
    
    return rt - 0.5 * g * rs


@close_to_zero
def stress_energy_momentum_tensor(coordinates: jnp.ndarray, metric: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Compute the stress-energy-momentum tensor from the Einstein field equations.
    
    The stress-energy-momentum tensor encodes mass-energy content and is related
    to the Einstein tensor by: T_ij = (c^4 / 8πG) G_ij
    
    Args:
        coordinates: Coordinates at which to evaluate
        metric: Metric function that takes coordinates and returns metric tensor
    
    Returns:
        Stress-energy-momentum tensor of shape (n, n)
    """
    G_tensor = einstein_tensor(coordinates, metric)
    
    # kappa = 8πG/c^4 where G is gravitational constant and c is speed of light
    kappa = (8 * jnp.pi * 6.67e-11) / ((299792458)**4)
    
    return G_tensor / kappa


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'TOLERANCE',
    'close_to_zero',
    'minkowski_metric',
    'spherical_polar_metric',
    'christoffel_symbols',
    'torsion_tensor',
    'riemann_tensor',
    'ricci_tensor',
    'ricci_scalar',
    'kretschmann_invariant',
    'einstein_tensor',
    'stress_energy_momentum_tensor',
]
