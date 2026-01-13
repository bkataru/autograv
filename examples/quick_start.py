"""
Quick Start Guide - autograv Library
=====================================

This guide shows you how to quickly get started with autograv.
"""

import jax.numpy as jnp
from autograv import (
    christoffel_symbols,
    riemann_tensor,
    ricci_tensor,
    ricci_scalar,
    einstein_tensor,
    kretschmann_invariant,
    spherical_polar_metric,
)

# ============================================================================
# Example 1: Define coordinates
# ============================================================================

# For a 2-sphere in spherical polar coordinates (r, θ, φ)
coordinates = jnp.array([5.0, jnp.pi/3, jnp.pi/2], dtype=jnp.float64)

print("Coordinates:", coordinates)

# ============================================================================
# Example 2: Compute Christoffel symbols
# ============================================================================

christoffels = christoffel_symbols(coordinates, spherical_polar_metric)
print("\nChristoffel symbols shape:", christoffels.shape)
print("Sample value Γ^1_12:", christoffels[1, 0, 1])

# ============================================================================
# Example 3: Compute curvature tensors
# ============================================================================

riemann = riemann_tensor(coordinates, spherical_polar_metric)
ricci = ricci_tensor(coordinates, spherical_polar_metric)
scalar = ricci_scalar(coordinates, spherical_polar_metric)

print("\nRiemann tensor shape:", riemann.shape)
print("Ricci tensor shape:", ricci.shape)
print("Ricci scalar:", scalar)

# ============================================================================
# Example 4: Compute Einstein tensor
# ============================================================================

einstein = einstein_tensor(coordinates, spherical_polar_metric)
print("\nEinstein tensor shape:", einstein.shape)

# ============================================================================
# Example 5: Define your own metric
# ============================================================================

from autograv import close_to_zero

@close_to_zero
def my_custom_metric(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Your custom metric function.
    
    Args:
        coordinates: Array of coordinates
    
    Returns:
        Metric tensor (2D array)
    """
    # Example: Simple diagonal metric
    x, y, z = coordinates
    return jnp.diag(jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64))

# Use it just like the built-in metrics
custom_coords = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
custom_christoffels = christoffel_symbols(custom_coords, my_custom_metric)
print("\nCustom metric Christoffel symbols shape:", custom_christoffels.shape)

# ============================================================================
# Example 6: Schwarzschild black hole
# ============================================================================

# Physical constants
G = 6.67e-11  # Gravitational constant
c = 299792458.0  # Speed of light
M = 1.989e30  # Solar mass (kg)

# Schwarzschild radius
rs = (2 * G * M) / c**2

@close_to_zero
def schwarzschild_metric(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Schwarzschild metric for a black hole."""
    t, r, theta, phi = coordinates
    
    return jnp.diag(jnp.array([
        -(1 - rs/r) * c**2,
        1/(1 - rs/r),
        r**2,
        r**2 * jnp.sin(theta)**2
    ], dtype=jnp.float64))

# Compute at some point outside the event horizon
bh_coords = jnp.array([0.0, rs * 10, jnp.pi/2, 0.0], dtype=jnp.float64)
kr = kretschmann_invariant(bh_coords, schwarzschild_metric)
print(f"\nKretschmann invariant at r=10*rs: {kr}")

print("\n" + "="*60)
print("See examples/ directory for more complete examples!")
print("="*60)
