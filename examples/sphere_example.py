"""Example 1: Computing quantities for the 2-sphere metric in spherical polar coordinates."""

import jax.numpy as jnp
from autograv import (
    spherical_polar_metric,
    christoffel_symbols,
    torsion_tensor,
    riemann_tensor,
    ricci_tensor,
    ricci_scalar,
    einstein_tensor,
    stress_energy_momentum_tensor,
    kretschmann_invariant,
)

# Coordinates: r = 5, θ = π/3, φ = π/2
coordinates = jnp.array([5, jnp.pi/3, jnp.pi/2], dtype=jnp.float64)
metric = spherical_polar_metric

print("=" * 80)
print("2-Sphere Metric in Spherical Polar Coordinates")
print("=" * 80)
print(f"Coordinates: r={coordinates[0]}, θ={coordinates[1]:.4f}, φ={coordinates[2]:.4f}")
print()

print("Christoffel symbols:")
print(christoffel_symbols(coordinates, metric))
print()

print("Torsion tensor (should be zero):")
print(torsion_tensor(coordinates, metric))
print()

print("Riemann tensor (should be zero for flat space):")
print(riemann_tensor(coordinates, metric))
print()

print("Ricci tensor (should be zero for flat space):")
print(ricci_tensor(coordinates, metric))
print()

print(f"Ricci scalar (should be zero for flat space): {ricci_scalar(coordinates, metric)}")
print()

print("Einstein tensor (should be zero for flat space):")
print(einstein_tensor(coordinates, metric))
print()

print("Stress-energy-momentum tensor (should be zero for vacuum):")
print(stress_energy_momentum_tensor(coordinates, metric))
print()

print(f"Kretschmann invariant (should be zero for flat space): {kretschmann_invariant(coordinates, metric)}")
