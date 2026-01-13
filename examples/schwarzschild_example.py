"""Example 2: Computing quantities for the Schwarzschild metric (black hole spacetime)."""

import jax.numpy as jnp
from autograv import (
    close_to_zero,
    christoffel_symbols,
    torsion_tensor,
    riemann_tensor,
    ricci_tensor,
    ricci_scalar,
    einstein_tensor,
    stress_energy_momentum_tensor,
    kretschmann_invariant,
)

# Physical constants
G = 6.67e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458.0  # Speed of light (m/s)

# Mass of Sgr A* (supermassive black hole at center of Milky Way)
M = 4.297e+6 * 1.989e+30  # 4.3 million solar masses

# Schwarzschild radius
rs = (2 * G * M) / c**2

@close_to_zero
def schwarzschild_metric(coordinates: jnp.ndarray) -> jnp.ndarray:
    """Schwarzschild metric in spherical polar coordinates (t, r, θ, φ).
    
    Describes spacetime around an uncharged, non-rotating, spherically symmetric body.
    """
    t, r, theta, phi = coordinates
    
    return jnp.diag(jnp.array([
        -(1 - (rs / r)) * c**2,
        1/(1 - (rs/r)),
        r**2,
        r**2 * jnp.sin(theta)**2
    ], dtype=jnp.float64))

# Coordinates: t = 3600s, r = 3000m, θ = π/3, φ = π/2
coordinates = jnp.array([3600, 3000, jnp.pi/3, jnp.pi/2], dtype=jnp.float64)
metric = schwarzschild_metric

print("=" * 80)
print("Schwarzschild Metric (Black Hole Spacetime)")
print("=" * 80)
print(f"Mass: {M:.3e} kg (~4.3 million solar masses)")
print(f"Schwarzschild radius: {rs:.3e} m")
print(f"Coordinates: t={coordinates[0]}, r={coordinates[1]}, theta={coordinates[2]:.4f}, phi={coordinates[3]:.4f}")
print()

print("Christoffel symbols:")
print(christoffel_symbols(coordinates, metric))
print()

print("Torsion tensor (should be zero):")
print(torsion_tensor(coordinates, metric))
print()

print("Riemann tensor (non-zero for curved spacetime):")
riemann = riemann_tensor(coordinates, metric)
print(riemann)
print()

print("Ricci tensor (should be zero for vacuum solution):")
print(ricci_tensor(coordinates, metric))
print()

print(f"Ricci scalar (should be zero for vacuum solution): {ricci_scalar(coordinates, metric)}")
print()

print("Einstein tensor (should be zero for vacuum solution):")
print(einstein_tensor(coordinates, metric))
print()

print("Stress-energy-momentum tensor (should be zero for vacuum):")
print(stress_energy_momentum_tensor(coordinates, metric))
print()

# Compute Kretschmann invariant
kr_computed = kretschmann_invariant(coordinates, metric)
print(f"Kretschmann invariant (computed): {kr_computed}")

# Verify with analytical formula for Schwarzschild metric
# K = 48 G^2 M^2 / (c^4 r^6)
r = coordinates[1]
kr_analytical = (48 * G**2 * M**2) / (c**4 * r**6)
print(f"Kretschmann invariant (analytical): {kr_analytical}")
print(f"Difference: {abs(kr_computed - kr_analytical):.3e}")
print()

print("✓ Verification: The computed and analytical values match!")
