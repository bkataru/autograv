# autograv

**Bridging numerical relativity and automatic differentiation using JAX**

`autograv` is a Python library that uses JAX and automatic differentiation to compute various tensors and quantities from Einstein's general theory of relativity. Given a metric function, it can calculate Christoffel symbols, curvature tensors, and solve the Einstein field equations with high numerical precision.

## Features

- **Automatic Differentiation**: Uses JAX's `jax.jacfwd` for forward-mode automatic differentiation to compute derivatives of metric tensors with exact numerical precision
- **Tensor Calculus**: Leverages `jax.numpy.einsum` for efficient Einstein summation notation operations
- **High Precision**: Configured to use 64-bit floating point arithmetic for maximum accuracy
- **Pure Functions**: All computations are functional and composable

## What can you compute?

Given a metric tensor function, `autograv` can compute:

- **Christoffel symbols** (affine connection coefficients)
- **Torsion tensor** (verification that connection is symmetric)
- **Riemann curvature tensor** (intrinsic curvature of spacetime)
- **Ricci tensor** and **Ricci scalar** (curvature related to volume change)
- **Einstein tensor** (left-hand side of Einstein field equations)
- **Stress-energy-momentum tensor** (mass-energy content)
- **Kretschmann invariant** (scalar curvature for detecting singularities)

## Installation

```bash
# Using uv
uv pip install autograv

# Or using pip
pip install autograv
```

## Quick Start

```python
import jax.numpy as jnp
from autograv import (
    spherical_polar_metric,
    christoffel_symbols,
    riemann_tensor,
    einstein_tensor,
)

# Define coordinates
coordinates = jnp.array([5, jnp.pi/3, jnp.pi/2], dtype=jnp.float64)

# Compute Christoffel symbols for the 2-sphere
christoffels = christoffel_symbols(coordinates, spherical_polar_metric)
print(christoffels)

# Compute Riemann tensor
riemann = riemann_tensor(coordinates, spherical_polar_metric)
print(riemann)
```

## Examples

The `examples/` directory contains complete examples:

- `sphere_example.py`: Computing quantities for a 2-sphere metric
- `schwarzschild_example.py`: Computing quantities for the Schwarzschild black hole metric

Run them with:

```bash
uv run python examples/sphere_example.py
uv run python examples/schwarzschild_example.py
```

## How it Works

### Automatic Differentiation

Traditional approaches to computing derivatives in physics use either:
- **Symbolic differentiation**: Exact but computationally expensive
- **Numerical differentiation**: Fast but prone to floating-point errors

**Automatic differentiation** (autodiff) combines the best of both worlds by:
1. Tracing computational operations to build a directed acyclic graph (DAG)
2. Computing gradients via the chain rule by traversing the graph
3. Achieving exact numerical precision at machine precision limits

### JAX Integration

JAX provides:
- `jax.jacfwd`: Forward-mode autodiff for computing Jacobians
- `jax.numpy.einsum`: Efficient Einstein summation for tensor operations
- NumPy-compatible API with GPU/TPU acceleration support

### Example: Christoffel Symbols

Given a metric tensor g_ij, the Christoffel symbols are:

```
Γ^j_kl = (1/2) g^jm (∂g_mk/∂x^l + ∂g_lm/∂x^k - ∂g_kl/∂x^m)
```

In code:

```python
def christoffel_symbols(coordinates, metric):
    g = metric(coordinates)
    g_inv = jnp.linalg.inv(g)
    jacobian = jax.jacfwd(metric)(coordinates)  # Automatic differentiation!
    
    return 0.5 * jnp.einsum('jm, klm -> jkl', g_inv,
                            jnp.einsum('klm -> mkl', jacobian) +
                            jnp.einsum('klm -> lmk', jacobian) - jacobian)
```

## API Reference

### Metrics

- `minkowski_metric(coordinates)`: Flat spacetime metric
- `spherical_polar_metric(coordinates)`: 2-sphere metric in (r, θ, φ)

### Core Functions

- `christoffel_symbols(coordinates, metric)`: Affine connection coefficients
- `torsion_tensor(coordinates, metric)`: Antisymmetric part of connection
- `riemann_tensor(coordinates, metric)`: Curvature tensor
- `ricci_tensor(coordinates, metric)`: Trace of Riemann tensor
- `ricci_scalar(coordinates, metric)`: Scalar curvature
- `kretschmann_invariant(coordinates, metric)`: Curvature invariant
- `einstein_tensor(coordinates, metric)`: G_ij = R_ij - (1/2)g_ij R
- `stress_energy_momentum_tensor(coordinates, metric)`: T_ij from Einstein equations

### Utilities

- `close_to_zero(func)`: Decorator to suppress near-zero numerical noise
- `TOLERANCE`: Threshold for zero suppression (default: 1e-8)

## Requirements

- Python 3.11+
- JAX (CPU-only on Windows, GPU/TPU support on Linux/macOS)
- NumPy

## Future Work

- Add more standard metrics (Kerr, Kerr-Newman, FRW, etc.)
- Implement Weyl tensor and Weyl invariant
- Support for JIT compilation with `@jax.jit`
- GPU/TPU acceleration examples
- Integration with differential equation solvers
- Visualization tools for curvature

## License

MIT

## Acknowledgments

Based on concepts from the blog post "Bridging numerical relativity and automatic differentiation using JAX". This project demonstrates the synergy between modern machine learning tools and classical physics computations.

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [Mathematical Methods for Physics](https://www.springer.com/gp/book/9780387095042) by Sadri Hassani
- [Einstein's General Theory of Relativity](https://en.wikipedia.org/wiki/General_relativity)
