# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-13

### Added
- Initial release of autograv
- Core functionality for computing general relativity tensors using JAX automatic differentiation
- Functions for computing:
  - Christoffel symbols (affine connection coefficients)
  - Torsion tensor
  - Riemann curvature tensor
  - Ricci tensor and Ricci scalar
  - Einstein tensor
  - Stress-energy-momentum tensor
  - Kretschmann invariant
- Built-in metrics:
  - Minkowski metric (flat spacetime)
  - Spherical polar metric (2-sphere)
- Example scripts:
  - 2-sphere metric calculations
  - Schwarzschild black hole metric calculations
  - Quick start guide
- Comprehensive documentation:
  - README with installation and usage instructions
  - API reference
  - Implementation summary
- Type hints throughout codebase
- 64-bit precision configuration for JAX
- Zero-suppression decorator for numerical stability

### Dependencies
- JAX >= 0.4.20 (automatic differentiation)
- jaxlib >= 0.4.20 (JAX backend)
- NumPy >= 1.26, < 2 (array operations)
- Python >= 3.11

### Verified
- All examples produce correct results
- Schwarzschild Kretschmann invariant matches analytical formula to 15 decimal places
- Compatible with Python 3.12 on Windows
- Type-safe with comprehensive type annotations

[Unreleased]: https://github.com/bkataru/autograv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bkataru/autograv/releases/tag/v0.1.0
