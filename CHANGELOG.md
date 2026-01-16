# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Zig Core Implementation**: High-performance native implementation of core tensor operations
  - Matrix operations (multiply, transpose, trace, inverse)
  - Metric tensor functions (Minkowski, spherical polar)
  - C ABI exports for FFI compatibility
  - Zig unit tests for all core functions
- **Python FFI Bindings**: `autograv.zig_ffi` module for accessing Zig library from Python
  - ctypes-based interface with NumPy integration
  - Automatic library loading and detection
  - Graceful fallback to JAX if Zig unavailable
  - Type-safe wrappers matching Zig C ABI
- **Build System**: Comprehensive build infrastructure for Zig
  - `build.zig` configuration for cross-platform compilation
  - Shell script (`build_zig.sh`) for Linux/macOS
  - PowerShell script (`build_zig.ps1`) for Windows
  - Support for debug and release builds
- **CI/CD**: GitHub Actions workflow for Zig builds
  - Multi-platform testing (Linux, macOS, Windows)
  - Multiple Zig versions (0.11.0, 0.12.0)
  - Cross-compilation verification
  - Artifact uploads for build outputs
- **Documentation**:
  - `src/zig/README.md`: Complete Zig API reference
  - `ZIG_INTEGRATION.md`: Architecture and integration guide
  - `DEVELOPMENT.md`: Contributor guide for Zig development
  - Updated main README with Zig core section
- **Examples**:
  - `examples/zig_ffi_example.py`: Demonstrates FFI usage with performance comparison

### Changed
- Updated package description to mention optional Zig core
- Added Zig-related keywords to pyproject.toml
- Updated `.gitignore` to exclude Zig build artifacts

### Technical Details
- **Architecture**: Hybrid Python/Zig design
  - JAX for automatic differentiation and GPU/TPU
  - Zig for performance-critical native operations
  - Optional Zig core with automatic detection
- **Compatibility**: Zig library works standalone (no Python required)
- **Cross-platform**: Single Zig codebase compiles for Linux, macOS, Windows
- **Future**: Foundation for JAX/XLA custom operation integration

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
