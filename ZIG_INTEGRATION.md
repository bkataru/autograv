# Zig Integration with autograv

## Overview

This document describes the integration of Zig as a high-performance core for the autograv library, providing both direct Zig usage and Python FFI bindings.

## Architecture

### Hybrid Python/Zig Design

```
┌─────────────────────────────────────────────────────────────┐
│                        Python Layer                          │
│  - High-level API (autograv/__init__.py)                    │
│  - JAX integration for autodiff                             │
│  - Example scripts and documentation                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────────┐     ┌──────────────────────┐
│   JAX Backend     │     │    Zig Core (FFI)    │
│  - Autodiff       │     │  - Matrix ops        │
│  - GPU/TPU        │     │  - Metrics           │
│  - JIT compile    │     │  - Performance       │
└───────────────────┘     └──────────────────────┘
```

### Component Responsibilities

**Python/JAX Layer:**
- Automatic differentiation (gradients, Jacobians)
- High-level tensor operations (einsum)
- GPU/TPU acceleration
- Numerical precision configuration
- User-facing API

**Zig Core:**
- Basic matrix operations (multiply, inverse, transpose)
- Metric tensor computations
- Future: Christoffel symbols via manual or automatic differentiation
- C ABI exports for FFI
- Cross-platform compiled performance

## Unique Capabilities

### 1. Performance-Critical Path

The Zig core excels at:
- **Hot loop optimization**: Compiled native code with zero runtime overhead
- **Memory efficiency**: Explicit control over allocations
- **Predictable performance**: No GC pauses or JIT warmup

Use Zig for:
- Large matrix operations that don't need autodiff
- Embedded systems or resource-constrained environments
- Real-time applications requiring deterministic performance

### 2. Standalone Zig Library

The Zig implementation can be used **without Python**:

```zig
// Pure Zig usage - no Python required
const autograv = @import("autograv/core.zig");

pub fn main() !void {
    var coords = [_]f64{ 5.0, std.math.pi / 3.0, std.math.pi / 2.0 };
    var metric_data: [9]f64 = undefined;
    var metric = autograv.Matrix{
        .data = &metric_data,
        .rows = 3,
        .cols = 3,
    };
    
    _ = autograv.spherical_polar_metric(&coords, &metric);
    // Use metric...
}
```

This enables:
- Integration in Zig applications
- Use in systems programming contexts
- Embedded or bare-metal environments

### 3. Cross-Platform Binary Distribution

Zig's cross-compilation capabilities allow building for multiple platforms from a single machine:

```bash
# Build for all platforms
zig build -Dtarget=x86_64-linux
zig build -Dtarget=x86_64-windows
zig build -Dtarget=x86_64-macos
zig build -Dtarget=aarch64-macos
```

Benefits:
- No need for platform-specific build servers
- Consistent binary across platforms
- Easier PyPI wheel distribution

### 4. FFI Flexibility

The C ABI compatibility means the Zig core can be called from:
- Python (via ctypes, CFFI, or pybind11)
- JavaScript/Node.js (via node-ffi)
- Ruby (via FFI gem)
- Julia (via ccall)
- Any language with C FFI support

### 5. Future: JAX/XLA Custom Operations

The long-term goal is to use Zig to create XLA custom operations:

```python
# Future: Zig-backed JAX custom primitive
from jax import custom_vjp
import autograv.zig_xla

@custom_vjp
def christoffel_symbols_zig(coords, metric):
    # Forward pass uses Zig implementation
    return autograv.zig_xla.christoffel_symbols(coords, metric)

# VJP for backward pass autodiff
def christoffel_symbols_zig_vjp(coords, metric):
    # ...
```

This would enable:
- GPU acceleration of Zig code via XLA
- Integration into JAX's autodiff system
- JIT compilation with XLA optimizations

## Current Implementation Status

### ✅ Completed

- [x] Zig build system (build.zig)
- [x] Core matrix operations (multiply, transpose, trace)
- [x] Metric tensor functions (Minkowski, spherical polar)
- [x] C ABI exports for FFI
- [x] Python ctypes bindings (zig_ffi.py)
- [x] Example demonstrating Zig FFI usage
- [x] Comprehensive documentation
- [x] Unit tests for core functions

### 🚧 In Progress / Future Work

- [ ] Matrix inversion (currently placeholder)
- [ ] Automatic differentiation in Zig
- [ ] Christoffel symbols computation
- [ ] Riemann and Ricci tensors
- [ ] Einstein tensor
- [ ] JAX/XLA integration
- [ ] SIMD vectorization
- [ ] Multi-threading for large operations
- [ ] Additional metrics (Schwarzschild, Kerr, FLRW)

## Building and Distribution

### Development Workflow

1. **Develop Zig code**: Edit `src/zig/core.zig`
2. **Test**: `zig build test`
3. **Build library**: `zig build`
4. **Test Python bindings**: `python examples/zig_ffi_example.py`

### PyPI Distribution

For PyPI packages with compiled extensions:

**Option 1: Pure Python Wheel (Current)**
- Distribute Python code only
- Users build Zig library locally (optional)
- Falls back to JAX if Zig unavailable

**Option 2: Platform-Specific Wheels**
- Build wheels for each platform (manylinux, macosx, win_amd64)
- Include pre-compiled Zig library
- Requires wheel-building infrastructure

**Option 3: Source Distribution + Build Hook**
- Include Zig source in sdist
- Build during `pip install` if Zig available
- Requires users to have Zig installed

**Recommended**: Start with Option 1, transition to Option 2 as project matures.

### CI/CD Integration

Add to `.github/workflows/build.yml`:

```yaml
name: Build Zig Core

on: [push, pull_request]

jobs:
  build-zig:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Zig
      uses: goto-bus-stop/setup-zig@v2
      with:
        version: 0.11.0
    
    - name: Build Zig library
      run: zig build
    
    - name: Run Zig tests
      run: zig build test
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: zig-lib-${{ matrix.os }}
        path: zig-out/lib/
```

## Performance Considerations

### When to Use Zig vs JAX

**Use Zig when:**
- No automatic differentiation needed
- Deterministic performance required
- Memory usage critical
- Deploying to embedded/resource-constrained systems
- Cross-language interop needed

**Use JAX when:**
- Need automatic differentiation
- Want GPU/TPU acceleration
- Using JAX's JIT compilation
- Working with existing JAX code
- Prototyping new algorithms

### Benchmarks

Initial benchmarks (100×100 matrix multiplication):
- Zig (naive O(n³)): ~X ms
- NumPy (BLAS): ~Y ms
- JAX (CPU): ~Z ms
- JAX (GPU): ~W ms

*Note: Actual benchmarks to be added after library compilation*

### Optimization Opportunities

1. **SIMD Vectorization**
   ```zig
   const vec = @Vector(4, f64);
   // Vectorize inner loops
   ```

2. **Cache Blocking**
   ```zig
   // Tile matrix multiplication for cache efficiency
   const BLOCK_SIZE = 64;
   ```

3. **Multi-threading**
   ```zig
   // Parallelize outer loops
   var threads: [4]std.Thread = undefined;
   ```

## Integration with JAX/XLA

### Challenge: Building XLA from Zig

The issue mentions "figure out how to use Zig as a build system to successfully build jax and/or xla."

**Reality Check**: JAX/XLA are massive C++ projects with complex build systems (Bazel). Using Zig as the primary build system for XLA is not practical.

**Practical Approach**:
1. Build Zig library independently
2. Create XLA custom call interface in C++
3. Link Zig library with XLA custom op
4. Register with JAX

### XLA Custom Call Example (Future)

```cpp
// xla_custom_ops.cc
extern "C" {
    // Zig functions
    int minkowski_metric(const double* coords, Matrix* metric);
}

void MinkowskiMetricXLA(void* out, const void** in) {
    const double* coords = static_cast<const double*>(in[0]);
    Matrix* metric = static_cast<Matrix*>(out);
    minkowski_metric(coords, metric);
}

XLA_REGISTER_CUSTOM_CALL("minkowski_metric", MinkowskiMetricXLA);
```

```python
# Python side
from jax.lib import xla_client

def minkowski_metric_xla(coords):
    return xla_client.ops.CustomCall(
        c, b"minkowski_metric",
        operands=[coords],
        shape=xla_client.Shape.array_shape(np.dtype(np.float64), (4, 4)),
    )
```

## Testing Strategy

### Unit Tests

Zig tests for core functionality:
```zig
test "matrix operations" {
    // Test each operation
}
```

Python tests for FFI:
```python
def test_zig_ffi():
    if is_zig_available():
        # Test FFI bindings
    else:
        pytest.skip("Zig library not available")
```

### Integration Tests

Compare Zig and JAX outputs:
```python
def test_metric_consistency():
    coords = np.array([5.0, np.pi/3, np.pi/2])
    
    # JAX version
    metric_jax = spherical_polar_metric(coords)
    
    # Zig version
    if is_zig_available():
        metric_zig = spherical_polar_metric_zig(coords)
        np.testing.assert_allclose(metric_jax, metric_zig, rtol=1e-10)
```

## Documentation Updates

### README.md Updates Needed

Add section on Zig core:
```markdown
## Zig Core (Optional)

For maximum performance, autograv includes an optional Zig core:

### Installation
1. Install Zig: https://ziglang.org/download/
2. Build: `zig build`
3. The library will be loaded automatically if available

### Benefits
- Native code performance
- Cross-platform compatibility
- Can be used from Zig directly
- FFI compatible with multiple languages
```

## Conclusion

The Zig integration provides:

1. **High performance**: Native compiled code for critical operations
2. **Flexibility**: Use from Python, Zig, or other languages
3. **Cross-platform**: Single codebase for all platforms
4. **Future-proof**: Foundation for JAX/XLA integration
5. **Optional**: Falls back to pure JAX if unavailable

This hybrid approach combines the best of both worlds:
- JAX for autodiff and GPU acceleration
- Zig for performance-critical kernels and standalone usage
