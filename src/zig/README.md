# Zig Core for autograv

This directory contains the high-performance Zig implementation of core tensor operations for autograv.

## Overview

The Zig core provides:
- High-performance matrix operations (multiplication, inversion, trace)
- Metric tensor computations (Minkowski, spherical polar)
- C-compatible ABI for FFI integration with Python
- Future: Christoffel symbols and curvature tensor computations

## Why Zig?

Zig was chosen for the core implementation because:

1. **Performance**: Compiled to native code with no runtime overhead
2. **Safety**: Compile-time safety checks prevent common bugs
3. **C Interoperability**: Natural C ABI compatibility for FFI
4. **Cross-platform**: Single codebase compiles to Windows, Linux, macOS
5. **No dependencies**: Self-contained, no external runtime needed

## Building from Source

### Prerequisites

- Zig 0.11.0 or 0.12.0 (tested versions)
  - Download: https://ziglang.org/download/
  - Verify: `zig version`
  - Note: Other versions may work but are untested

### Build the Zig Library

```bash
# Build the shared library
zig build

# The library will be in zig-out/lib/
# - Linux: libautograv_core.so
# - macOS: libautograv_core.dylib  
# - Windows: autograv_core.dll

# Run Zig tests
zig build test
```

### Build Options

```bash
# Release build (optimized)
zig build -Doptimize=ReleaseFast

# Debug build (with symbols)
zig build -Doptimize=Debug

# Cross-compile for different platforms
zig build -Dtarget=x86_64-windows
zig build -Dtarget=aarch64-macos
```

## Using from Zig

```zig
const std = @import("std");
const autograv = @import("core.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Allocate matrix data
    var metric_data = try allocator.alloc(f64, 16);
    defer allocator.free(metric_data);

    // Create coordinates
    var coords = [_]f64{ 0.0, 1.0, 2.0, 3.0 };

    // Compute Minkowski metric
    var metric = autograv.Matrix{
        .data = metric_data.ptr,
        .rows = 4,
        .cols = 4,
    };

    const result = autograv.minkowski_metric(&coords, &metric);
    if (result == 0) {
        std.debug.print("Minkowski metric computed successfully\n", .{});
    }
}
```

## Using from Python

The Zig library is accessible from Python via the `zig_ffi` module:

```python
import numpy as np
from autograv.zig_ffi import (
    is_zig_available,
    minkowski_metric_zig,
    spherical_polar_metric_zig,
    matrix_multiply,
)

# Check if Zig library is available
if is_zig_available():
    print("Zig core library loaded successfully")
    
    # Compute Minkowski metric
    coords = np.array([0.0, 1.0, 2.0, 3.0])
    metric = minkowski_metric_zig(coords)
    print("Minkowski metric:")
    print(metric)
    
    # Compute spherical polar metric
    coords = np.array([5.0, np.pi/3, np.pi/2])
    metric = spherical_polar_metric_zig(coords)
    print("Spherical polar metric:")
    print(metric)
    
    # Matrix multiplication
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = matrix_multiply(a, b)
    print("Matrix product:")
    print(c)
else:
    print("Zig library not available, falling back to pure Python/JAX")
```

## API Reference

### Matrix Operations

#### `matrix_multiply(a, b, c) -> i32`
Multiply two matrices: C = A × B

**Parameters:**
- `a`: Input matrix A (m × n)
- `b`: Input matrix B (n × p)
- `c`: Output matrix C (m × p)

**Returns:** 0 on success, negative on error

#### `matrix_inverse(input, output) -> i32`
Compute matrix inverse using Gauss-Jordan elimination

**Parameters:**
- `input`: Input square matrix
- `output`: Output inverse matrix

**Returns:** 0 on success, negative on error

#### `matrix_trace(m) -> f64`
Compute trace (sum of diagonal elements)

**Parameters:**
- `m`: Input square matrix

**Returns:** Trace value

#### `matrix_transpose(input, output) -> i32`
Transpose a matrix

**Parameters:**
- `input`: Input matrix (m × n)
- `output`: Output matrix (n × m)

**Returns:** 0 on success, negative on error

### Metric Tensor Functions

#### `minkowski_metric(coords, metric) -> i32`
Compute Minkowski metric tensor with signature (-1, 1, 1, 1)

**Parameters:**
- `coords`: Spacetime coordinates [t, x, y, z] (4 elements)
- `metric`: Output 4×4 metric tensor

**Returns:** 0 on success, negative on error

#### `spherical_polar_metric(coords, metric) -> i32`
Compute spherical polar metric for 2-sphere

**Parameters:**
- `coords`: Spherical coordinates [r, θ, φ] (3 elements)
- `metric`: Output 3×3 metric tensor

**Returns:** 0 on success, negative on error

**Metric formula:**
```
ds² = dr² + r²dθ² + r²sin²(θ)dφ²
```

## Architecture

### Memory Model

The Zig implementation uses a simple C-compatible memory model:

```zig
pub const Matrix = extern struct {
    data: [*]f64,      // Pointer to matrix data (row-major)
    rows: usize,        // Number of rows
    cols: usize,        // Number of columns
};
```

- **Row-major layout**: Element (i,j) is at index `i * cols + j`
- **No ownership**: Caller manages memory allocation/deallocation
- **C ABI**: `extern struct` ensures C compatibility

### Error Handling

Functions return integer status codes:
- `0`: Success
- `-1`: Invalid dimensions
- `-2`: Output dimension mismatch
- `-3`: Singular matrix (non-invertible)

### Thread Safety

Current implementation is **not thread-safe**. All functions assume single-threaded access. Future versions may add:
- Immutable data structures
- Atomic operations
- Mutex-protected shared state

## Performance Characteristics

### Matrix Multiplication
- **Time complexity**: O(n³) for n×n matrices
- **Space complexity**: O(1) auxiliary space
- **SIMD**: Not yet vectorized (future optimization)

### Matrix Inversion
- **Algorithm**: Gauss-Jordan elimination (placeholder)
- **Time complexity**: O(n³)
- **Numerical stability**: Basic pivoting (production would use partial/full pivoting)

## Future Work

### Planned Features

1. **Automatic Differentiation**
   - Implement dual numbers for forward-mode AD
   - Compute Christoffel symbols via autodiff
   - Support higher-order derivatives

2. **Tensor Operations**
   - Einstein summation (einsum)
   - Tensor contraction
   - Index manipulation

3. **Curvature Computations**
   - Riemann curvature tensor
   - Ricci tensor and scalar
   - Kretschmann invariant

4. **Performance Optimizations**
   - SIMD vectorization
   - Multi-threading for large matrices
   - GPU compute via Vulkan/CUDA

5. **Additional Metrics**
   - Schwarzschild metric
   - Kerr metric (rotating black hole)
   - Friedmann–Lemaître–Robertson–Walker (FLRW)

### JAX/XLA Integration

The ultimate goal is to integrate with JAX/XLA for:
- JIT compilation of Zig code
- GPU/TPU acceleration
- Hybrid Zig+JAX pipelines

**Approach:**
1. Use Zig as a build system to compile XLA custom ops
2. Export Zig functions as XLA custom calls
3. Register with JAX's custom primitive system

**Challenges:**
- XLA C++ API complexity
- Build system integration
- Cross-platform compatibility

## Contributing

### Code Style

- Follow Zig's standard formatting: `zig fmt`
- Use explicit types where helpful
- Prefer comptime for type-level programming
- Document public APIs with doc comments

### Testing

All exported functions must have tests:

```zig
test "function_name" {
    const allocator = std.testing.allocator;
    
    // Test setup
    // ...
    
    try std.testing.expectEqual(expected, actual);
}
```

Run tests with:
```bash
zig build test
```

### Pull Requests

1. Run `zig fmt` on all changed files
2. Add tests for new functionality
3. Update this README if adding public APIs
4. Verify cross-platform compatibility (at minimum Linux + Windows)

## License

MIT License - see LICENSE file in repository root

## References

- [Zig Language Documentation](https://ziglang.org/documentation/master/)
- [Zig Build System](https://ziglang.org/learn/build-system/)
- [Python ctypes Documentation](https://docs.python.org/3/library/ctypes.html)
- [JAX Custom Operations](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
- [XLA Custom Calls](https://www.tensorflow.org/xla/custom_call)
