# Development Guide for autograv Zig Core

This guide is for contributors who want to work on the Zig implementation of autograv.

## Setup Development Environment

### Prerequisites

1. **Zig** (0.11.0 or newer)
   - Download: https://ziglang.org/download/
   - Verify: `zig version`

2. **Python** (3.11+)
   - Required for FFI testing
   - Install dependencies: `pip install numpy jax`

3. **Git**
   - For version control

### Directory Structure

```
autograv/
├── src/
│   ├── autograv/          # Python package
│   │   ├── __init__.py    # Main Python/JAX implementation
│   │   └── zig_ffi.py     # Python FFI bindings to Zig
│   └── zig/               # Zig source code
│       ├── core.zig       # Core tensor operations
│       └── README.md      # Zig API documentation
├── build.zig              # Zig build configuration
├── examples/
│   └── zig_ffi_example.py # FFI usage example
├── zig-out/               # Build output (gitignored)
│   └── lib/               # Compiled libraries
└── zig-cache/             # Build cache (gitignored)
```

## Development Workflow

### 1. Make Changes to Zig Code

Edit `src/zig/core.zig`:

```zig
// Add new function
export fn my_new_function(
    input: *const Matrix,
    output: *Matrix,
) callconv(.C) i32 {
    // Implementation
    return 0;
}
```

### 2. Add Tests

Add test to `src/zig/core.zig`:

```zig
test "my_new_function" {
    const allocator = std.testing.allocator;
    
    // Test setup
    var input_data = try allocator.alloc(f64, 9);
    defer allocator.free(input_data);
    
    // ... test implementation ...
    
    try std.testing.expectEqual(expected, actual);
}
```

### 3. Build and Test

```bash
# Build library
zig build

# Run tests
zig build test

# Format code
zig fmt src/zig/
```

### 4. Update Python FFI Bindings

Edit `src/autograv/zig_ffi.py`:

```python
# Add function signature
if _lib:
    _lib.my_new_function.argtypes = [
        ctypes.POINTER(Matrix),
        ctypes.POINTER(Matrix),
    ]
    _lib.my_new_function.restype = ctypes.c_int32

# Add Python wrapper
def my_new_function(input_arr: np.ndarray) -> np.ndarray:
    """Python wrapper for my_new_function."""
    if _lib is None:
        raise RuntimeError("Zig library not available")
    
    input_mat = Matrix.from_numpy(input_arr)
    output = np.zeros_like(input_arr)
    output_mat = Matrix.from_numpy(output)
    
    result = _lib.my_new_function(
        ctypes.byref(input_mat),
        ctypes.byref(output_mat),
    )
    
    if result != 0:
        raise RuntimeError(f"Function failed with code {result}")
    
    return output_mat.to_numpy()
```

### 5. Test Python Bindings

```bash
# Run Python example
python examples/zig_ffi_example.py

# Or test interactively
python -c "from src.autograv.zig_ffi import my_new_function; print(my_new_function(...))"
```

### 6. Update Documentation

Update `src/zig/README.md` with:
- Function signature
- Parameters and return values
- Usage example
- Performance characteristics

## Code Style

### Zig Code Style

Follow Zig's standard conventions:

```zig
// Use snake_case for functions and variables
pub fn matrix_multiply(a: *const Matrix, b: *const Matrix) void {}

// Use PascalCase for types
pub const Matrix = extern struct {};

// Explicit error handling
pub fn riskyOperation() !void {
    return error.SomethingWentWrong;
}

// Document public APIs
/// Compute the trace of a square matrix.
/// Returns the sum of diagonal elements.
export fn matrix_trace(m: *const Matrix) callconv(.C) f64 {}
```

**Format before committing:**
```bash
zig fmt src/zig/
zig fmt build.zig
```

### Python Code Style

Follow PEP 8 and type hints:

```python
def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices using Zig implementation.
    
    Args:
        a: First matrix (m × n)
        b: Second matrix (n × p)
    
    Returns:
        Result matrix (m × p)
    
    Raises:
        RuntimeError: If Zig library not available
        ValueError: If dimensions incompatible
    """
    pass
```

## Testing

### Unit Tests (Zig)

Run all Zig tests:
```bash
zig build test
```

Run specific test:
```bash
zig test src/zig/core.zig --test-filter "matrix_multiply"
```

### Integration Tests (Python)

Create tests that verify Zig and JAX produce identical results:

```python
import numpy as np
from autograv import spherical_polar_metric
from autograv.zig_ffi import spherical_polar_metric_zig, is_zig_available

def test_metric_consistency():
    if not is_zig_available():
        pytest.skip("Zig library not available")
    
    coords = np.array([5.0, np.pi/3, np.pi/2])
    
    # Compare implementations
    metric_jax = spherical_polar_metric(coords)
    metric_zig = spherical_polar_metric_zig(coords)
    
    np.testing.assert_allclose(metric_jax, metric_zig, rtol=1e-10)
```

## Debugging

### Debug Build

Build with debug symbols:
```bash
zig build -Doptimize=Debug
```

### Verbose Output

```bash
# Verbose build
zig build --verbose

# Show compilation steps
zig build -freference-trace
```

### GDB/LLDB Debugging

```bash
# Build with debug info
zig build -Doptimize=Debug

# Debug with GDB (Linux)
gdb zig-out/lib/libautograv_core.so

# Debug with LLDB (macOS)
lldb zig-out/lib/libautograv_core.dylib
```

### Python FFI Debugging

```python
import ctypes
ctypes.CDLL.RTLD_GLOBAL = ctypes.RTLD_NOW | ctypes.RTLD_GLOBAL

# Enable verbose ctypes
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Profiling Zig Code

Use built-in profiling:

```zig
const std = @import("std");

test "benchmark matrix multiply" {
    var timer = try std.time.Timer.start();
    
    // Run operation
    matrix_multiply(&a, &b, &c);
    
    const elapsed = timer.read();
    std.debug.print("Time: {d}ns\n", .{elapsed});
}
```

### Optimization Levels

```bash
# Debug (no optimization)
zig build -Doptimize=Debug

# Release with safety checks
zig build -Doptimize=ReleaseSafe

# Release optimized
zig build -Doptimize=ReleaseFast

# Smallest binary
zig build -Doptimize=ReleaseSmall
```

### SIMD Optimization (Future)

```zig
// Example SIMD vector operations
const Vec4 = @Vector(4, f64);

fn vectorized_add(a: Vec4, b: Vec4) Vec4 {
    return a + b;  // SIMD add
}
```

## Common Issues

### Library Not Found

**Problem:** Python can't find the Zig library

**Solution:**
```bash
# Ensure library is built
zig build

# Check output directory
ls -la zig-out/lib/

# Set PYTHONPATH if needed
export PYTHONPATH=/path/to/autograv:$PYTHONPATH
```

### ABI Compatibility

**Problem:** Crashes or corrupt data when calling from Python

**Solution:**
- Ensure `extern struct` is used for FFI types
- Use `callconv(.C)` for exported functions
- Verify pointer alignment
- Check row-major vs column-major layout

### Cross-Compilation Issues

**Problem:** Library builds on one platform but not another

**Solution:**
```bash
# List available targets
zig targets

# Build for specific target
zig build -Dtarget=x86_64-linux-gnu

# Check dependencies
ldd zig-out/lib/libautograv_core.so  # Linux
otool -L zig-out/lib/libautograv_core.dylib  # macOS
```

## Advanced Topics

### Adding Automatic Differentiation

Future work: Implement dual numbers for forward-mode AD

```zig
pub const Dual = struct {
    value: f64,
    derivative: f64,
    
    pub fn add(a: Dual, b: Dual) Dual {
        return .{
            .value = a.value + b.value,
            .derivative = a.derivative + b.derivative,
        };
    }
};
```

### JAX/XLA Integration

To create XLA custom operations:

1. Create C++ wrapper for Zig functions
2. Register with XLA
3. Create JAX primitive

See `ZIG_INTEGRATION.md` for detailed plans.

### GPU Acceleration

Future: Use Zig with Vulkan or CUDA compute

```zig
// Conceptual - not yet implemented
const vk = @import("vulkan");

pub fn gpu_matrix_multiply(a: Matrix, b: Matrix) !Matrix {
    // Initialize Vulkan
    // Upload data to GPU
    // Run compute shader
    // Download result
}
```

## Contributing

### Pull Request Checklist

Before submitting a PR:

- [ ] Code formatted: `zig fmt src/zig/`
- [ ] Tests pass: `zig build test`
- [ ] Python bindings updated (if needed)
- [ ] Documentation updated
- [ ] Example works: `python examples/zig_ffi_example.py`
- [ ] CHANGELOG.md updated
- [ ] Cross-platform compatibility verified

### Commit Messages

Follow conventional commits:

```
feat(zig): add matrix inversion function
fix(ffi): correct pointer handling in matrix_multiply
docs(zig): update API reference for new functions
test(zig): add benchmarks for matrix operations
```

### Code Review

Reviewers will check:
- Memory safety (no leaks, use-after-free)
- Error handling (all error cases covered)
- Performance (no obvious inefficiencies)
- Documentation (public APIs documented)
- Tests (adequate coverage)

## Resources

### Zig Language

- [Official Documentation](https://ziglang.org/documentation/master/)
- [Zig Standard Library](https://ziglang.org/documentation/master/std/)
- [Zig Build System](https://ziglang.org/learn/build-system/)
- [Zig Language Reference](https://ziglang.org/documentation/master/#Language-Reference)

### FFI and Interop

- [Python ctypes](https://docs.python.org/3/library/ctypes.html)
- [NumPy C-API](https://numpy.org/doc/stable/reference/c-api/)
- [Zig C Interop](https://ziglang.org/documentation/master/#C)

### Performance

- [Zig SIMD](https://ziglang.org/documentation/master/#Vectors)
- [Cache-Oblivious Algorithms](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
- [BLAS/LAPACK](http://www.netlib.org/blas/)

### General Relativity

- [Carroll - Spacetime and Geometry](https://www.preposterousuniverse.com/spacetimeandgeometry/)
- [Wald - General Relativity](https://press.uchicago.edu/ucp/books/book/chicago/G/bo5952261.html)

## Questions?

- Open an issue on GitHub
- Check existing documentation in `src/zig/README.md`
- See `ZIG_INTEGRATION.md` for architecture details
