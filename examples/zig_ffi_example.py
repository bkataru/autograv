#!/usr/bin/env python3
"""
Example: Using the Zig Core Library from Python

This example demonstrates how to use the high-performance Zig implementation
of tensor operations through Python's FFI interface.
"""

import numpy as np

try:
    from autograv.zig_ffi import (
        is_zig_available,
        get_library_path,
        minkowski_metric_zig,
        spherical_polar_metric_zig,
        matrix_multiply,
        matrix_trace,
    )
except ImportError:
    print("Warning: zig_ffi module not found. Please ensure autograv is installed.")
    exit(1)


def main():
    print("=" * 70)
    print("Autograv Zig Core FFI Example")
    print("=" * 70)
    print()
    
    # Check if Zig library is available
    if not is_zig_available():
        print("❌ Zig core library is not available")
        print()
        print("To use the Zig core:")
        print("1. Install Zig: https://ziglang.org/download/")
        print("2. Build the library: zig build")
        print("3. Ensure the library is in zig-out/lib/")
        print()
        print("Falling back to pure Python/JAX implementation...")
        return
    
    print("✅ Zig core library loaded successfully")
    print(f"   Library path: {get_library_path()}")
    print()
    
    # Example 1: Minkowski Metric
    print("-" * 70)
    print("Example 1: Minkowski Metric (Flat Spacetime)")
    print("-" * 70)
    
    coords = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    print(f"Coordinates: t={coords[0]}, x={coords[1]}, y={coords[2]}, z={coords[3]}")
    print()
    
    metric = minkowski_metric_zig(coords)
    print("Minkowski metric tensor g_μν with signature (-1, 1, 1, 1):")
    print(metric)
    print()
    
    # Verify properties
    trace = matrix_trace(metric)
    print(f"Trace: {trace:.1f} (should be 2.0 for signature -+++)")
    print()
    
    # Example 2: Spherical Polar Metric
    print("-" * 70)
    print("Example 2: Spherical Polar Metric (2-Sphere)")
    print("-" * 70)
    
    r = 5.0
    theta = np.pi / 3.0  # 60 degrees
    phi = np.pi / 2.0    # 90 degrees
    coords = np.array([r, theta, phi], dtype=np.float64)
    
    print(f"Coordinates: r={r}, θ={theta:.4f}, φ={phi:.4f}")
    print()
    
    metric = spherical_polar_metric_zig(coords)
    print("Spherical polar metric tensor g_ij:")
    print(metric)
    print()
    
    # Verify diagonal elements
    print("Metric components:")
    print(f"  g_rr = {metric[0, 0]:.4f} (should be 1.0)")
    print(f"  g_θθ = {metric[1, 1]:.4f} (should be r² = {r**2:.4f})")
    print(f"  g_φφ = {metric[2, 2]:.4f} (should be r²sin²θ = {r**2 * np.sin(theta)**2:.4f})")
    print()
    
    # Example 3: Matrix Operations
    print("-" * 70)
    print("Example 3: Matrix Operations")
    print("-" * 70)
    
    # Create test matrices
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float64)
    
    B = np.array([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ], dtype=np.float64)
    
    print("Matrix A (2×3):")
    print(A)
    print()
    
    print("Matrix B (3×2):")
    print(B)
    print()
    
    # Matrix multiplication using Zig
    C = matrix_multiply(A, B)
    print("Matrix C = A × B (2×2):")
    print(C)
    print()
    
    # Verify with NumPy
    C_numpy = A @ B
    print("Verification with NumPy:")
    print(C_numpy)
    print()
    
    difference = np.abs(C - C_numpy).max()
    print(f"Maximum difference: {difference:.2e}")
    if difference < 1e-10:
        print("✅ Results match!")
    else:
        print("❌ Results differ!")
    print()
    
    # Example 4: Performance Comparison
    print("-" * 70)
    print("Example 4: Performance Comparison (Zig vs NumPy)")
    print("-" * 70)
    
    import time
    
    # Create larger matrices for timing
    n = 100
    A_large = np.random.randn(n, n).astype(np.float64)
    B_large = np.random.randn(n, n).astype(np.float64)
    
    # Warm-up runs
    for _ in range(3):
        _ = matrix_multiply(A_large, B_large)
        _ = A_large @ B_large
    
    # Time Zig implementation with more iterations for statistical significance
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        C_zig = matrix_multiply(A_large, B_large)
    zig_time = (time.perf_counter() - start) / iterations
    
    # Time NumPy implementation
    start = time.perf_counter()
    for _ in range(iterations):
        C_numpy = A_large @ B_large
    numpy_time = (time.perf_counter() - start) / iterations
    
    print(f"Matrix size: {n}×{n}")
    print(f"Iterations: {iterations}")
    print(f"Zig implementation:   {zig_time*1000:.3f} ms")
    print(f"NumPy implementation: {numpy_time*1000:.3f} ms")
    print(f"Speedup: {numpy_time/zig_time:.2f}x")
    print()
    print("Note: NumPy uses optimized BLAS libraries (MKL, OpenBLAS)")
    print("      The Zig implementation is a simple O(n³) algorithm")
    print("      Future optimizations: SIMD, cache blocking, parallelization")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The Zig core library provides:")
    print("  ✅ High-performance native code execution")
    print("  ✅ Seamless Python integration via ctypes")
    print("  ✅ Compatible with NumPy arrays")
    print("  ✅ Cross-platform support (Linux, macOS, Windows)")
    print()
    print("Future enhancements:")
    print("  🚧 Automatic differentiation for Christoffel symbols")
    print("  🚧 Curvature tensor computations")
    print("  🚧 JAX/XLA integration for GPU acceleration")
    print("  🚧 SIMD vectorization and multi-threading")
    print()


if __name__ == "__main__":
    main()
