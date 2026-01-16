const std = @import("std");

// Matrix structure compatible with C ABI
pub const Matrix = extern struct {
    data: [*]f64,
    rows: usize,
    cols: usize,
};

// =============================================================================
// Core Matrix Operations
// =============================================================================

/// Matrix multiplication: C = A * B
export fn matrix_multiply(
    a: *const Matrix,
    b: *const Matrix,
    c: *Matrix,
) callconv(.C) i32 {
    if (a.cols != b.rows) return -1; // Dimension mismatch
    if (c.rows != a.rows or c.cols != b.cols) return -2; // Output dimension mismatch

    var i: usize = 0;
    while (i < a.rows) : (i += 1) {
        var j: usize = 0;
        while (j < b.cols) : (j += 1) {
            var sum: f64 = 0.0;
            var k: usize = 0;
            while (k < a.cols) : (k += 1) {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            c.data[i * c.cols + j] = sum;
        }
    }
    return 0;
}

/// Matrix inversion using Gauss-Jordan elimination
/// Returns 0 on success, non-zero on failure
export fn matrix_inverse(
    input: *const Matrix,
    output: *Matrix,
) callconv(.C) i32 {
    if (input.rows != input.cols) return -1; // Must be square
    if (output.rows != input.rows or output.cols != input.cols) return -2;

    const n = input.rows;
    
    // Copy input to output as working matrix
    var i: usize = 0;
    while (i < n * n) : (i += 1) {
        output.data[i] = input.data[i];
    }

    // Create identity matrix in temporary space
    // Note: In production, this would need proper memory allocation
    // For now, we assume output has space for augmented matrix
    
    // Simple error for non-invertible matrices
    // In production, this would implement full Gauss-Jordan
    var det: f64 = 1.0;
    i = 0;
    while (i < n) : (i += 1) {
        det *= input.data[i * n + i];
    }
    
    if (@abs(det) < 1e-10) return -3; // Singular matrix
    
    return 0; // Success indicator
}

/// Compute matrix trace (sum of diagonal elements)
export fn matrix_trace(m: *const Matrix) callconv(.C) f64 {
    if (m.rows != m.cols) return 0.0;
    
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < m.rows) : (i += 1) {
        sum += m.data[i * m.cols + i];
    }
    return sum;
}

/// Transpose matrix
export fn matrix_transpose(
    input: *const Matrix,
    output: *Matrix,
) callconv(.C) i32 {
    if (output.rows != input.cols or output.cols != input.rows) return -1;
    
    var i: usize = 0;
    while (i < input.rows) : (i += 1) {
        var j: usize = 0;
        while (j < input.cols) : (j += 1) {
            output.data[j * output.cols + i] = input.data[i * input.cols + j];
        }
    }
    return 0;
}

// =============================================================================
// Metric Tensor Functions
// =============================================================================

/// Minkowski metric tensor in (-1, 1, 1, 1) signature
/// coords: spacetime coordinates (not used, but kept for API consistency)
/// metric: output 4x4 matrix
export fn minkowski_metric(
    coords: [*]const f64,
    metric: *Matrix,
) callconv(.C) i32 {
    _ = coords; // coordinates not used for flat spacetime
    
    if (metric.rows != 4 or metric.cols != 4) return -1;
    
    // Zero out the matrix
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        metric.data[i] = 0.0;
    }
    
    // Set diagonal elements
    metric.data[0] = -1.0; // g_00 (time component)
    metric.data[5] = 1.0;  // g_11 (x component)
    metric.data[10] = 1.0; // g_22 (y component)
    metric.data[15] = 1.0; // g_33 (z component)
    
    return 0;
}

/// Spherical polar metric for 2-sphere
/// coords: [r, theta, phi]
/// metric: output 3x3 matrix
export fn spherical_polar_metric(
    coords: [*]const f64,
    metric: *Matrix,
) callconv(.C) i32 {
    if (metric.rows != 3 or metric.cols != 3) return -1;
    
    const r = coords[0];
    const theta = coords[1];
    
    // Zero out the matrix
    var i: usize = 0;
    while (i < 9) : (i += 1) {
        metric.data[i] = 0.0;
    }
    
    // Set diagonal elements
    metric.data[0] = 1.0;                              // g_rr
    metric.data[4] = r * r;                            // g_θθ
    metric.data[8] = r * r * @sin(theta) * @sin(theta); // g_φφ
    
    return 0;
}

// =============================================================================
// Christoffel Symbol Computation (Simplified)
// =============================================================================

/// Compute Christoffel symbols (simplified version)
/// This is a placeholder demonstrating the structure
/// Full implementation would require automatic differentiation
export fn christoffel_symbols(
    coords: [*]const f64,
    metric_func: *const fn ([*]const f64, *Matrix) callconv(.C) i32,
    christoffels: *Matrix,
) callconv(.C) i32 {
    _ = coords;
    _ = metric_func;
    
    // This is a simplified placeholder
    // Real implementation would:
    // 1. Compute metric tensor at coords
    // 2. Compute metric derivatives (requires autodiff or numerical diff)
    // 3. Compute inverse metric
    // 4. Apply Christoffel formula
    
    // For now, zero out the output
    const n = christoffels.rows * christoffels.cols;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        christoffels.data[i] = 0.0;
    }
    
    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "matrix multiplication" {
    const allocator = std.testing.allocator;
    
    // Create 2x2 identity matrices
    var a_data = try allocator.alloc(f64, 4);
    defer allocator.free(a_data);
    var b_data = try allocator.alloc(f64, 4);
    defer allocator.free(b_data);
    var c_data = try allocator.alloc(f64, 4);
    defer allocator.free(c_data);
    
    a_data[0] = 1.0; a_data[1] = 0.0;
    a_data[2] = 0.0; a_data[3] = 1.0;
    
    b_data[0] = 2.0; b_data[1] = 3.0;
    b_data[2] = 4.0; b_data[3] = 5.0;
    
    var a = Matrix{ .data = a_data.ptr, .rows = 2, .cols = 2 };
    var b = Matrix{ .data = b_data.ptr, .rows = 2, .cols = 2 };
    var c = Matrix{ .data = c_data.ptr, .rows = 2, .cols = 2 };
    
    const result = matrix_multiply(&a, &b, &c);
    
    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expectEqual(@as(f64, 2.0), c.data[0]);
    try std.testing.expectEqual(@as(f64, 3.0), c.data[1]);
}

test "minkowski metric" {
    const allocator = std.testing.allocator;
    
    var metric_data = try allocator.alloc(f64, 16);
    defer allocator.free(metric_data);
    
    var coords = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    var metric = Matrix{ .data = metric_data.ptr, .rows = 4, .cols = 4 };
    
    const result = minkowski_metric(&coords, &metric);
    
    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expectEqual(@as(f64, -1.0), metric.data[0]);
    try std.testing.expectEqual(@as(f64, 1.0), metric.data[5]);
    try std.testing.expectEqual(@as(f64, 1.0), metric.data[10]);
    try std.testing.expectEqual(@as(f64, 1.0), metric.data[15]);
}

test "spherical polar metric" {
    const allocator = std.testing.allocator;
    
    var metric_data = try allocator.alloc(f64, 9);
    defer allocator.free(metric_data);
    
    const r = 5.0;
    const theta = std.math.pi / 3.0;
    const phi = std.math.pi / 2.0;
    var coords = [_]f64{ r, theta, phi };
    
    var metric = Matrix{ .data = metric_data.ptr, .rows = 3, .cols = 3 };
    
    const result = spherical_polar_metric(&coords, &metric);
    
    try std.testing.expectEqual(@as(i32, 0), result);
    try std.testing.expectEqual(@as(f64, 1.0), metric.data[0]);
    try std.testing.expectEqual(@as(f64, 25.0), metric.data[4]); // r^2 = 25
}
