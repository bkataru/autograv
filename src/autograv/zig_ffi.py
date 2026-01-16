"""
Python FFI bindings for the Zig autograv core library.

This module provides a Python interface to the high-performance Zig
implementation of tensor operations for general relativity.
"""

import ctypes
import os
import sys
import platform
import numpy as np
from typing import Optional, Callable
from pathlib import Path


# =============================================================================
# Library Loading
# =============================================================================

def _find_library() -> Optional[str]:
    """Locate the autograv_core shared library."""
    # Determine library name based on platform
    if platform.system() == "Windows":
        lib_name = "autograv_core.dll"
    elif platform.system() == "Darwin":
        lib_name = "libautograv_core.dylib"
    else:
        lib_name = "libautograv_core.so"
    
    # Search paths
    search_paths = [
        Path(__file__).parent / "lib",  # Package lib directory
        Path(__file__).parent / "zig-out" / "lib",  # Build output
        Path.cwd() / "zig-out" / "lib",  # Local build
    ]
    
    for path in search_paths:
        lib_path = path / lib_name
        if lib_path.exists():
            return str(lib_path)
    
    return None


# Try to load the library
_lib_path = _find_library()
_lib = None

if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError as e:
        import warnings
        warnings.warn(f"Failed to load Zig library from {_lib_path}: {e}")


# =============================================================================
# C Structure Definitions
# =============================================================================

class Matrix(ctypes.Structure):
    """C-compatible matrix structure."""
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
    ]
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Matrix':
        """Create a Matrix from a NumPy array."""
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        
        data_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return cls(data=data_ptr, rows=arr.shape[0], cols=arr.shape[1])
    
    def to_numpy(self) -> np.ndarray:
        """Convert Matrix to NumPy array."""
        size = self.rows * self.cols
        arr = np.ctypeslib.as_array(self.data, shape=(self.rows, self.cols))
        return arr.copy()


# =============================================================================
# Function Signatures
# =============================================================================

if _lib:
    # Matrix operations
    _lib.matrix_multiply.argtypes = [
        ctypes.POINTER(Matrix),
        ctypes.POINTER(Matrix),
        ctypes.POINTER(Matrix),
    ]
    _lib.matrix_multiply.restype = ctypes.c_int32
    
    _lib.matrix_inverse.argtypes = [
        ctypes.POINTER(Matrix),
        ctypes.POINTER(Matrix),
    ]
    _lib.matrix_inverse.restype = ctypes.c_int32
    
    _lib.matrix_trace.argtypes = [ctypes.POINTER(Matrix)]
    _lib.matrix_trace.restype = ctypes.c_double
    
    _lib.matrix_transpose.argtypes = [
        ctypes.POINTER(Matrix),
        ctypes.POINTER(Matrix),
    ]
    _lib.matrix_transpose.restype = ctypes.c_int32
    
    # Metric functions
    _lib.minkowski_metric.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(Matrix),
    ]
    _lib.minkowski_metric.restype = ctypes.c_int32
    
    _lib.spherical_polar_metric.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(Matrix),
    ]
    _lib.spherical_polar_metric.restype = ctypes.c_int32


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices using the Zig implementation.
    
    Args:
        a: First matrix (m x n)
        b: Second matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    
    Raises:
        RuntimeError: If Zig library is not available
        ValueError: If matrix dimensions are incompatible
    """
    if _lib is None:
        raise RuntimeError("Zig library not available")
    
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible dimensions: {a.shape} and {b.shape}")
    
    # Prepare inputs
    a_mat = Matrix.from_numpy(a)
    b_mat = Matrix.from_numpy(b)
    
    # Prepare output
    c = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    c_mat = Matrix.from_numpy(c)
    
    # Call Zig function
    result = _lib.matrix_multiply(
        ctypes.byref(a_mat),
        ctypes.byref(b_mat),
        ctypes.byref(c_mat),
    )
    
    if result != 0:
        raise RuntimeError(f"Matrix multiplication failed with code {result}")
    
    return c_mat.to_numpy()


def matrix_trace(m: np.ndarray) -> float:
    """
    Compute the trace of a matrix using the Zig implementation.
    
    Args:
        m: Square matrix
    
    Returns:
        Sum of diagonal elements
    
    Raises:
        RuntimeError: If Zig library is not available
        ValueError: If matrix is not square
    """
    if _lib is None:
        raise RuntimeError("Zig library not available")
    
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"Matrix must be square, got {m.shape}")
    
    m_mat = Matrix.from_numpy(m)
    return _lib.matrix_trace(ctypes.byref(m_mat))


def minkowski_metric_zig(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute Minkowski metric using the Zig implementation.
    
    Args:
        coordinates: Spacetime coordinates [t, x, y, z]
    
    Returns:
        4x4 Minkowski metric tensor with signature (-1, 1, 1, 1)
    
    Raises:
        RuntimeError: If Zig library is not available
    """
    if _lib is None:
        raise RuntimeError("Zig library not available")
    
    coords = coordinates.astype(np.float64)
    coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    metric = np.zeros((4, 4), dtype=np.float64)
    metric_mat = Matrix.from_numpy(metric)
    
    result = _lib.minkowski_metric(coords_ptr, ctypes.byref(metric_mat))
    
    if result != 0:
        raise RuntimeError(f"Minkowski metric computation failed with code {result}")
    
    return metric_mat.to_numpy()


def spherical_polar_metric_zig(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute spherical polar metric using the Zig implementation.
    
    Args:
        coordinates: Spherical coordinates [r, theta, phi]
    
    Returns:
        3x3 spherical polar metric tensor
    
    Raises:
        RuntimeError: If Zig library is not available
    """
    if _lib is None:
        raise RuntimeError("Zig library not available")
    
    coords = coordinates.astype(np.float64)
    coords_ptr = coords.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    metric = np.zeros((3, 3), dtype=np.float64)
    metric_mat = Matrix.from_numpy(metric)
    
    result = _lib.spherical_polar_metric(coords_ptr, ctypes.byref(metric_mat))
    
    if result != 0:
        raise RuntimeError(f"Spherical metric computation failed with code {result}")
    
    return metric_mat.to_numpy()


# =============================================================================
# Availability Check
# =============================================================================

def is_zig_available() -> bool:
    """Check if the Zig core library is available."""
    return _lib is not None


def get_library_path() -> Optional[str]:
    """Get the path to the loaded Zig library."""
    return _lib_path


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Matrix',
    'matrix_multiply',
    'matrix_trace',
    'minkowski_metric_zig',
    'spherical_polar_metric_zig',
    'is_zig_available',
    'get_library_path',
]
