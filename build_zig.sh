#!/usr/bin/env bash
# Build script for autograv Zig core library

set -e  # Exit on error

echo "========================================"
echo "Building autograv Zig Core Library"
echo "========================================"
echo

# Check if Zig is installed
if ! command -v zig &> /dev/null; then
    echo "❌ Error: Zig is not installed"
    echo
    echo "Please install Zig from: https://ziglang.org/download/"
    echo
    echo "On Linux/macOS:"
    echo "  1. Download Zig tarball"
    echo "  2. Extract: tar -xf zig-*.tar.xz"
    echo "  3. Add to PATH: export PATH=\$PATH:/path/to/zig"
    echo
    echo "On macOS with Homebrew:"
    echo "  brew install zig"
    echo
    exit 1
fi

# Show Zig version
ZIG_VERSION=$(zig version)
echo "✓ Zig version: $ZIG_VERSION"
echo

# Parse command line arguments
BUILD_MODE="ReleaseFast"
RUN_TESTS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_MODE="Debug"
            shift
            ;;
        --release)
            BUILD_MODE="ReleaseFast"
            shift
            ;;
        --no-test)
            RUN_TESTS=0
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --debug      Build in debug mode (default: release)"
            echo "  --release    Build in release mode with optimizations"
            echo "  --no-test    Skip running tests after build"
            echo "  --help       Show this help message"
            echo
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the library
echo "Building Zig library in $BUILD_MODE mode..."
if [ "$BUILD_MODE" = "Debug" ]; then
    zig build -Doptimize=Debug
else
    zig build -Doptimize=ReleaseFast
fi
echo "✓ Build complete"
echo

# Run tests if requested
if [ $RUN_TESTS -eq 1 ]; then
    echo "Running Zig tests..."
    zig build test
    echo "✓ All tests passed"
    echo
fi

# Show build artifacts
echo "Build artifacts:"
echo "----------------"
if [ -d "zig-out/lib" ]; then
    ls -lh zig-out/lib/
    echo
    
    # Determine library file name based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        LIB_FILE="zig-out/lib/libautograv_core.so"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        LIB_FILE="zig-out/lib/libautograv_core.dylib"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        LIB_FILE="zig-out/lib/autograv_core.dll"
    else
        LIB_FILE=""
    fi
    
    if [ -n "$LIB_FILE" ] && [ -f "$LIB_FILE" ]; then
        SIZE=$(du -h "$LIB_FILE" | cut -f1)
        echo "✓ Library: $LIB_FILE ($SIZE)"
    fi
else
    echo "⚠ Warning: Build artifacts not found in zig-out/lib/"
fi

echo
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Test Python bindings: python examples/zig_ffi_example.py"
echo "2. See documentation: cat src/zig/README.md"
echo "3. View integration guide: cat ZIG_INTEGRATION.md"
echo
