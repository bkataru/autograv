# Build script for autograv Zig core library (Windows PowerShell)

Write-Host "========================================"
Write-Host "Building autograv Zig Core Library"
Write-Host "========================================"
Write-Host ""

# Check if Zig is installed
$zigPath = Get-Command zig -ErrorAction SilentlyContinue
if (-not $zigPath) {
    Write-Host "❌ Error: Zig is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Zig from: https://ziglang.org/download/"
    Write-Host ""
    Write-Host "Installation steps:"
    Write-Host "  1. Download Zig for Windows"
    Write-Host "  2. Extract the archive"
    Write-Host "  3. Add the zig directory to your PATH"
    Write-Host ""
    exit 1
}

# Show Zig version
$zigVersion = zig version
Write-Host "✓ Zig version: $zigVersion" -ForegroundColor Green
Write-Host ""

# Parse command line arguments
$buildMode = "ReleaseFast"
$runTests = $true

foreach ($arg in $args) {
    switch ($arg) {
        "--debug" {
            $buildMode = "Debug"
        }
        "--release" {
            $buildMode = "ReleaseFast"
        }
        "--no-test" {
            $runTests = $false
        }
        "--help" {
            Write-Host "Usage: .\build_zig.ps1 [OPTIONS]"
            Write-Host ""
            Write-Host "Options:"
            Write-Host "  --debug      Build in debug mode (default: release)"
            Write-Host "  --release    Build in release mode with optimizations"
            Write-Host "  --no-test    Skip running tests after build"
            Write-Host "  --help       Show this help message"
            Write-Host ""
            exit 0
        }
        default {
            Write-Host "Unknown option: $arg" -ForegroundColor Yellow
            Write-Host "Use --help for usage information"
            exit 1
        }
    }
}

# Build the library
Write-Host "Building Zig library in $buildMode mode..."
if ($buildMode -eq "Debug") {
    zig build -Doptimize=Debug
} else {
    zig build -Doptimize=ReleaseFast
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "✓ Build complete" -ForegroundColor Green
Write-Host ""

# Run tests if requested
if ($runTests) {
    Write-Host "Running Zig tests..."
    zig build test
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ All tests passed" -ForegroundColor Green
    } else {
        Write-Host "❌ Some tests failed" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ""
}

# Show build artifacts
Write-Host "Build artifacts:"
Write-Host "----------------"
if (Test-Path "zig-out\lib") {
    Get-ChildItem "zig-out\lib" | Format-Table Name, Length, LastWriteTime
    Write-Host ""
    
    $libFile = "zig-out\lib\autograv_core.dll"
    if (Test-Path $libFile) {
        $size = (Get-Item $libFile).Length
        $sizeKB = [math]::Round($size / 1KB, 2)
        Write-Host "✓ Library: $libFile ($sizeKB KB)" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ Warning: Build artifacts not found in zig-out\lib\" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================"
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Test Python bindings: python examples\zig_ffi_example.py"
Write-Host "2. See documentation: type src\zig\README.md"
Write-Host "3. View integration guide: type ZIG_INTEGRATION.md"
Write-Host ""
