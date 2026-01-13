# autograv - Quick Reference Card

## 🚀 Quick Start Commands

### Test the Library Locally
```bash
cd autograv

# Run examples
uv run examples/quick_start.py
uv run examples/schwarzschild_example.py
uv run examples/sphere_example.py
```

### Build the Package
```bash
cd autograv

# Clean previous builds
rm -rf dist/

# Build with uv
uv build --no-sources

# Verify build artifacts
ls dist/
# Should show: autograv-0.1.0.tar.gz and autograv-0.1.0-py3-none-any.whl
```

### Publish to TestPyPI (Recommended First Step)
```bash
# Set your TestPyPI token
$env:UV_PUBLISH_TOKEN="pypi-your-test-token-here"  # Windows PowerShell
# or
export UV_PUBLISH_TOKEN="pypi-your-test-token-here"  # Linux/macOS

# Publish
uv publish --publish-url https://test.pypi.org/legacy/

# Test installation
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ autograv

# Quick test
python -c "from autograv import christoffel_symbols; print('✓ Works!')"
```

### Publish to PyPI (Production)
```bash
# Set your PyPI token
$env:UV_PUBLISH_TOKEN="pypi-your-prod-token-here"  # Windows PowerShell
# or
export UV_PUBLISH_TOKEN="pypi-your-prod-token-here"  # Linux/macOS

# Publish (PERMANENT - cannot undo!)
uv publish

# Verify
pip install autograv
python -c "from autograv import christoffel_symbols; print('✓ Published!')"
```

### Set Up Git
```bash
cd autograv

# Run setup script
.\setup-git.ps1  # Windows PowerShell
# or
# bash setup-git.sh  # Linux/macOS (if you create a bash version)

# Follow the prompts to:
# - Initialize git
# - Create .gitignore
# - Make initial commit
# - Tag v0.1.0
```

### Push to GitHub
```bash
# Create repo at: https://github.com/new
# Then:

git remote add origin https://github.com/YOUR-USERNAME/autograv.git
git branch -M main
git push -u origin main
git push origin v0.1.0
```

### Publish via GitHub Actions (Automated)
```bash
# 1. Configure Trusted Publishing on PyPI:
#    https://pypi.org/manage/account/publishing/
#    - Project: autograv
#    - Owner: YOUR-USERNAME  
#    - Repo: autograv
#    - Workflow: publish.yml

# 2. Create GitHub Release:
#    https://github.com/YOUR-USERNAME/autograv/releases/new
#    - Tag: v0.1.0
#    - Title: Release 0.1.0
#    - Description: (copy from CHANGELOG.md)
#    - Click "Publish release"

# 3. Watch the magic happen:
#    https://github.com/YOUR-USERNAME/autograv/actions
```

---

## 📖 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | User documentation and API reference |
| `COMPLETE_SUMMARY.md` | This file - what was built and why |
| `PROJECT_STATUS.md` | Current status and next steps |
| `PUBLISHING.md` | Comprehensive publishing guide |
| `PUBLISHING_CHECKLIST.md` | Research findings and due diligence |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |
| `CHANGELOG.md` | Version history |

---

## 🔑 Key File Locations

| Component | Path |
|-----------|------|
| Main library | `src/autograv/__init__.py` |
| Package metadata | `pyproject.toml` |
| License | `LICENSE` |
| Examples | `examples/*.py` |
| PyPI workflow | `.github/workflows/publish.yml` |
| Test workflow | `.github/workflows/test.yml` |
| Git setup | `setup-git.ps1` |

---

## ⚡ Library Usage Examples

### Basic Usage
```python
import jax.numpy as jnp
from autograv import christoffel_symbols, spherical_polar_metric

# Define coordinates (r, theta)
coords = jnp.array([5.0, 1.0], dtype=jnp.float64)

# Compute Christoffel symbols
christ = christoffel_symbols(coords, spherical_polar_metric)

print(christ)
```

### Schwarzschild Black Hole
```python
import jax.numpy as jnp
from autograv import kretschmann_invariant

def schwarzschild_metric(coords):
    t, r, theta, phi = coords
    M = 2.0  # Schwarzschild radius
    
    g00 = -(1 - M/r)
    g11 = 1/(1 - M/r)
    g22 = r**2
    g33 = r**2 * jnp.sin(theta)**2
    
    return jnp.array([
        [g00, 0, 0, 0],
        [0, g11, 0, 0],
        [0, 0, g22, 0],
        [0, 0, 0, g33]
    ])

coords = jnp.array([0.0, 10.0, jnp.pi/2, 0.0])
K = kretschmann_invariant(coords, schwarzschild_metric)
print(f"Kretschmann invariant: {K}")
```

### Einstein Tensor
```python
from autograv import einstein_tensor

# For any metric function
G = einstein_tensor(coords, metric_fn)

# Should satisfy: G_ij = (8πG/c^4) T_ij
# (Einstein field equations)
```

---

## 🔗 Important URLs

| Resource | URL |
|----------|-----|
| Create PyPI account | https://pypi.org/account/register/ |
| Create TestPyPI account | https://test.pypi.org/account/register/ |
| PyPI API tokens | https://pypi.org/manage/account/ |
| TestPyPI API tokens | https://test.pypi.org/manage/account/ |
| Trusted Publishing setup | https://pypi.org/manage/account/publishing/ |
| Create GitHub repo | https://github.com/new |
| uv documentation | https://docs.astral.sh/uv/ |
| JAX documentation | https://jax.readthedocs.io/ |

---

## ✅ Pre-Publish Checklist

Before publishing to PyPI:

- [x] Code implemented and tested
- [x] Examples verified
- [x] Documentation complete
- [x] pyproject.toml metadata complete
- [x] LICENSE file exists
- [x] CHANGELOG.md updated
- [x] Package builds successfully
- [ ] PyPI account created
- [ ] API token generated (or Trusted Publishing configured)
- [ ] Tested on TestPyPI (recommended)
- [ ] GitHub repository created (optional but recommended)
- [ ] Git initialized and committed (optional but recommended)

---

## 🆘 Troubleshooting

### Build fails
```bash
# Update uv
uv self update

# Check dependencies
uv pip list

# Verify Python version
python --version  # Should be 3.10+
```

### Import fails after installation
```bash
# Check installation
pip show autograv

# Verify environment
python -c "import sys; print(sys.path)"

# Reinstall
pip uninstall autograv
pip install autograv
```

### JAX not available
```bash
# Windows: Use JAX 0.4.20
uv pip install jax==0.4.20 jaxlib==0.4.20

# Linux/macOS: Can use latest
uv pip install --upgrade jax jaxlib
```

### Publishing fails with 403
```bash
# Check token is set
echo $UV_PUBLISH_TOKEN  # Linux/macOS
echo $env:UV_PUBLISH_TOKEN  # Windows

# Regenerate token if expired
# Go to: https://pypi.org/manage/account/
```

---

## 📞 Getting Help

1. **Read the docs first:**
   - `PUBLISHING.md` for publishing issues
   - `PROJECT_STATUS.md` for status questions
   - `README.md` for usage questions

2. **Check troubleshooting:**
   - See PUBLISHING.md "Troubleshooting" section

3. **Resources:**
   - uv docs: https://docs.astral.sh/uv/
   - PyPI help: https://pypi.org/help/
   - Python packaging: https://packaging.python.org/

---

## 🎉 Success Indicators

You'll know it worked when:

✅ **Build succeeds:**
```
Successfully built dist/autograv-0.1.0.tar.gz
Successfully built dist/autograv-0.1.0-py3-none-any.whl
```

✅ **TestPyPI publish succeeds:**
```
Uploading autograv-0.1.0.tar.gz
Uploading autograv-0.1.0-py3-none-any.whl
```

✅ **Installation works:**
```bash
pip install autograv
# Collecting autograv
# Successfully installed autograv-0.1.0
```

✅ **Import works:**
```python
>>> from autograv import christoffel_symbols
>>> print("Success!")
Success!
```

✅ **Package visible on PyPI:**
- Visit: https://pypi.org/project/autograv/

---

**That's it! You're ready to publish autograv to PyPI! 🚀**

For detailed explanations, see `COMPLETE_SUMMARY.md` or `PUBLISHING.md`.
