# Publishing Guide for autograv

This guide provides step-by-step instructions for publishing autograv to PyPI.

## Prerequisites

- [x] Package built successfully (`uv build` completed)
- [x] All metadata in `pyproject.toml` is complete
- [x] LICENSE file exists
- [x] README.md is comprehensive
- [x] CHANGELOG.md is up to date
- [ ] GitHub repository created
- [ ] Tests are passing (if tests exist)

## Option 1: Manual Publishing (Quickstart)

### Step 1: Create PyPI Account

1. Go to [PyPI](https://pypi.org/account/register/)
2. Register an account
3. Verify your email

### Step 2: Create TestPyPI Account (Recommended for testing)

1. Go to [TestPyPI](https://test.pypi.org/account/register/)
2. Register an account (separate from PyPI)
3. Verify your email

### Step 3: Generate API Token

**For TestPyPI (recommended first):**
1. Go to https://test.pypi.org/manage/account/
2. Scroll to "API tokens"
3. Click "Add API token"
4. Give it a name: "autograv-test"
5. Scope: "Entire account" (or specific project after first upload)
6. Copy the token (starts with `pypi-`)

**For PyPI (production):**
1. Go to https://pypi.org/manage/account/
2. Follow same steps as above
3. Name it "autograv-prod"

### Step 4: Build the Package

```bash
cd autograv

# Clean previous builds
rm -rf dist/

# Build with uv (recommended)
uv build --no-sources

# Verify artifacts
ls dist/
# Should show:
#   autograv-0.1.0.tar.gz
#   autograv-0.1.0-py3-none-any.whl
```

### Step 5: Publish to TestPyPI (Dry Run)

```bash
# Set token as environment variable (Linux/macOS)
export UV_PUBLISH_TOKEN="pypi-your-test-token-here"

# Or on Windows PowerShell
$env:UV_PUBLISH_TOKEN="pypi-your-test-token-here"

# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Or specify token directly (not recommended)
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-your-test-token
```

### Step 6: Test Installation from TestPyPI

```bash
# Create a fresh environment
uv venv test-env
source test-env/bin/activate  # Linux/macOS
# or
test-env\Scripts\activate      # Windows

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    autograv

# Test import
python -c "from autograv import christoffel_symbols; print('✓ Import successful!')"

# Run a quick example
python -c "
import jax.numpy as jnp
from autograv import christoffel_symbols, spherical_polar_metric
coords = jnp.array([5.0, 1.0, 1.0], dtype=jnp.float64)
result = christoffel_symbols(coords, spherical_polar_metric)
print('✓ Functions work correctly!')
"

# Clean up
deactivate
rm -rf test-env
```

### Step 7: Publish to PyPI (Production)

**⚠️ WARNING: This step is PERMANENT! You cannot delete or modify versions once published.**

```bash
# Set production token
export UV_PUBLISH_TOKEN="pypi-your-prod-token-here"
# or
$env:UV_PUBLISH_TOKEN="pypi-your-prod-token-here"

# Publish to PyPI
uv publish

# The package will be available at: https://pypi.org/project/autograv/
```

### Step 8: Verify Production Installation

```bash
# Install from PyPI
pip install autograv

# Test it works
python -c "from autograv import christoffel_symbols; print('✓ Published successfully!')"
```

### Step 9: Tag the Release

```bash
# Create git tag
git tag v0.1.0
git push origin v0.1.0

# Or create a GitHub release through the web interface
```

---

## Option 2: Automated Publishing with GitHub Actions (Recommended)

This method uses PyPI's **Trusted Publishing** feature - no API tokens needed!

### Step 1: Create GitHub Repository

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: autograv v0.1.0"

# Create repo on GitHub and push
git remote add origin https://github.com/bkataru/autograv.git
git branch -M main
git push -u origin main
```

### Step 2: Configure Trusted Publishing on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `autograv`
   - **Owner**: `bkataru` (your GitHub username)
   - **Repository name**: `autograv`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `release` (optional but recommended)
4. Click "Add"

### Step 3: Create GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]
  workflow_dispatch:  # Allow manual trigger

jobs:
  build:
    name: Build distribution
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - name: Build package
        run: uv build --no-sources
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    name: Publish to PyPI
    needs: build
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write  # Required for trusted publishing
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
      
      - name: Publish to PyPI
        run: uv publish
```

### Step 4: Create a Release

```bash
# Tag the version
git tag v0.1.0
git push origin v0.1.0

# Then create a GitHub Release through the web interface:
# 1. Go to https://github.com/bkataru/autograv/releases/new
# 2. Choose tag: v0.1.0
# 3. Title: "Release 0.1.0"
# 4. Description: Copy from CHANGELOG.md
# 5. Click "Publish release"
```

The GitHub Action will automatically build and publish to PyPI!

---

## Option 3: Publishing to TestPyPI with GitHub Actions

For testing the automated workflow before going to production:

Create `.github/workflows/test-publish.yml`:

```yaml
name: Publish to TestPyPI

on:
  push:
    tags:
      - 'test-v*'  # Trigger on test tags like test-v0.1.0

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      
      - name: Build package
        run: uv build --no-sources
      
      - name: Publish to TestPyPI
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.TEST_PYPI_TOKEN }}
        run: uv publish --publish-url https://test.pypi.org/legacy/
```

Then add your TestPyPI token to GitHub Secrets:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `TEST_PYPI_TOKEN`
4. Value: Your TestPyPI token
5. Click "Add secret"

Test with:
```bash
git tag test-v0.1.0
git push origin test-v0.1.0
```

---

## Post-Publication Checklist

After successfully publishing:

- [ ] Verify package appears on PyPI: https://pypi.org/project/autograv/
- [ ] Test installation: `pip install autograv`
- [ ] Update project status badge in README if desired
- [ ] Announce on relevant platforms:
  - [ ] Python subreddit
  - [ ] Physics computing forums
  - [ ] Twitter/X with #python #physics hashtags
  - [ ] LinkedIn
- [ ] Consider submitting to:
  - [ ] conda-forge (for conda users)
  - [ ] Awesome Python lists
  - [ ] Scientific Python ecosystem
- [ ] Monitor GitHub issues for feedback
- [ ] Star your own repo to celebrate! 🎉

---

## Troubleshooting

### Build Failures

**Error: "No module named 'uv_build'"**
- Update uv: `uv self update`
- Check pyproject.toml has correct build-system

**Error: "Failed to build autograv"**
- Check all imports work: `python -c "import autograv"`
- Verify __init__.py exports are correct
- Check for syntax errors

### Publishing Failures

**Error: "403 Forbidden"**
- Check your API token is correct
- Verify token has correct permissions
- For Trusted Publishing, ensure GitHub Actions has id-token: write permission

**Error: "400 File already exists"**
- Version already published to PyPI
- Increment version in pyproject.toml
- Rebuild and republish

**Error: "Package name already taken"**
- Someone else registered the name
- Choose a different name (e.g., `autograv-py`, `py-autograv`)
- Consider reaching out to current owner if package is abandoned

### Installation Issues

**NumPy version conflicts**
- This is expected - autograv requires numpy<2 for JAX compatibility
- Ensure you're using compatible versions

**JAX not available on Windows**
- Use jaxlib 0.4.20 (last Windows-compatible version)
- Or use WSL2/Linux for newer JAX versions

---

## Version Management

For future releases:

```bash
# Bump version
uv version --bump minor    # 0.1.0 → 0.2.0
uv version --bump patch    # 0.1.0 → 0.1.1
uv version 1.0.0          # Set specific version

# Update CHANGELOG.md

# Rebuild and republish
uv build --no-sources
uv publish

# Tag the release
git tag v0.2.0
git push origin v0.2.0
```

---

## Useful Commands

```bash
# Check package info
uv version

# Validate pyproject.toml
uv build --check

# List what will be included in package
tar -tzf dist/autograv-0.1.0.tar.gz

# Extract and inspect wheel contents
unzip -l dist/autograv-0.1.0-py3-none-any.whl

# Check package metadata
pip show autograv
```

---

## Resources

- uv Publishing Guide: https://docs.astral.sh/uv/guides/package/
- PyPI Help: https://pypi.org/help/
- Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- Python Packaging Guide: https://packaging.python.org/
- Semantic Versioning: https://semver.org/
