# PyPI Publishing Checklist for autograv

## Research Summary

Based on research of modern Python packaging best practices using Astral's `uv` tool and PyPI guidelines:

### Tools & Documentation Reviewed
- ✅ Astral uv documentation (https://docs.astral.sh/uv/guides/package/)
- ✅ Python Packaging Authority guides
- ✅ PEP 639 (License metadata)
- ✅ PEP 423 (Naming conventions)
- ✅ PyPI Trusted Publishing workflow

### Key Findings
1. **uv 0.9.17** supports `uv build` and `uv publish` commands
2. **Package name "autograv"** appears available on PyPI (verified)
3. **Trusted Publishing** from GitHub Actions is recommended (no API tokens needed)
4. **Modern pyproject.toml** format is fully supported by uv-build
5. **PEP 639** license format should be used

---

## Pre-Publishing Checklist

### ☑ 1. Package Metadata (pyproject.toml)

#### Required Fields
- [x] `name` - "autograv" (verified available)
- [x] `version` - "0.1.0" (using semantic versioning)
- [x] `description` - One-line summary present
- [x] `readme` - README.md exists and is comprehensive
- [x] `requires-python` - ">=3.11" (appropriate for JAX)
- [x] `dependencies` - Core deps listed (numpy, jax, jaxlib)
- [x] `authors` - Author information present
- [x] `build-system` - uv_build configured

#### Missing/Needs Enhancement
- [ ] **LICENSE** - Need to add MIT license file
- [ ] **license** field - Need to add SPDX identifier ("MIT")
- [ ] **license-files** - Need to reference LICENSE file
- [ ] **keywords** - Add search keywords for PyPI
- [ ] **classifiers** - Add PyPI trove classifiers
- [ ] **urls** - Add project URLs (Repository, Issues, etc.)
- [ ] **optional-dependencies** - Consider adding dev/test extras

### ☑ 2. Code Quality

- [x] Type hints present
- [x] Docstrings comprehensive
- [x] Examples working (verified)
- [ ] **Tests** - Consider adding unit tests
- [ ] **CI/CD** - Set up GitHub Actions for testing

### ☑ 3. Documentation

- [x] README.md comprehensive with:
  - [x] Project description
  - [x] Installation instructions
  - [x] Quick start guide
  - [x] API reference
  - [x] Examples
- [x] IMPLEMENTATION_SUMMARY.md for context
- [ ] **CHANGELOG.md** - Create for version history
- [ ] **CONTRIBUTING.md** - Add if accepting contributions

### ☑ 4. Legal/Licensing

- [ ] **LICENSE file** - Create MIT license
- [ ] **Copyright notices** - Ensure proper attribution
- [ ] **Third-party licenses** - JAX, NumPy are compatible (all permissive)

### ☑ 5. Build System

- [x] `pyproject.toml` uses modern format
- [x] `uv_build` backend configured
- [ ] **Test build locally** - Run `uv build`
- [ ] **Verify build outputs** - Check dist/ directory
- [ ] **Test installation** - Try installing from built wheel

### ☑ 6. Version Management

- [x] Current version: 0.1.0
- [ ] **Version strategy** - Document versioning approach
- [ ] **Git tags** - Tag releases in git

### ☑ 7. Platform Compatibility

**Current Status:**
- ✅ Windows (tested with Python 3.12, JAX 0.4.20)
- ⚠️ Linux/macOS - Should work but not tested
- ⚠️ JAX compatibility - CPU-only on Windows, GPU on Linux/macOS

**Considerations:**
- NumPy pinned to <2 for JAX 0.4.20 compatibility
- jaxlib 0.4.20 is last version supporting Windows
- Python 3.11+ required (lower bound could be 3.9 for JAX)

### ☑ 8. PyPI Publishing Options

#### Option A: Manual Publishing (Simple, for testing)
```bash
# 1. Build package
uv build

# 2. Publish to TestPyPI first
uv publish --index testpypi --token <token>

# 3. Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ autograv

# 4. Publish to PyPI
uv publish --token <token>
```

#### Option B: Trusted Publishing via GitHub Actions (Recommended)
- No API tokens needed
- Secure and automated
- Publish on git tag/release
- Requires GitHub repository setup

---

## Action Items (Priority Order)

### 🔴 Critical (Must Have Before Publishing)

1. **Add LICENSE file**
   - Create MIT license file
   - Update pyproject.toml with license metadata

2. **Enhance pyproject.toml**
   - Add classifiers
   - Add keywords
   - Add project URLs
   - Add license field

3. **Test build process**
   - Run `uv build`
   - Verify artifacts in dist/
   - Test installation from wheel

4. **Create CHANGELOG.md**
   - Document 0.1.0 release notes

### 🟡 Important (Should Have)

5. **Test on TestPyPI**
   - Register on test.pypi.org
   - Publish test version
   - Verify installation works

6. **Add unit tests**
   - Test core functions
   - Verify examples work
   - Set up pytest

7. **Set up GitHub repository**
   - Push code to GitHub
   - Configure GitHub Actions
   - Set up Trusted Publishing

### 🟢 Nice to Have (Future)

8. **Documentation site**
   - Consider ReadTheDocs
   - API documentation
   - Tutorials

9. **CI/CD pipeline**
   - Automated testing
   - Automated publishing
   - Coverage reports

10. **Additional examples**
    - More metrics (Kerr, FRW, etc.)
    - Jupyter notebooks
    - Visualization examples

---

## Publishing Workflow (Recommended)

### Step 1: Local Preparation
```bash
# Update version
uv version 0.1.0

# Build package
uv build --no-sources

# Verify build
ls dist/
# Should see: autograv-0.1.0.tar.gz and autograv-0.1.0-py3-none-any.whl
```

### Step 2: TestPyPI (Dry Run)
```bash
# Publish to TestPyPI
uv publish --index testpypi --token <test-pypi-token>

# Test installation
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    autograv

# Verify it works
python -c "from autograv import christoffel_symbols; print('Success!')"
```

### Step 3: Production PyPI
```bash
# Publish to PyPI (FINAL STEP - Cannot undo!)
uv publish --token <pypi-token>

# Verify on PyPI
# Visit: https://pypi.org/project/autograv/

# Test installation
pip install autograv
```

### Step 4: Post-Publication
- Tag release in git: `git tag v0.1.0 && git push --tags`
- Create GitHub release with notes
- Announce on relevant forums/communities
- Update project status in README

---

## Security Considerations

1. **Never commit tokens** - Use environment variables or GitHub secrets
2. **Use Trusted Publishing** - Preferred over API tokens
3. **Verify builds** - Check dist/ contents before publishing
4. **Test thoroughly** - Always use TestPyPI first
5. **Version immutability** - Once published, versions cannot be modified

---

## Package Management After Publishing

Users will be able to install with:
```bash
# Using pip
pip install autograv

# Using uv
uv add autograv

# Using conda (after conda-forge submission)
conda install -c conda-forge autograv

# Using pixi
pixi add autograv
```

---

## References

- uv Publishing Guide: https://docs.astral.sh/uv/guides/package/
- PyPA Packaging Guide: https://packaging.python.org/
- PEP 639 (Licensing): https://peps.python.org/pep-0639/
- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- Semantic Versioning: https://semver.org/
