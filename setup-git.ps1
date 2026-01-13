# Git Setup Script for autograv
# Run this script to initialize git and prepare for GitHub

Write-Host "🚀 autograv Git Setup Script" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Initialize git if not already initialized
if (-not (Test-Path .git)) {
    Write-Host "📦 Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git initialized`n" -ForegroundColor Green
} else {
    Write-Host "✅ Git already initialized`n" -ForegroundColor Green
}

# Create .gitignore if it doesn't exist
if (-not (Test-Path .gitignore)) {
    Write-Host "📝 Creating .gitignore..." -ForegroundColor Yellow
    
    $gitignore = @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# uv
.uv/
uv.lock

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build artifacts
*.whl
*.tar.gz
"@
    
    $gitignore | Out-File -FilePath .gitignore -Encoding utf8
    Write-Host "✅ .gitignore created`n" -ForegroundColor Green
} else {
    Write-Host "✅ .gitignore already exists`n" -ForegroundColor Green
}

# Configure git user if not set
$gitUserName = git config user.name
$gitUserEmail = git config user.email

if (-not $gitUserName) {
    Write-Host "⚠️  Git user name not configured globally" -ForegroundColor Yellow
    $userName = Read-Host "Enter your name (for git commits)"
    git config user.name "$userName"
}

if (-not $gitUserEmail) {
    Write-Host "⚠️  Git user email not configured globally" -ForegroundColor Yellow
    $userEmail = Read-Host "Enter your email (for git commits)"
    git config user.email "$userEmail"
}

Write-Host "`n📋 Current git configuration:" -ForegroundColor Cyan
Write-Host "   Name:  $(git config user.name)" -ForegroundColor White
Write-Host "   Email: $(git config user.email)`n" -ForegroundColor White

# Stage all files
Write-Host "📦 Staging files for commit..." -ForegroundColor Yellow
git add .

# Show status
Write-Host "`n📊 Git status:" -ForegroundColor Cyan
git status --short

# Prompt for initial commit
Write-Host "`n" -NoNewline
$commit = Read-Host "Create initial commit? (y/n)"

if ($commit -eq 'y' -or $commit -eq 'Y') {
    git commit -m "Initial commit: autograv v0.1.0

- Complete library implementation with 10 core GR tensor functions
- 2 built-in metrics (Minkowski, spherical polar)
- 3 example scripts with verified results
- Comprehensive documentation (README, CHANGELOG, guides)
- PyPI-ready metadata in pyproject.toml
- GitHub Actions workflows for CI/CD
- MIT licensed
"
    Write-Host "✅ Initial commit created`n" -ForegroundColor Green
    
    # Create v0.1.0 tag
    Write-Host "🏷️  Creating v0.1.0 tag..." -ForegroundColor Yellow
    git tag -a v0.1.0 -m "Release v0.1.0 - Initial release"
    Write-Host "✅ Tag v0.1.0 created`n" -ForegroundColor Green
}

# Instructions for GitHub
Write-Host "`n📝 Next Steps:" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "1. Create a new repository on GitHub:" -ForegroundColor Yellow
Write-Host "   https://github.com/new`n" -ForegroundColor White

Write-Host "2. Repository settings:" -ForegroundColor Yellow
Write-Host "   - Name: autograv" -ForegroundColor White
Write-Host "   - Description: Bridge numerical relativity and autodiff with JAX" -ForegroundColor White
Write-Host "   - Public repository" -ForegroundColor White
Write-Host "   - DON'T initialize with README (we have one)" -ForegroundColor White
Write-Host "   - DON'T add .gitignore (we have one)" -ForegroundColor White
Write-Host "   - DON'T add license (we have MIT)`n" -ForegroundColor White

Write-Host "3. Add remote and push:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/YOUR-USERNAME/autograv.git" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor White
Write-Host "   git push origin v0.1.0`n" -ForegroundColor White

Write-Host "4. Configure Trusted Publishing on PyPI:" -ForegroundColor Yellow
Write-Host "   - Go to: https://pypi.org/manage/account/publishing/" -ForegroundColor White
Write-Host "   - Click 'Add a new pending publisher'" -ForegroundColor White
Write-Host "   - PyPI Project Name: autograv" -ForegroundColor White
Write-Host "   - Owner: YOUR-USERNAME" -ForegroundColor White
Write-Host "   - Repository: autograv" -ForegroundColor White
Write-Host "   - Workflow: publish.yml" -ForegroundColor White
Write-Host "   - Environment: release (optional)`n" -ForegroundColor White

Write-Host "5. Create a GitHub Release to trigger publishing:" -ForegroundColor Yellow
Write-Host "   - Go to: https://github.com/YOUR-USERNAME/autograv/releases/new" -ForegroundColor White
Write-Host "   - Choose tag: v0.1.0" -ForegroundColor White
Write-Host "   - Title: Release 0.1.0" -ForegroundColor White
Write-Host "   - Copy description from CHANGELOG.md" -ForegroundColor White
Write-Host "   - Click 'Publish release'`n" -ForegroundColor White

Write-Host "6. Watch the GitHub Action publish to PyPI automatically! 🎉`n" -ForegroundColor Yellow

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`n" -ForegroundColor Cyan

Write-Host "📚 For more details, see:" -ForegroundColor Cyan
Write-Host "   - PUBLISHING.md (comprehensive publishing guide)" -ForegroundColor White
Write-Host "   - PROJECT_STATUS.md (current status and options)" -ForegroundColor White
Write-Host "   - README.md (user documentation)`n" -ForegroundColor White

Write-Host "✨ Git setup complete! Ready for GitHub.`n" -ForegroundColor Green
