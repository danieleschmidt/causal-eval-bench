#!/bin/bash
set -e

echo "🚀 Setting up advanced development environment..."

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="/home/vscode/.local/bin:$PATH"
fi

# Configure Poetry
echo "⚙️ Configuring Poetry..."
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false

# Install dependencies
echo "📚 Installing Python dependencies..."
poetry install --all-extras --sync

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Setup development environment
echo "🛠️ Setting up development tools..."

# Install additional development tools
pip install --upgrade \
    jupyterlab \
    jupyter-lab-git \
    jupyterlab-code-formatter \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyterlab-system-monitor \
    jupyterlab-git

# Install Node.js tools for modern development
npm install -g \
    @commitlint/cli \
    @commitlint/config-conventional \
    semantic-release \
    @semantic-release/changelog \
    @semantic-release/git \
    @semantic-release/github \
    prettier \
    markdownlint-cli

# Create Jupyter Lab configuration
mkdir -p /home/vscode/.jupyter
cat > /home/vscode/.jupyter/lab_user_settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings << EOF
{
    "theme": "JupyterLab Dark"
}
EOF

# Setup git configuration for development
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input
git config --global core.filemode false

# Create useful development aliases
cat >> /home/vscode/.bashrc << 'EOF'

# Causal Eval Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Project specific aliases
alias dev='make dev'
alias test='make test'
alias lint='make lint'
alias format='make format'
alias docs='make docs-serve'
alias build='make build'
alias clean='make clean'
alias notebook='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'

# Poetry aliases  
alias pi='poetry install'
alias pa='poetry add'
alias pr='poetry remove'
alias ps='poetry shell'
alias pu='poetry update'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcb='docker-compose build'
alias dcl='docker-compose logs'

# Python aliases
alias py='python'
alias ipy='ipython'
alias ptest='python -m pytest'
alias ptype='python -m mypy'
alias pformat='python -m black . && python -m isort .'

echo "🎯 Development environment ready!"
echo "🔧 Available commands: dev, test, lint, format, docs, build, clean, notebook"
echo "📝 Use 'notebook' to start Jupyter Lab on port 8888"
echo "🐛 Use 'make debug' for debugging setup"
EOF

# Create performance monitoring script
cat > /home/vscode/monitor.py << 'EOF'
#!/usr/bin/env python3
"""Development environment performance monitor."""

import psutil
import time
import json
from datetime import datetime

def monitor_system():
    while True:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
        
        print(f"🖥️  CPU: {stats['cpu_percent']:5.1f}% | "
              f"💾 RAM: {stats['memory_percent']:5.1f}% | "
              f"💿 Disk: {stats['disk_usage']:5.1f}%")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_system()
EOF

chmod +x /home/vscode/monitor.py

# Setup development database if needed
if [ -f "scripts/init-db.sql" ]; then
    echo "🗄️ Database initialization script found"
fi

# Create enhanced .env template if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug

# Database Configuration
DATABASE_URL=postgresql://causal_user:causal_pass@localhost:5432/causal_eval_dev
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET=dev-jwt-secret-change-in-production

# External APIs (add your keys)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_API_KEY=

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Development Tools
JUPYTER_PORT=8888
DOCS_PORT=8080
EOF
    echo "📝 Created .env template - please add your API keys"
fi

# Initialize git hooks if in a git repository
if [ -d ".git" ]; then
    echo "🔧 Setting up additional git hooks..."
    
    # Create commit message template
    cat > .gitmessage << 'EOF'
# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# Type: feat, fix, docs, style, refactor, test, chore
# Scope: component affected (optional)
# Subject: imperative, present tense, no period
# Body: explain what and why (optional)  
# Footer: breaking changes, issue references (optional)
EOF
    
    git config commit.template .gitmessage
fi

echo "✅ Advanced development environment setup complete!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  make dev        - Start development server"
echo "  make test       - Run test suite"
echo "  make docs       - Build and serve documentation"
echo "  notebook        - Start Jupyter Lab"
echo "  python monitor.py - Monitor development environment"
echo ""
echo "🔧 Tools Available:"
echo "  - Poetry for dependency management"
echo "  - Pre-commit hooks for code quality"
echo "  - Jupyter Lab for interactive development"  
echo "  - Semantic release for automated versioning"
echo "  - Advanced VS Code extensions"
echo "  - Performance monitoring tools"
echo ""
echo "🚀 Happy coding!"