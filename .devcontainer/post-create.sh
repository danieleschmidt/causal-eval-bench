#!/bin/bash

# Post-create script for dev container setup
set -e

echo "🚀 Starting post-create setup..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install --with dev,test,docs,lint

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env 2>/dev/null || cat > .env << 'EOF'
# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://causal_eval:dev_password@postgres:5432/causal_eval_dev

# Redis
REDIS_URL=redis://redis:6379/0

# API Keys (add your keys here)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Security
SECRET_KEY=dev-secret-key-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Monitoring
SENTRY_DSN=
PROMETHEUS_METRICS_ENABLED=true
EOF
fi

# Set up git configuration if not already set
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@causal-eval-bench.local"
fi

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p logs tmp/test-results tmp/coverage

# Set up development database
echo "🗄️  Setting up development database..."
# Wait for postgres to be ready
until pg_isready -h postgres -p 5432 -U causal_eval; do
    echo "Waiting for postgres..."
    sleep 2
done

# Run database migrations if they exist
if [ -f "alembic.ini" ]; then
    echo "🔄 Running database migrations..."
    poetry run alembic upgrade head
fi

# Install additional development tools
echo "🛠️  Installing additional development tools..."
pip install --user \
    httpie \
    ipython \
    jupyter \
    jupyterlab

# Set up shell aliases
echo "🐚 Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# Causal Eval Bench Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Project specific aliases
alias poetry-shell='poetry shell'
alias test='poetry run pytest'
alias test-watch='poetry run pytest-watch'
alias lint='poetry run ruff check .'
alias format='poetry run black . && poetry run isort .'
alias typecheck='poetry run mypy .'
alias serve='poetry run causal-eval-server'
alias worker='poetry run causal-eval-worker'
alias docs-serve='poetry run mkdocs serve'
alias docs-build='poetry run mkdocs build'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcb='docker-compose build'
alias dcs='docker-compose stop'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gb='git branch'
alias gco='git checkout'
EOF

# Set up fish shell configuration if fish is available
if command -v fish &> /dev/null; then
    echo "🐟 Setting up fish shell configuration..."
    mkdir -p ~/.config/fish
    cat >> ~/.config/fish/config.fish << 'EOF'
# Causal Eval Bench Development Configuration
set -x PYTHONPATH /workspaces/causal-eval-bench
set -x ENVIRONMENT development

# Aliases
abbr ll 'ls -alF'
abbr test 'poetry run pytest'
abbr lint 'poetry run ruff check .'
abbr format 'poetry run black . && poetry run isort .'
abbr serve 'poetry run causal-eval-server'
EOF
fi

echo "✅ Post-create setup completed!"
echo ""
echo "🎉 Welcome to Causal Eval Bench development environment!"
echo ""
echo "Quick start commands:"
echo "  poetry run causal-eval-server  # Start the API server"
echo "  poetry run pytest              # Run tests"
echo "  poetry run mkdocs serve        # Serve documentation"
echo "  make help                       # See all available commands"
echo ""
echo "Happy coding! 🚀"
