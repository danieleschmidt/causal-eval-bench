FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        postgresql-client \
        redis-tools \
        curl \
        wget \
        git \
        vim \
        tmux \
        htop \
        tree \
        jq \
        zip \
        unzip \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip \
    && pip install poetry==1.6.1

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install Node.js tools
RUN npm install -g \
    markdownlint-cli \
    prettier \
    @commitlint/cli \
    @commitlint/config-conventional

# Install additional Python tools
RUN pip install \
    pre-commit \
    commitizen \
    detect-secrets

# Configure git (will be overridden by user settings)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Set up workspace
WORKDIR /workspaces/causal-eval-bench

# Copy and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev,test,docs

# Set up pre-commit hooks
COPY .pre-commit-config.yaml ./
RUN pre-commit install

# Create directories for development
RUN mkdir -p /tmp/test-results /tmp/coverage

# Set environment variables
ENV PYTHONPATH=/workspaces/causal-eval-bench
ENV ENVIRONMENT=development
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["sleep", "infinity"]
