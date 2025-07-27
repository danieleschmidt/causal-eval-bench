# Installation

This guide covers different ways to install and set up Causal Eval Bench for your environment.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM recommended
- **Storage**: 1GB free space for installation and data

## Installation Methods

### 1. PyPI Installation (Recommended)

The easiest way to install Causal Eval Bench is via pip:

```bash
pip install causal-eval-bench
```

For the latest development version:

```bash
pip install causal-eval-bench[all]
```

### 2. From Source

For development or to get the latest features:

```bash
git clone https://github.com/your-org/causal-eval-bench.git
cd causal-eval-bench
pip install -e ".[all]"
```

### 3. Docker Installation

For a containerized environment:

```bash
docker pull your-org/causal-eval:latest
docker run -it -p 8000:8000 your-org/causal-eval:latest
```

### 4. Conda Installation

If you prefer conda:

```bash
conda install -c conda-forge causal-eval-bench
```

## Verification

Verify your installation:

```bash
causal-eval --version
```

Or in Python:

```python
import causal_eval
print(causal_eval.__version__)
```

## API Keys Setup

### Required API Keys

Causal Eval Bench requires API keys for model providers. Set up at least one:

#### OpenAI
```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```

#### Anthropic  
```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
```

#### Google AI
```bash
export GOOGLE_AI_API_KEY="your-google-ai-api-key"
```

#### Hugging Face
```bash
export HUGGINGFACE_API_TOKEN="hf_your-huggingface-token"
```

### Environment File

Create a `.env` file in your project directory:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your API keys
nano .env
```

## Database Setup

### SQLite (Default)

No additional setup required. Causal Eval Bench will create a local SQLite database automatically.

### PostgreSQL (Production)

For production deployments, install PostgreSQL:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start the service
sudo service postgresql start  # Linux
brew services start postgresql  # macOS
```

Create a database:

```sql
CREATE DATABASE causal_eval_bench;
CREATE USER causal_eval_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE causal_eval_bench TO causal_eval_user;
```

Update your `.env` file:

```bash
DATABASE_URL=postgresql://causal_eval_user:your_password@localhost:5432/causal_eval_bench
```

## Redis Setup (Optional)

For caching and background tasks:

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS  
brew install redis

# Start Redis
sudo service redis-server start  # Linux
brew services start redis        # macOS
```

Update your `.env` file:

```bash
REDIS_URL=redis://localhost:6379/0
```

## Development Installation

For contributing to Causal Eval Bench:

```bash
# Clone the repository
git clone https://github.com/your-org/causal-eval-bench.git
cd causal-eval-bench

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev,test,docs

# Set up pre-commit hooks
poetry run pre-commit install

# Run tests to verify installation
poetry run pytest
```

## Docker Development Setup

For a complete development environment:

```bash
# Clone the repository
git clone https://github.com/your-org/causal-eval-bench.git
cd causal-eval-bench

# Start all services
docker-compose up --build

# The API will be available at http://localhost:8000
# Documentation at http://localhost:8080
```

## Troubleshooting

### Common Issues

#### Permission Errors

If you encounter permission errors:

```bash
pip install --user causal-eval-bench
```

#### SSL Certificate Errors

For corporate networks:

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org causal-eval-bench
```

#### Memory Issues

If you encounter memory issues during installation:

```bash
pip install --no-cache-dir causal-eval-bench
```

### Platform-Specific Issues

#### Windows

On Windows, you might need to install Visual C++ Build Tools:

1. Download from [Microsoft Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install the "C++ build tools" workload

#### macOS

On macOS, ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

#### Linux

On Ubuntu/Debian, install required system packages:

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

### Getting Help

If you encounter issues:

1. Check our [FAQ](../about/support.md#faq)
2. Search [GitHub Issues](https://github.com/your-org/causal-eval-bench/issues)
3. Join our [Discord community](https://discord.gg/causal-eval)
4. Email support: support@causal-eval-bench.org

## Next Steps

After installation:

1. Follow the [Quick Start Guide](quickstart.md)
2. Configure your [settings](configuration.md)
3. Run your [first evaluation](../user-guide/running-evaluations.md)

## Version Management

### Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade causal-eval-bench
```

### Version Pinning

For reproducible environments, pin the version:

```bash
pip install causal-eval-bench==0.1.0
```

### Development Versions

To install pre-release versions:

```bash
pip install --pre causal-eval-bench
```

## Optional Dependencies

### Performance Optimizations

For better performance:

```bash
pip install causal-eval-bench[performance]
```

This includes:
- `uvloop` for faster async operations
- `orjson` for faster JSON processing
- `cchardet` for faster encoding detection

### Machine Learning Extensions

For advanced ML features:

```bash
pip install causal-eval-bench[ml]
```

This includes:
- `torch` for PyTorch models
- `transformers` for Hugging Face models
- `sentence-transformers` for embeddings

### All Extensions

To install everything:

```bash
pip install causal-eval-bench[all]
```