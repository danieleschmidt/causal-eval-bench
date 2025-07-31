#!/usr/bin/env python3
"""
Development Environment Setup Script
Automated setup and validation for the Causal Eval Bench development environment.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import json
import platform


class DevSetup:
    """Development environment setup and validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
    def run_command(self, command: List[str], description: str, check: bool = True) -> Tuple[bool, str, str]:
        """Run a command with error handling."""
        print(f"🔧 {description}...")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                check=check
            )
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr
        except Exception as e:
            return False, "", str(e)
    
    def check_python_version(self) -> bool:
        """Check if Python version is supported."""
        print(f"🐍 Checking Python version: {self.python_version}")
        
        supported_versions = ["3.9", "3.10", "3.11", "3.12"]
        if self.python_version not in supported_versions:
            print(f"❌ Python {self.python_version} is not supported")
            print(f"   Supported versions: {', '.join(supported_versions)}")
            return False
        
        print(f"✅ Python {self.python_version} is supported")
        return True
    
    def check_poetry(self) -> bool:
        """Check if Poetry is installed and install if needed."""
        print("📦 Checking Poetry installation...")
        
        if shutil.which("poetry"):
            success, stdout, stderr = self.run_command(
                ["poetry", "--version"],
                "Getting Poetry version",
                check=False
            )
            if success:
                print(f"✅ Poetry is installed: {stdout.strip()}")
                return True
        
        print("❌ Poetry not found. Please install Poetry first:")
        print("   curl -sSL https://install.python-poetry.org | python3 -")
        print("   or visit: https://python-poetry.org/docs/#installation")
        return False
    
    def check_docker(self) -> bool:
        """Check if Docker is installed and running."""
        print("🐳 Checking Docker installation...")
        
        if not shutil.which("docker"):
            print("❌ Docker not found. Please install Docker:")
            if self.system == "darwin":
                print("   https://docs.docker.com/desktop/mac/install/")
            elif self.system == "linux":
                print("   https://docs.docker.com/engine/install/")
            elif self.system == "windows":
                print("   https://docs.docker.com/desktop/windows/install/")
            return False
        
        # Check if Docker daemon is running
        success, stdout, stderr = self.run_command(
            ["docker", "info"],
            "Checking Docker daemon",
            check=False
        )
        
        if not success:
            print("❌ Docker daemon is not running. Please start Docker.")
            return False
        
        print("✅ Docker is installed and running")
        return True
    
    def check_docker_compose(self) -> bool:
        """Check if Docker Compose is available."""
        print("🐳 Checking Docker Compose...")
        
        # Try docker-compose command first
        if shutil.which("docker-compose"):
            success, stdout, stderr = self.run_command(
                ["docker-compose", "--version"],
                "Getting Docker Compose version",
                check=False
            )
            if success:
                print(f"✅ Docker Compose (standalone): {stdout.strip()}")
                return True
        
        # Try docker compose (plugin)
        success, stdout, stderr = self.run_command(
            ["docker", "compose", "version"],
            "Getting Docker Compose plugin version",
            check=False
        )
        
        if success:
            print(f"✅ Docker Compose (plugin): {stdout.strip()}")
            return True
        
        print("❌ Docker Compose not found. Please install Docker Compose:")
        print("   https://docs.docker.com/compose/install/")
        return False
    
    def setup_poetry_environment(self) -> bool:
        """Setup Poetry virtual environment and install dependencies."""
        print("📦 Setting up Poetry environment...")
        
        # Configure Poetry to create venv in project
        success, _, _ = self.run_command(
            ["poetry", "config", "virtualenvs.in-project", "true"],
            "Configuring Poetry virtual environment location",
            check=False
        )
        
        # Install dependencies
        success, stdout, stderr = self.run_command(
            ["poetry", "install", "--with", "dev,test,docs"],
            "Installing project dependencies"
        )
        
        if not success:
            print("❌ Failed to install dependencies:")
            print(stderr)
            return False
        
        print("✅ Dependencies installed successfully")
        return True
    
    def setup_pre_commit_hooks(self) -> bool:
        """Setup pre-commit hooks."""
        print("🪝 Setting up pre-commit hooks...")
        
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "pre-commit", "install"],
            "Installing pre-commit hooks"
        )
        
        if not success:
            print("❌ Failed to install pre-commit hooks:")
            print(stderr)
            return False
        
        # Also install commit-msg hooks
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "pre-commit", "install", "--hook-type", "commit-msg"],
            "Installing commit-msg hooks",
            check=False
        )
        
        print("✅ Pre-commit hooks installed")
        return True
    
    def create_env_file(self) -> bool:
        """Create .env file from template if it doesn't exist."""
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists():
            print("✅ .env file already exists")
            return True
        
        env_content = """# Causal Eval Bench Environment Configuration
# Copy this file to .env and customize for your environment

# Database Configuration
DATABASE_URL=postgresql://causal_eval_user:causal_eval_pass@localhost:5432/causal_eval_bench
TEST_DATABASE_URL=sqlite:///./test_causal_eval_bench.db

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_SECRET_KEY=your-secret-key-here-change-in-production
API_DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# Model API Keys (for testing - add your own keys)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Monitoring
SENTRY_DSN=your-sentry-dsn-here
PROMETHEUS_ENABLED=true

# Development Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
"""
        
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✅ Created .env file with default configuration")
            print("   💡 Please edit .env file to add your API keys and customize settings")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def validate_installation(self) -> bool:
        """Validate the installation by running basic checks."""
        print("🔍 Validating installation...")
        
        # Check if we can import the project (if it exists)
        if (self.project_root / "causal_eval").exists():
            success, stdout, stderr = self.run_command(
                ["poetry", "run", "python", "-c", "import causal_eval; print('✅ Import successful')"],
                "Testing package import",
                check=False
            )
            if not success:
                print("⚠️  Package import failed (this is expected if core package isn't implemented yet)")
        
        # Run a quick test to ensure the environment works
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "python", "--version"],
            "Testing Poetry environment"
        )
        
        if not success:
            print("❌ Poetry environment validation failed")
            return False
        
        print("✅ Installation validation completed")
        return True
    
    def setup_vscode_settings(self) -> bool:
        """Create VS Code settings for the project."""
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)
        
        settings_file = vscode_dir / "settings.json"
        if settings_file.exists():
            print("✅ VS Code settings already exist")
            return True
        
        settings = {
            "python.defaultInterpreterPath": "./.venv/bin/python",
            "python.terminal.activateEnvironment": True,
            "python.linting.enabled": True,
            "python.linting.ruffEnabled": True,
            "python.linting.mypyEnabled": True,
            "python.linting.banditEnabled": True,
            "python.formatting.provider": "black",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests/"],
            "files.exclude": {
                "**/__pycache__": True,
                "**/.pytest_cache": True,
                "**/.mypy_cache": True,
                "**/.ruff_cache": True,
                "**/htmlcov": True
            },
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {
                "source.organizeImports": True
            }
        }
        
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print("✅ Created VS Code settings")
            return True
        except Exception as e:
            print(f"❌ Failed to create VS Code settings: {e}")
            return False
    
    def run_setup(self, skip_docker: bool = False) -> bool:
        """Run the complete development setup."""
        print("🚀 Starting development environment setup...")
        print("=" * 60)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Poetry Installation", self.check_poetry),
        ]
        
        if not skip_docker:
            checks.extend([
                ("Docker Installation", self.check_docker),
                ("Docker Compose", self.check_docker_compose),
            ])
        
        setup_steps = [
            ("Poetry Environment", self.setup_poetry_environment),
            ("Pre-commit Hooks", self.setup_pre_commit_hooks),
            ("Environment File", self.create_env_file),
            ("VS Code Settings", self.setup_vscode_settings),
            ("Installation Validation", self.validate_installation),
        ]
        
        # Run prerequisite checks
        print("📋 Checking prerequisites...")
        for name, check_func in checks:
            if not check_func():
                print(f"❌ Setup failed at: {name}")
                return False
        
        print("\n🔧 Running setup steps...")
        # Run setup steps
        for name, setup_func in setup_steps:
            if not setup_func():
                print(f"❌ Setup failed at: {name}")
                return False
        
        print("=" * 60)
        print("🎉 Development environment setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Edit .env file to add your API keys")
        print("   2. Run 'make run' to start the development services")
        print("   3. Run 'make test' to verify everything works")
        print("   4. Check out the documentation: make docs")
        print("\n💡 Useful commands:")
        print("   make help      - Show all available commands")
        print("   make lint      - Run code quality checks")  
        print("   make test      - Run the test suite")
        print("   make format    - Format code")
        
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup development environment")
    parser.add_argument("--skip-docker", action="store_true", 
                       help="Skip Docker-related checks")
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Run setup
    setup = DevSetup(project_root)
    success = setup.run_setup(skip_docker=args.skip_docker)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()