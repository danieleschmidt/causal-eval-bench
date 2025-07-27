#!/usr/bin/env python3
"""Build automation script for Causal Eval Bench."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    click.echo(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            click.echo(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running command: {e}", err=True)
        if e.stderr:
            click.echo(f"Error output: {e.stderr}", err=True)
        if check:
            sys.exit(1)
        return e


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Build automation for Causal Eval Bench."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    if verbose:
        click.echo("Verbose mode enabled")


@cli.command()
@click.option('--target', '-t', default='runtime', help='Build target (runtime, development, production, testing)')
@click.option('--tag', default=None, help='Docker image tag')
@click.option('--platform', default='linux/amd64', help='Target platform')
@click.option('--push', is_flag=True, help='Push image after build')
@click.option('--cache', is_flag=True, default=True, help='Use build cache')
@click.pass_context
def docker(ctx, target, tag, platform, push, cache):
    """Build Docker image."""
    verbose = ctx.obj['verbose']
    
    if not tag:
        # Generate tag from git commit or current version
        try:
            git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
            tag = f"causal-eval-bench:{target}-{git_sha}"
        except subprocess.CalledProcessError:
            tag = f"causal-eval-bench:{target}-latest"
    
    build_cmd = [
        'docker', 'build',
        '--target', target,
        '--platform', platform,
        '--tag', tag,
    ]
    
    if not cache:
        build_cmd.append('--no-cache')
    
    # Add build args
    build_cmd.extend([
        '--build-arg', f'GIT_COMMIT={subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()}',
        '--build-arg', f'BUILD_DATE={subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip()}',
    ])
    
    build_cmd.append('.')
    
    click.echo(f"Building Docker image: {tag}")
    result = run_command(build_cmd)
    
    if result.returncode == 0:
        click.echo(f"✅ Successfully built {tag}")
        
        if push:
            click.echo(f"Pushing {tag}...")
            push_result = run_command(['docker', 'push', tag])
            if push_result.returncode == 0:
                click.echo(f"✅ Successfully pushed {tag}")
    else:
        click.echo(f"❌ Build failed for {tag}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--format', type=click.Choice(['wheel', 'sdist', 'both']), default='both')
@click.option('--output-dir', '-o', default='dist', help='Output directory')
@click.pass_context
def package(ctx, format, output_dir):
    """Build Python package."""
    verbose = ctx.obj['verbose']
    
    # Clean previous builds
    click.echo("Cleaning previous builds...")
    run_command(['rm', '-rf', 'dist/', 'build/', '*.egg-info'], check=False)
    
    # Build package
    click.echo("Building Python package...")
    
    if format in ['wheel', 'both']:
        click.echo("Building wheel...")
        run_command(['poetry', 'build', '--format', 'wheel'])
    
    if format in ['sdist', 'both']:
        click.echo("Building source distribution...")
        run_command(['poetry', 'build', '--format', 'sdist'])
    
    # Move to output directory if different
    if output_dir != 'dist':
        os.makedirs(output_dir, exist_ok=True)
        run_command(['mv', 'dist/*', output_dir], check=False)
    
    click.echo(f"✅ Package built successfully in {output_dir}")


@cli.command()
@click.option('--coverage', is_flag=True, help='Run with coverage')
@click.option('--parallel', is_flag=True, help='Run tests in parallel')
@click.option('--markers', '-m', help='Run tests with specific markers')
@click.pass_context
def test(ctx, coverage, parallel, markers):
    """Run tests."""
    verbose = ctx.obj['verbose']
    
    test_cmd = ['poetry', 'run', 'pytest']
    
    if coverage:
        test_cmd.extend(['--cov=causal_eval', '--cov-report=html', '--cov-report=xml'])
    
    if parallel:
        test_cmd.extend(['-n', 'auto'])
    
    if markers:
        test_cmd.extend(['-m', markers])
    
    if verbose:
        test_cmd.append('-v')
    
    test_cmd.append('tests/')
    
    click.echo("Running tests...")
    result = run_command(test_cmd)
    
    if result.returncode == 0:
        click.echo("✅ All tests passed")
    else:
        click.echo("❌ Some tests failed", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def lint(ctx):
    """Run code quality checks."""
    verbose = ctx.obj['verbose']
    
    checks = [
        (['poetry', 'run', 'ruff', 'check', '.'], 'Ruff linting'),
        (['poetry', 'run', 'black', '--check', '.'], 'Black formatting'),
        (['poetry', 'run', 'isort', '--check-only', '.'], 'Import sorting'),
        (['poetry', 'run', 'mypy', '.'], 'Type checking'),
        (['poetry', 'run', 'bandit', '-r', '.'], 'Security scanning'),
    ]
    
    failed_checks = []
    
    for cmd, description in checks:
        click.echo(f"Running {description}...")
        result = run_command(cmd, check=False)
        
        if result.returncode == 0:
            click.echo(f"✅ {description} passed")
        else:
            click.echo(f"❌ {description} failed")
            failed_checks.append(description)
    
    if failed_checks:
        click.echo(f"\n❌ Failed checks: {', '.join(failed_checks)}", err=True)
        sys.exit(1)
    else:
        click.echo("\n✅ All quality checks passed")


@cli.command()
@click.pass_context
def format(ctx):
    """Format code."""
    verbose = ctx.obj['verbose']
    
    formatters = [
        (['poetry', 'run', 'black', '.'], 'Black formatting'),
        (['poetry', 'run', 'isort', '.'], 'Import sorting'),
        (['poetry', 'run', 'ruff', '--fix', '.'], 'Ruff auto-fixes'),
    ]
    
    for cmd, description in formatters:
        click.echo(f"Running {description}...")
        result = run_command(cmd)
        
        if result.returncode == 0:
            click.echo(f"✅ {description} completed")
    
    click.echo("\n✅ Code formatting completed")


@cli.command()
@click.pass_context
def docs(ctx):
    """Build documentation."""
    verbose = ctx.obj['verbose']
    
    click.echo("Building documentation...")
    result = run_command(['poetry', 'run', 'mkdocs', 'build'])
    
    if result.returncode == 0:
        click.echo("✅ Documentation built successfully")
        click.echo("📖 Documentation available in site/ directory")
    else:
        click.echo("❌ Documentation build failed", err=True)
        sys.exit(1)


@cli.command()
@click.option('--clean', is_flag=True, help='Clean before building')
@click.pass_context
def all(ctx, clean):
    """Run all build steps."""
    verbose = ctx.obj['verbose']
    
    if clean:
        click.echo("Cleaning build artifacts...")
        run_command(['rm', '-rf', 'dist/', 'build/', 'site/', 'htmlcov/', '.coverage'], check=False)
    
    steps = [
        ('format', 'Code formatting'),
        ('lint', 'Code quality checks'),
        ('test', 'Testing'),
        ('package', 'Package building'),
        ('docs', 'Documentation'),
        ('docker', 'Docker image'),
    ]
    
    for step, description in steps:
        click.echo(f"\n🔨 {description}...")
        try:
            ctx.invoke(eval(step))
        except SystemExit:
            click.echo(f"❌ {description} failed", err=True)
            sys.exit(1)
    
    click.echo("\n🎉 All build steps completed successfully!")


@cli.command()
@click.pass_context
def clean(ctx):
    """Clean build artifacts."""
    verbose = ctx.obj['verbose']
    
    artifacts = [
        'dist/',
        'build/',
        'site/',
        'htmlcov/',
        '.coverage',
        '.pytest_cache/',
        '.mypy_cache/',
        '.ruff_cache/',
        '**/__pycache__/',
        '**/*.pyc',
        '*.egg-info/',
    ]
    
    click.echo("Cleaning build artifacts...")
    
    for pattern in artifacts:
        run_command(['find', '.', '-name', pattern, '-type', 'd', '-exec', 'rm', '-rf', '{}', '+'], check=False)
        run_command(['find', '.', '-name', pattern, '-type', 'f', '-delete'], check=False)
    
    click.echo("✅ Build artifacts cleaned")


if __name__ == '__main__':
    cli()
