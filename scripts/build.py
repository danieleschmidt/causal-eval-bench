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
@click.option('--multi-arch', is_flag=True, help='Build multi-architecture image')
@click.option('--push', is_flag=True, help='Push image after build')
@click.option('--cache', is_flag=True, default=True, help='Use build cache')
@click.option('--scan', is_flag=True, help='Scan image for vulnerabilities')
@click.option('--registry', default='ghcr.io/danieleschmidt', help='Container registry')
@click.pass_context
def docker(ctx, target, tag, platform, multi_arch, push, cache, scan, registry):
    """Build Docker image with advanced features."""
    verbose = ctx.obj['verbose']
    
    # Generate version and tags
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        version = f"{target}-{git_sha}"
    except subprocess.CalledProcessError:
        git_sha = "unknown"
        git_branch = "unknown"
        version = f"{target}-latest"
    
    if not tag:
        base_name = "causal-eval-bench"
        tag = f"{registry}/{base_name}:{version}"
        latest_tag = f"{registry}/{base_name}:{target}-latest"
    else:
        latest_tag = None
    
    if multi_arch:
        # Multi-architecture build with buildx
        platforms = "linux/amd64,linux/arm64"
        
        # Ensure buildx builder exists
        builder_name = "causal-eval-builder"
        run_command([
            'docker', 'buildx', 'create', '--name', builder_name,
            '--driver', 'docker-container', '--use'
        ], check=False)
        
        build_cmd = [
            'docker', 'buildx', 'build',
            '--platform', platforms,
            '--target', target,
            '--tag', tag,
        ]
        
        if latest_tag:
            build_cmd.extend(['--tag', latest_tag])
        
        if push:
            build_cmd.append('--push')
        else:
            build_cmd.append('--load')
            
    else:
        # Standard single-architecture build
        build_cmd = [
            'docker', 'build',
            '--target', target,
            '--platform', platform,
            '--tag', tag,
        ]
        
        if latest_tag:
            build_cmd.extend(['--tag', latest_tag])
    
    if not cache:
        build_cmd.append('--no-cache')
    
    # Add comprehensive build args
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        build_date = subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ']).decode().strip()
    except subprocess.CalledProcessError:
        git_commit = "unknown"
        build_date = "unknown"
    
    build_cmd.extend([
        '--build-arg', f'GIT_COMMIT={git_commit}',
        '--build-arg', f'GIT_BRANCH={git_branch}',
        '--build-arg', f'BUILD_DATE={build_date}',
        '--build-arg', f'VERSION={version}',
    ])
    
    build_cmd.append('.')
    
    click.echo(f"Building Docker image: {tag}")
    if multi_arch:
        click.echo(f"Multi-architecture build: {platforms}")
    
    result = run_command(build_cmd)
    
    if result.returncode == 0:
        click.echo(f"✅ Successfully built {tag}")
        
        # Security scan if requested
        if scan:
            click.echo("🔍 Scanning for vulnerabilities...")
            scan_result = run_command(['trivy', 'image', '--severity', 'HIGH,CRITICAL', tag], check=False)
            if scan_result.returncode == 0:
                click.echo("✅ Security scan passed")
            else:
                click.echo("⚠️ Security vulnerabilities found")
        
        # Push if requested (only for non-buildx builds)
        if push and not multi_arch:
            for push_tag in [tag, latest_tag] if latest_tag else [tag]:
                click.echo(f"Pushing {push_tag}...")
                push_result = run_command(['docker', 'push', push_tag])
                if push_result.returncode == 0:
                    click.echo(f"✅ Successfully pushed {push_tag}")
                else:
                    click.echo(f"❌ Failed to push {push_tag}", err=True)
                    
        # Show image size
        size_result = run_command(['docker', 'images', '--format', 'table {{.Size}}', tag], check=False)
        if size_result.returncode == 0 and size_result.stdout:
            size = size_result.stdout.strip().split('\n')[-1]
            click.echo(f"📏 Image size: {size}")
            
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
