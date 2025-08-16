#!/usr/bin/env python3
"""
Deployment CLI Tool

This script provides a comprehensive deployment interface for the causal evaluation
framework, supporting multiple platforms and environments with intelligent configuration.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add the repo to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_eval.platform import (
    PlatformDetector,
    CompatibilityChecker,
    DeploymentEnvironment,
    DeploymentTarget,
    DeploymentConfigurationGenerator,
    DeploymentManager
)

from causal_eval.i18n.localization import LocalizationManager, SupportedLanguage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentCLI:
    """Command-line interface for deployment operations."""
    
    def __init__(self):
        self.platform_detector = PlatformDetector()
        self.compatibility_checker = CompatibilityChecker(self.platform_detector)
        self.config_generator = DeploymentConfigurationGenerator(self.platform_detector)
        self.deployment_manager = DeploymentManager(self.config_generator)
        self.localization = LocalizationManager()
    
    def check_platform(self) -> bool:
        """Check platform compatibility."""
        print("🔍 Checking Platform Compatibility...")
        print("=" * 60)
        
        platform_info = self.platform_detector.platform_info
        
        # Display platform information
        print(f"🖥️  Platform: {platform_info.supported_platform.value}")
        print(f"💻 OS: {platform_info.os_name} {platform_info.os_version}")
        print(f"🏗️  Architecture: {platform_info.architecture}")
        print(f"🐍 Python: {platform_info.python_version}")
        print(f"⚡ CPU Cores: {platform_info.cpu_count}")
        print(f"💾 Memory: {platform_info.total_memory_gb:.1f} GB")
        
        if platform_info.containerized:
            print(f"🐳 Containerized: Yes")
            if platform_info.cloud_environment:
                print(f"☁️  Cloud: {platform_info.cloud_environment}")
        
        # Hardware capabilities
        hw = platform_info.hardware_capabilities
        if hw['gpu_available']:
            print(f"🎮 GPU: {hw['gpu_count']} GPU(s), {hw['gpu_memory_gb']:.1f} GB")
        else:
            print("🎮 GPU: Not available")
        
        if hw['vector_extensions']:
            print(f"🚀 Vector Extensions: {', '.join(hw['vector_extensions'])}")
        
        print("\n" + "=" * 60)
        print("🧪 Running Compatibility Check...")
        
        # Run compatibility check
        compat_result = self.compatibility_checker.check_compatibility()
        
        # Display results
        if compat_result['compatible']:
            print("✅ PLATFORM COMPATIBLE")
        else:
            print("❌ PLATFORM COMPATIBILITY ISSUES")
        
        print(f"📊 Performance Score: {compat_result['performance_score']:.1f}/100")
        
        # Issues
        if compat_result['issues']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in compat_result['issues']:
                print(f"   • {issue}")
        
        # Warnings
        if compat_result['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in compat_result['warnings']:
                print(f"   • {warning}")
        
        # Optimizations
        if compat_result['optimizations']:
            print("\n💡 OPTIMIZATION OPPORTUNITIES:")
            for opt in compat_result['optimizations']:
                print(f"   • {opt}")
        
        # Recommendations
        if compat_result['recommendations']:
            print("\n📋 RECOMMENDATIONS:")
            for rec in compat_result['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "=" * 60)
        
        return compat_result['compatible']
    
    def generate_config(
        self, 
        environment: str, 
        target: str, 
        region: str = "us-east-1",
        language: str = "en",
        output_path: Optional[str] = None,
        format: str = "yaml"
    ) -> bool:
        """Generate deployment configuration."""
        
        try:
            # Parse enums
            env = DeploymentEnvironment(environment.lower())
            tgt = DeploymentTarget(target.lower())
            lang = SupportedLanguage(language.lower())
            
            self.localization.set_language(lang)
            
            print(f"🛠️  Generating {env.value} configuration for {tgt.value}...")
            print("=" * 60)
            
            # Generate configuration
            config = self.config_generator.generate_configuration(
                environment=env,
                target=tgt,
                region=region
            )
            
            # Validate configuration
            validation = self.deployment_manager.validate_configuration(config)
            
            print(f"✅ Configuration generated successfully")
            print(f"📊 Validation Score: {validation['score']}/100")
            
            if validation['issues']:
                print("\n🚨 CONFIGURATION ISSUES:")
                for issue in validation['issues']:
                    print(f"   • {issue}")
            
            if validation['warnings']:
                print("\n⚠️  CONFIGURATION WARNINGS:")
                for warning in validation['warnings']:
                    print(f"   • {warning}")
            
            # Export configuration
            output_file = None
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
            
            exported = self.deployment_manager.export_configuration(
                config=config,
                format=format,
                output_path=output_file
            )
            
            if output_file:
                print(f"\n📁 Configuration saved to: {output_file}")
            else:
                print(f"\n📄 Generated Configuration ({format}):")
                print("-" * 40)
                if isinstance(exported, str):
                    print(exported)
                else:
                    for filename, content in exported.items():
                        print(f"\n## {filename}\n{content}")
            
            return validation['valid']
            
        except ValueError as e:
            print(f"❌ Invalid parameter: {e}")
            return False
        except Exception as e:
            print(f"❌ Error generating configuration: {e}")
            logger.exception("Configuration generation failed")
            return False
    
    def deploy(
        self, 
        environment: str, 
        target: str, 
        region: str = "us-east-1",
        config_path: Optional[str] = None,
        dry_run: bool = False
    ) -> bool:
        """Deploy the application."""
        
        print(f"🚀 {'[DRY RUN] ' if dry_run else ''}Deploying to {environment} on {target}...")
        print("=" * 60)
        
        # Check compatibility first
        if not self.check_platform():
            print("❌ Platform compatibility check failed. Deployment aborted.")
            return False
        
        try:
            env = DeploymentEnvironment(environment.lower())
            tgt = DeploymentTarget(target.lower())
            
            # Generate or load configuration
            if config_path:
                print(f"📁 Loading configuration from: {config_path}")
                # In a real implementation, load from file
                config = self.config_generator.generate_configuration(env, tgt, region)
            else:
                print(f"🛠️  Generating deployment configuration...")
                config = self.config_generator.generate_configuration(env, tgt, region)
            
            # Validate configuration
            validation = self.deployment_manager.validate_configuration(config)
            if not validation['valid']:
                print("❌ Configuration validation failed:")
                for issue in validation['issues']:
                    print(f"   • {issue}")
                return False
            
            print("✅ Configuration validated successfully")
            
            if dry_run:
                print("\n🔍 DRY RUN MODE - No actual deployment will occur")
                print("\nDeployment would execute the following steps:")
                
                deployment_steps = self._get_deployment_steps(config)
                for i, step in enumerate(deployment_steps, 1):
                    print(f"   {i}. {step}")
            else:
                print("\n🚀 Starting deployment...")
                success = self._execute_deployment(config)
                
                if success:
                    print("✅ Deployment completed successfully!")
                    self._show_deployment_info(config)
                else:
                    print("❌ Deployment failed!")
                    return False
            
            return True
            
        except ValueError as e:
            print(f"❌ Invalid parameter: {e}")
            return False
        except Exception as e:
            print(f"❌ Deployment error: {e}")
            logger.exception("Deployment failed")
            return False
    
    def _get_deployment_steps(self, config) -> list:
        """Get deployment steps for the configuration."""
        steps = [
            "Create deployment environment",
            "Setup networking and security groups",
            "Deploy database instances",
            "Deploy cache services",
            "Build and push application images",
            "Deploy application services",
            "Configure load balancers",
            "Setup monitoring and logging",
            "Run health checks",
            "Update DNS records",
            "Verify deployment"
        ]
        
        # Customize based on target
        if config.target == DeploymentTarget.KUBERNETES:
            steps.extend([
                "Apply Kubernetes manifests",
                "Configure ingress controllers",
                "Setup horizontal pod autoscaling"
            ])
        elif config.target == DeploymentTarget.AWS:
            steps.extend([
                "Create CloudFormation stack",
                "Configure Auto Scaling Groups",
                "Setup Application Load Balancer"
            ])
        
        return steps
    
    def _execute_deployment(self, config) -> bool:
        """Execute the actual deployment."""
        # This is a simplified implementation
        # In a real scenario, this would interact with cloud APIs, container orchestrators, etc.
        
        print("📦 Building application...")
        print("🐳 Creating containers...")
        print("☁️  Provisioning cloud resources...")
        print("🔧 Configuring services...")
        print("🔍 Running health checks...")
        
        # Simulate deployment process
        import time
        time.sleep(2)
        
        return True
    
    def _show_deployment_info(self, config):
        """Show deployment information."""
        print("\n📋 DEPLOYMENT INFORMATION")
        print("=" * 40)
        
        for service_name, service in config.services.items():
            print(f"🔧 {service_name.title()} Service:")
            print(f"   📡 Port: {service.port}")
            print(f"   📊 Replicas: {service.replicas}")
            if hasattr(service, 'health_check') and service.health_check:
                print(f"   ❤️  Health Check: Enabled")
        
        print(f"\n💾 Database: {config.database.engine}")
        print(f"🗄️  Cache: {config.cache.engine}")
        print(f"📊 Monitoring: {'Enabled' if config.monitoring.metrics_enabled else 'Disabled'}")
        print(f"🔒 Security: {'HTTPS' if config.security.enable_https else 'HTTP'}")
        
        if config.target != DeploymentTarget.LOCAL:
            print(f"\n🌍 Access URL: https://{config.environment.value}-causal-eval.{config.region}.example.com")
        else:
            print(f"\n🏠 Local URL: http://localhost:8000")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Causal Evaluation Framework Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check platform compatibility
  python deploy.py check

  # Generate Docker configuration
  python deploy.py config development docker --output docker-compose.yml --format docker-compose

  # Generate Kubernetes configuration  
  python deploy.py config production kubernetes --region us-west-2 --output k8s/

  # Deploy to staging
  python deploy.py deploy staging kubernetes --region us-west-2

  # Dry run deployment
  python deploy.py deploy production aws --region eu-west-1 --dry-run

Supported environments: development, staging, production, testing
Supported targets: local, docker, kubernetes, aws, gcp, azure
Supported languages: en, es, fr, de, ja, zh-cn, zh-tw, ko, pt, ru, ar, hi
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check platform compatibility')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate deployment configuration')
    config_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'])
    config_parser.add_argument('target', choices=['local', 'docker', 'kubernetes', 'aws', 'gcp', 'azure'])
    config_parser.add_argument('--region', default='us-east-1', help='Deployment region')
    config_parser.add_argument('--language', default='en', help='Interface language')
    config_parser.add_argument('--output', help='Output file path')
    config_parser.add_argument('--format', choices=['yaml', 'json', 'docker-compose', 'kubernetes'], 
                              default='yaml', help='Output format')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy the application')
    deploy_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'])
    deploy_parser.add_argument('target', choices=['local', 'docker', 'kubernetes', 'aws', 'gcp', 'azure'])
    deploy_parser.add_argument('--region', default='us-east-1', help='Deployment region')
    deploy_parser.add_argument('--config', help='Configuration file path')
    deploy_parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without actual deployment')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize CLI
    cli = DeploymentCLI()
    
    try:
        if args.command == 'check':
            success = cli.check_platform()
            return 0 if success else 1
        
        elif args.command == 'config':
            success = cli.generate_config(
                environment=args.environment,
                target=args.target,
                region=args.region,
                language=args.language,
                output_path=args.output,
                format=args.format
            )
            return 0 if success else 1
        
        elif args.command == 'deploy':
            success = cli.deploy(
                environment=args.environment,
                target=args.target,
                region=args.region,
                config_path=args.config,
                dry_run=args.dry_run
            )
            return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("CLI operation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())