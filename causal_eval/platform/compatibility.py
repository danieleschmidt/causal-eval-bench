"""
Cross-Platform Compatibility Module

This module ensures the causal evaluation framework works seamlessly across
different operating systems, hardware architectures, and deployment environments.
"""

import os
import sys
import platform
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedPlatform(Enum):
    """Supported platforms for deployment."""
    
    LINUX_X86_64 = "linux-x86_64"
    LINUX_ARM64 = "linux-arm64"
    WINDOWS_X86_64 = "windows-x86_64"
    MACOS_X86_64 = "darwin-x86_64"
    MACOS_ARM64 = "darwin-arm64"
    DOCKER_LINUX = "docker-linux"
    KUBERNETES = "kubernetes"
    CLOUD_SERVERLESS = "cloud-serverless"


@dataclass
class PlatformInfo:
    """Information about the current platform."""
    
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    cpu_count: int
    total_memory_gb: float
    supported_platform: SupportedPlatform
    containerized: bool
    cloud_environment: Optional[str]
    hardware_capabilities: Dict[str, Any]


@dataclass
class CompatibilityRequirements:
    """Compatibility requirements for the framework."""
    
    min_python_version: Tuple[int, int, int]
    min_memory_gb: float
    min_cpu_cores: int
    required_packages: List[str]
    optional_packages: List[str]
    supported_platforms: List[SupportedPlatform]
    hardware_acceleration: Dict[str, bool]


class PlatformDetector:
    """Detects and analyzes the current platform."""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        logger.info(f"Platform detected: {self.platform_info.supported_platform.value}")
    
    def _detect_platform(self) -> PlatformInfo:
        """Detect current platform information."""
        
        # Basic platform info
        os_name = platform.system().lower()
        os_version = platform.release()
        architecture = platform.machine().lower()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        cpu_count = os.cpu_count() or 1
        
        # Memory detection
        total_memory_gb = self._get_total_memory_gb()
        
        # Determine supported platform
        supported_platform = self._determine_supported_platform(os_name, architecture)
        
        # Container detection
        containerized = self._is_containerized()
        
        # Cloud environment detection
        cloud_environment = self._detect_cloud_environment()
        
        # Hardware capabilities
        hardware_capabilities = self._detect_hardware_capabilities()
        
        return PlatformInfo(
            os_name=os_name,
            os_version=os_version,
            architecture=architecture,
            python_version=python_version,
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            supported_platform=supported_platform,
            containerized=containerized,
            cloud_environment=cloud_environment,
            hardware_capabilities=hardware_capabilities
        )
    
    def _get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            if platform.system() == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return kb / (1024 * 1024)
            elif platform.system() == "Darwin":
                result = subprocess.run(['sysctl', 'hw.memsize'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    bytes_mem = int(result.stdout.split(': ')[1])
                    return bytes_mem / (1024 ** 3)
            elif platform.system() == "Windows":
                import psutil
                return psutil.virtual_memory().total / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Could not detect memory: {e}")
        
        return 4.0  # Default fallback
    
    def _determine_supported_platform(self, os_name: str, architecture: str) -> SupportedPlatform:
        """Determine the supported platform enum."""
        
        # Check if running in container first
        if self._is_containerized():
            if self._is_kubernetes():
                return SupportedPlatform.KUBERNETES
            else:
                return SupportedPlatform.DOCKER_LINUX
        
        # Check cloud serverless environments
        if self._is_serverless():
            return SupportedPlatform.CLOUD_SERVERLESS
        
        # Standard platform detection
        if os_name == "linux":
            if "x86_64" in architecture or "amd64" in architecture:
                return SupportedPlatform.LINUX_X86_64
            elif "aarch64" in architecture or "arm64" in architecture:
                return SupportedPlatform.LINUX_ARM64
        elif os_name == "darwin":
            if "x86_64" in architecture:
                return SupportedPlatform.MACOS_X86_64
            elif "arm64" in architecture:
                return SupportedPlatform.MACOS_ARM64
        elif os_name == "windows":
            return SupportedPlatform.WINDOWS_X86_64
        
        # Default fallback
        return SupportedPlatform.LINUX_X86_64
    
    def _is_containerized(self) -> bool:
        """Check if running in a container."""
        indicators = [
            os.path.exists('/.dockerenv'),
            os.path.exists('/proc/self/cgroup') and self._check_cgroup_container(),
            os.environ.get('container') is not None,
            os.environ.get('KUBERNETES_SERVICE_HOST') is not None
        ]
        return any(indicators)
    
    def _check_cgroup_container(self) -> bool:
        """Check cgroup for container indicators."""
        try:
            with open('/proc/self/cgroup', 'r') as f:
                content = f.read()
                return 'docker' in content or 'kubepods' in content
        except Exception:
            return False
    
    def _is_kubernetes(self) -> bool:
        """Check if running in Kubernetes."""
        return (
            os.environ.get('KUBERNETES_SERVICE_HOST') is not None or
            os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/')
        )
    
    def _is_serverless(self) -> bool:
        """Check if running in serverless environment."""
        serverless_indicators = [
            'AWS_LAMBDA_FUNCTION_NAME',
            'AZURE_FUNCTIONS_ENVIRONMENT',
            'GOOGLE_CLOUD_PROJECT',
            'VERCEL',
            'NETLIFY'
        ]
        return any(os.environ.get(indicator) for indicator in serverless_indicators)
    
    def _detect_cloud_environment(self) -> Optional[str]:
        """Detect cloud environment."""
        if os.environ.get('AWS_REGION'):
            return 'aws'
        elif os.environ.get('AZURE_CLIENT_ID'):
            return 'azure'
        elif os.environ.get('GOOGLE_CLOUD_PROJECT'):
            return 'gcp'
        elif os.environ.get('DIGITALOCEAN_TOKEN'):
            return 'digitalocean'
        return None
    
    def _detect_hardware_capabilities(self) -> Dict[str, Any]:
        """Detect hardware capabilities."""
        capabilities = {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'vector_extensions': [],
            'numa_nodes': 1,
            'virtualized': False
        }
        
        # GPU detection
        try:
            # Try NVIDIA GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=count,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                capabilities['gpu_available'] = True
                capabilities['gpu_count'] = len(lines)
                capabilities['gpu_memory_gb'] = sum(int(line.split(',')[1]) for line in lines) / 1024
        except Exception:
            pass
        
        # CPU extensions
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    if 'avx2' in content:
                        capabilities['vector_extensions'].append('avx2')
                    if 'avx512' in content:
                        capabilities['vector_extensions'].append('avx512')
        except Exception:
            pass
        
        # Virtualization detection
        try:
            if platform.system() == "Linux":
                result = subprocess.run(['systemd-detect-virt'], 
                                      capture_output=True, text=True)
                capabilities['virtualized'] = result.returncode == 0 and result.stdout.strip() != 'none'
        except Exception:
            pass
        
        return capabilities


class CompatibilityChecker:
    """Checks compatibility requirements."""
    
    def __init__(self, detector: PlatformDetector):
        self.detector = detector
        self.requirements = self._get_framework_requirements()
    
    def _get_framework_requirements(self) -> CompatibilityRequirements:
        """Get framework compatibility requirements."""
        return CompatibilityRequirements(
            min_python_version=(3, 9, 0),
            min_memory_gb=2.0,
            min_cpu_cores=2,
            required_packages=[
                'fastapi',
                'pydantic',
                'sqlalchemy',
                'alembic',
                'redis',
                'numpy',
                'scipy',
                'scikit-learn',
                'matplotlib',
                'seaborn'
            ],
            optional_packages=[
                'torch',
                'tensorflow',
                'transformers',
                'psutil',
                'psycopg2-binary',
                'prometheus-client'
            ],
            supported_platforms=[
                SupportedPlatform.LINUX_X86_64,
                SupportedPlatform.LINUX_ARM64,
                SupportedPlatform.MACOS_X86_64,
                SupportedPlatform.MACOS_ARM64,
                SupportedPlatform.WINDOWS_X86_64,
                SupportedPlatform.DOCKER_LINUX,
                SupportedPlatform.KUBERNETES,
                SupportedPlatform.CLOUD_SERVERLESS
            ],
            hardware_acceleration={
                'gpu_recommended': True,
                'vector_extensions_recommended': True,
                'min_memory_for_gpu': 8.0
            }
        )
    
    def check_compatibility(self) -> Dict[str, Any]:
        """Check platform compatibility."""
        platform_info = self.detector.platform_info
        issues = []
        warnings = []
        optimizations = []
        
        # Python version check
        current_py = sys.version_info[:3]
        min_py = self.requirements.min_python_version
        if current_py < min_py:
            issues.append(f"Python {min_py[0]}.{min_py[1]}.{min_py[2]}+ required, found {current_py[0]}.{current_py[1]}.{current_py[2]}")
        
        # Platform support check
        if platform_info.supported_platform not in self.requirements.supported_platforms:
            issues.append(f"Platform {platform_info.supported_platform.value} not officially supported")
        
        # Memory check
        if platform_info.total_memory_gb < self.requirements.min_memory_gb:
            issues.append(f"Minimum {self.requirements.min_memory_gb}GB RAM required, found {platform_info.total_memory_gb:.1f}GB")
        
        # CPU check
        if platform_info.cpu_count < self.requirements.min_cpu_cores:
            warnings.append(f"Minimum {self.requirements.min_cpu_cores} CPU cores recommended, found {platform_info.cpu_count}")
        
        # Package availability check
        missing_packages = self._check_packages(self.requirements.required_packages)
        if missing_packages:
            issues.append(f"Missing required packages: {', '.join(missing_packages)}")
        
        # Optional packages
        missing_optional = self._check_packages(self.requirements.optional_packages)
        if missing_optional:
            warnings.append(f"Missing optional packages for enhanced features: {', '.join(missing_optional)}")
        
        # Hardware optimizations
        if not platform_info.hardware_capabilities['gpu_available']:
            optimizations.append("GPU acceleration not available - consider GPU-enabled instance for better performance")
        
        if not platform_info.hardware_capabilities['vector_extensions']:
            optimizations.append("CPU vector extensions (AVX2/AVX512) not detected - performance may be limited")
        
        # Container-specific checks
        if platform_info.containerized:
            optimizations.append("Running in container - ensure adequate resource limits")
        
        # Cloud-specific recommendations
        if platform_info.cloud_environment:
            optimizations.append(f"Running on {platform_info.cloud_environment} - consider cloud-native optimizations")
        
        return {
            'compatible': len(issues) == 0,
            'platform_info': platform_info,
            'issues': issues,
            'warnings': warnings,
            'optimizations': optimizations,
            'performance_score': self._calculate_performance_score(platform_info),
            'recommendations': self._generate_recommendations(platform_info, issues, warnings)
        }
    
    def _check_packages(self, packages: List[str]) -> List[str]:
        """Check if packages are available."""
        missing = []
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        return missing
    
    def _calculate_performance_score(self, platform_info: PlatformInfo) -> float:
        """Calculate performance score (0-100)."""
        score = 50.0  # Base score
        
        # CPU score
        cpu_score = min(25, platform_info.cpu_count * 3)
        score += cpu_score
        
        # Memory score
        memory_score = min(15, platform_info.total_memory_gb * 2)
        score += memory_score
        
        # GPU bonus
        if platform_info.hardware_capabilities['gpu_available']:
            score += 10
        
        # Vector extensions bonus
        if platform_info.hardware_capabilities['vector_extensions']:
            score += 5
        
        # Platform optimization
        if platform_info.supported_platform in [
            SupportedPlatform.LINUX_X86_64,
            SupportedPlatform.DOCKER_LINUX,
            SupportedPlatform.KUBERNETES
        ]:
            score += 5
        
        return min(100.0, score)
    
    def _generate_recommendations(self, platform_info: PlatformInfo, 
                                issues: List[str], warnings: List[str]) -> List[str]:
        """Generate platform-specific recommendations."""
        recommendations = []
        
        if issues:
            recommendations.append("🚨 Critical: Address compatibility issues before deployment")
        
        # Performance recommendations
        if platform_info.total_memory_gb < 8:
            recommendations.append("💾 Consider upgrading to 8GB+ RAM for optimal performance")
        
        if platform_info.cpu_count < 4:
            recommendations.append("⚡ Consider 4+ CPU cores for better concurrent processing")
        
        if not platform_info.hardware_capabilities['gpu_available']:
            recommendations.append("🎮 GPU acceleration recommended for large-scale evaluations")
        
        # Platform-specific recommendations
        if platform_info.supported_platform == SupportedPlatform.WINDOWS_X86_64:
            recommendations.append("🪟 Consider using WSL2 or Docker for better compatibility")
        
        if platform_info.containerized:
            recommendations.append("🐳 Ensure container resource limits match requirements")
        
        if platform_info.cloud_environment == 'aws':
            recommendations.append("☁️ Consider using AWS optimized instances (c5, m5, or p3 series)")
        elif platform_info.cloud_environment == 'gcp':
            recommendations.append("☁️ Consider using GCP compute-optimized instances")
        elif platform_info.cloud_environment == 'azure':
            recommendations.append("☁️ Consider using Azure D-series or F-series instances")
        
        return recommendations


class PlatformOptimizer:
    """Optimizes framework settings for the current platform."""
    
    def __init__(self, detector: PlatformDetector):
        self.detector = detector
    
    def optimize_settings(self) -> Dict[str, Any]:
        """Generate optimized settings for current platform."""
        platform_info = self.detector.platform_info
        
        settings = {
            'database': self._optimize_database_settings(platform_info),
            'cache': self._optimize_cache_settings(platform_info),
            'processing': self._optimize_processing_settings(platform_info),
            'memory': self._optimize_memory_settings(platform_info),
            'networking': self._optimize_networking_settings(platform_info),
            'logging': self._optimize_logging_settings(platform_info)
        }
        
        return settings
    
    def _optimize_database_settings(self, platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize database settings."""
        settings = {
            'pool_size': min(20, platform_info.cpu_count * 2),
            'max_overflow': min(30, platform_info.cpu_count * 3),
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
        
        # Memory-based optimizations
        if platform_info.total_memory_gb >= 16:
            settings['pool_size'] *= 2
            settings['max_overflow'] *= 2
        
        # Container optimizations
        if platform_info.containerized:
            settings['pool_timeout'] = 10
            settings['pool_recycle'] = 1800
        
        return settings
    
    def _optimize_cache_settings(self, platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize cache settings."""
        # Base memory allocation for Redis (MB)
        base_memory = min(512, int(platform_info.total_memory_gb * 1024 * 0.25))
        
        settings = {
            'maxmemory': f"{base_memory}mb",
            'maxmemory_policy': "allkeys-lru",
            'save_frequency': "900 1 300 10 60 10000" if not platform_info.containerized else "",
            'tcp_keepalive': 300
        }
        
        return settings
    
    def _optimize_processing_settings(self, platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize processing settings."""
        # Worker processes
        workers = min(platform_info.cpu_count, 8)
        if platform_info.containerized:
            workers = min(workers, 4)  # Conservative for containers
        
        # Batch sizes based on memory
        if platform_info.total_memory_gb >= 16:
            batch_size = 64
        elif platform_info.total_memory_gb >= 8:
            batch_size = 32
        else:
            batch_size = 16
        
        settings = {
            'worker_processes': workers,
            'worker_threads': min(4, platform_info.cpu_count),
            'batch_size': batch_size,
            'async_timeout': 300,
            'max_concurrent_evaluations': workers * 2
        }
        
        # GPU optimizations
        if platform_info.hardware_capabilities['gpu_available']:
            settings['enable_gpu_acceleration'] = True
            settings['gpu_memory_fraction'] = 0.8
        
        return settings
    
    def _optimize_memory_settings(self, platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize memory settings."""
        # Calculate memory limits
        total_mb = int(platform_info.total_memory_gb * 1024)
        
        settings = {
            'max_heap_size': f"{int(total_mb * 0.6)}m",
            'gc_threshold': int(total_mb * 0.8),
            'enable_memory_profiling': platform_info.total_memory_gb < 8,
            'memory_limit_warning': int(total_mb * 0.85)
        }
        
        return settings
    
    def _optimize_networking_settings(self, platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize networking settings."""
        settings = {
            'connection_timeout': 30,
            'read_timeout': 60,
            'keepalive_timeout': 120,
            'max_connections': min(1000, platform_info.cpu_count * 100)
        }
        
        # Cloud-specific optimizations
        if platform_info.cloud_environment:
            settings['keepalive_timeout'] = 60
            settings['connection_timeout'] = 10
        
        # Container optimizations
        if platform_info.containerized:
            settings['max_connections'] = min(500, settings['max_connections'])
        
        return settings
    
    def _optimize_logging_settings(self, platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize logging settings."""
        settings = {
            'level': 'INFO',
            'format': 'json' if platform_info.cloud_environment else 'text',
            'rotation': 'daily' if not platform_info.containerized else 'size',
            'retention': 7,
            'buffer_size': 1024
        }
        
        # Performance-based adjustments
        if platform_info.total_memory_gb < 4:
            settings['level'] = 'WARNING'
            settings['buffer_size'] = 512
        
        return settings


# Export main classes
__all__ = [
    'SupportedPlatform',
    'PlatformInfo',
    'CompatibilityRequirements',
    'PlatformDetector',
    'CompatibilityChecker',
    'PlatformOptimizer'
]