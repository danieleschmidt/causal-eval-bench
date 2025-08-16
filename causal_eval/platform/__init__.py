"""
Platform Support Module

This module provides comprehensive platform compatibility and deployment support
for the causal evaluation framework, enabling global deployment across different
environments and architectures.
"""

from .compatibility import (
    SupportedPlatform,
    PlatformInfo,
    CompatibilityRequirements,
    PlatformDetector,
    CompatibilityChecker,
    PlatformOptimizer
)

from .deployment import (
    DeploymentEnvironment,
    DeploymentTarget,
    ServiceConfiguration,
    DatabaseConfiguration,
    CacheConfiguration,
    MonitoringConfiguration,
    SecurityConfiguration,
    DeploymentConfiguration,
    DeploymentConfigurationGenerator,
    DeploymentManager
)

__all__ = [
    # Compatibility
    'SupportedPlatform',
    'PlatformInfo',
    'CompatibilityRequirements',
    'PlatformDetector',
    'CompatibilityChecker',
    'PlatformOptimizer',
    
    # Deployment
    'DeploymentEnvironment',
    'DeploymentTarget',
    'ServiceConfiguration',
    'DatabaseConfiguration',
    'CacheConfiguration',
    'MonitoringConfiguration',
    'SecurityConfiguration',
    'DeploymentConfiguration',
    'DeploymentConfigurationGenerator',
    'DeploymentManager'
]

__version__ = "1.0.0"