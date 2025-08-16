"""
Deployment Configuration Module

This module provides comprehensive deployment configurations for various environments,
ensuring the causal evaluation framework can be deployed globally with optimal settings.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from .compatibility import SupportedPlatform, PlatformDetector, PlatformOptimizer

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DeploymentTarget(Enum):
    """Deployment target platforms."""
    
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    HEROKU = "heroku"
    VERCEL = "vercel"
    DIGITALOCEAN = "digitalocean"


@dataclass
class ServiceConfiguration:
    """Configuration for individual services."""
    
    name: str
    image: str
    version: str
    port: int
    replicas: int
    resources: Dict[str, Any]
    environment: Dict[str, str]
    health_check: Dict[str, Any]
    dependencies: List[str]


@dataclass
class DatabaseConfiguration:
    """Database configuration."""
    
    engine: str  # postgresql, sqlite, mysql
    host: str
    port: int
    database: str
    username: str
    password_env_var: str
    ssl_mode: str
    pool_settings: Dict[str, Any]
    migration_settings: Dict[str, Any]


@dataclass
class CacheConfiguration:
    """Cache configuration."""
    
    engine: str  # redis, memcached, memory
    host: str
    port: int
    password_env_var: Optional[str]
    database: int
    settings: Dict[str, Any]


@dataclass
class MonitoringConfiguration:
    """Monitoring and observability configuration."""
    
    metrics_enabled: bool
    metrics_port: int
    logging_level: str
    tracing_enabled: bool
    health_check_interval: int
    prometheus_config: Dict[str, Any]
    grafana_config: Dict[str, Any]


@dataclass
class SecurityConfiguration:
    """Security configuration."""
    
    enable_https: bool
    ssl_cert_path: Optional[str]
    ssl_key_path: Optional[str]
    enable_cors: bool
    cors_origins: List[str]
    api_rate_limiting: Dict[str, Any]
    authentication: Dict[str, Any]
    data_encryption: Dict[str, Any]


@dataclass
class DeploymentConfiguration:
    """Complete deployment configuration."""
    
    environment: DeploymentEnvironment
    target: DeploymentTarget
    region: str
    services: Dict[str, ServiceConfiguration]
    database: DatabaseConfiguration
    cache: CacheConfiguration
    monitoring: MonitoringConfiguration
    security: SecurityConfiguration
    scaling: Dict[str, Any]
    networking: Dict[str, Any]
    storage: Dict[str, Any]


class DeploymentConfigurationGenerator:
    """Generates deployment configurations for different environments."""
    
    def __init__(self, platform_detector: Optional[PlatformDetector] = None):
        self.platform_detector = platform_detector or PlatformDetector()
        self.platform_optimizer = PlatformOptimizer(self.platform_detector)
        
    def generate_configuration(
        self, 
        environment: DeploymentEnvironment,
        target: DeploymentTarget,
        region: str = "us-east-1",
        custom_settings: Optional[Dict[str, Any]] = None
    ) -> DeploymentConfiguration:
        """Generate deployment configuration."""
        
        # Get platform optimizations
        optimized_settings = self.platform_optimizer.optimize_settings()
        
        # Apply custom settings
        if custom_settings:
            optimized_settings.update(custom_settings)
        
        # Generate configuration based on target
        if target == DeploymentTarget.DOCKER:
            return self._generate_docker_config(environment, region, optimized_settings)
        elif target == DeploymentTarget.KUBERNETES:
            return self._generate_kubernetes_config(environment, region, optimized_settings)
        elif target == DeploymentTarget.AWS:
            return self._generate_aws_config(environment, region, optimized_settings)
        elif target == DeploymentTarget.GCP:
            return self._generate_gcp_config(environment, region, optimized_settings)
        elif target == DeploymentTarget.AZURE:
            return self._generate_azure_config(environment, region, optimized_settings)
        else:
            return self._generate_local_config(environment, region, optimized_settings)
    
    def _generate_docker_config(
        self, 
        environment: DeploymentEnvironment, 
        region: str,
        settings: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate Docker deployment configuration."""
        
        return DeploymentConfiguration(
            environment=environment,
            target=DeploymentTarget.DOCKER,
            region=region,
            services={
                "api": ServiceConfiguration(
                    name="causal-eval-api",
                    image="causal-eval-bench:latest",
                    version="1.0.0",
                    port=8000,
                    replicas=1,
                    resources={
                        "limits": {
                            "memory": f"{int(settings['memory']['max_heap_size'][:-1]) * 2}Mi",
                            "cpu": f"{settings['processing']['worker_processes']}000m"
                        },
                        "requests": {
                            "memory": f"{int(settings['memory']['max_heap_size'][:-1])}Mi",
                            "cpu": f"{settings['processing']['worker_processes'] * 500}m"
                        }
                    },
                    environment={
                        "ENVIRONMENT": environment.value,
                        "DATABASE_URL": "postgresql://user:password@db:5432/causal_eval",
                        "REDIS_URL": "redis://redis:6379/0",
                        "LOG_LEVEL": "INFO",
                        "WORKERS": str(settings['processing']['worker_processes'])
                    },
                    health_check={
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    },
                    dependencies=["db", "redis"]
                )
            },
            database=DatabaseConfiguration(
                engine="postgresql",
                host="db",
                port=5432,
                database="causal_eval",
                username="postgres",
                password_env_var="POSTGRES_PASSWORD",
                ssl_mode="disable",
                pool_settings=settings['database'],
                migration_settings={"auto_migrate": True}
            ),
            cache=CacheConfiguration(
                engine="redis",
                host="redis",
                port=6379,
                password_env_var=None,
                database=0,
                settings=settings['cache']
            ),
            monitoring=self._get_monitoring_config(environment),
            security=self._get_security_config(environment),
            scaling={"auto_scaling": False, "min_replicas": 1, "max_replicas": 3},
            networking={"enable_load_balancer": False, "internal_only": False},
            storage={"persistent_volumes": True, "backup_enabled": True}
        )
    
    def _generate_kubernetes_config(
        self, 
        environment: DeploymentEnvironment, 
        region: str,
        settings: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate Kubernetes deployment configuration."""
        
        replicas = 3 if environment == DeploymentEnvironment.PRODUCTION else 2
        
        return DeploymentConfiguration(
            environment=environment,
            target=DeploymentTarget.KUBERNETES,
            region=region,
            services={
                "api": ServiceConfiguration(
                    name="causal-eval-api",
                    image="causal-eval-bench:latest",
                    version="1.0.0",
                    port=8000,
                    replicas=replicas,
                    resources={
                        "limits": {
                            "memory": f"{int(settings['memory']['max_heap_size'][:-1]) * 2}Mi",
                            "cpu": f"{settings['processing']['worker_processes']}",
                            "ephemeral-storage": "2Gi"
                        },
                        "requests": {
                            "memory": f"{int(settings['memory']['max_heap_size'][:-1])}Mi",
                            "cpu": f"{settings['processing']['worker_processes'] * 0.5}",
                            "ephemeral-storage": "1Gi"
                        }
                    },
                    environment={
                        "ENVIRONMENT": environment.value,
                        "DATABASE_URL": "postgresql://user:password@postgres-service:5432/causal_eval",
                        "REDIS_URL": "redis://redis-service:6379/0",
                        "LOG_LEVEL": "INFO",
                        "WORKERS": str(settings['processing']['worker_processes']),
                        "KUBERNETES_NAMESPACE": "${KUBERNETES_NAMESPACE}"
                    },
                    health_check={
                        "httpGet": {
                            "path": "/health",
                            "port": 8000
                        },
                        "initialDelaySeconds": 30,
                        "periodSeconds": 10,
                        "timeoutSeconds": 5,
                        "failureThreshold": 3
                    },
                    dependencies=["postgres-service", "redis-service"]
                )
            },
            database=DatabaseConfiguration(
                engine="postgresql",
                host="postgres-service",
                port=5432,
                database="causal_eval",
                username="postgres",
                password_env_var="POSTGRES_PASSWORD",
                ssl_mode="require",
                pool_settings=settings['database'],
                migration_settings={"auto_migrate": True, "init_job": True}
            ),
            cache=CacheConfiguration(
                engine="redis",
                host="redis-service",
                port=6379,
                password_env_var="REDIS_PASSWORD",
                database=0,
                settings=settings['cache']
            ),
            monitoring=self._get_monitoring_config(environment, kubernetes=True),
            security=self._get_security_config(environment, kubernetes=True),
            scaling={
                "auto_scaling": True,
                "min_replicas": replicas,
                "max_replicas": replicas * 3,
                "target_cpu_utilization": 70,
                "target_memory_utilization": 80
            },
            networking={
                "enable_load_balancer": True,
                "ingress_enabled": True,
                "service_mesh": environment == DeploymentEnvironment.PRODUCTION
            },
            storage={
                "persistent_volumes": True,
                "storage_class": "fast-ssd",
                "backup_enabled": True,
                "snapshot_schedule": "0 2 * * *"
            }
        )
    
    def _generate_aws_config(
        self, 
        environment: DeploymentEnvironment, 
        region: str,
        settings: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate AWS deployment configuration."""
        
        instance_type = "t3.large" if environment == DeploymentEnvironment.DEVELOPMENT else "c5.xlarge"
        
        return DeploymentConfiguration(
            environment=environment,
            target=DeploymentTarget.AWS,
            region=region,
            services={
                "api": ServiceConfiguration(
                    name="causal-eval-api",
                    image="causal-eval-bench:latest",
                    version="1.0.0",
                    port=8000,
                    replicas=2 if environment == DeploymentEnvironment.DEVELOPMENT else 3,
                    resources={
                        "instance_type": instance_type,
                        "auto_scaling_group": {
                            "min_size": 2,
                            "max_size": 10,
                            "desired_capacity": 2
                        }
                    },
                    environment={
                        "ENVIRONMENT": environment.value,
                        "DATABASE_URL": "${DATABASE_URL}",
                        "REDIS_URL": "${REDIS_URL}",
                        "AWS_REGION": region,
                        "LOG_LEVEL": "INFO"
                    },
                    health_check={
                        "type": "ELB",
                        "grace_period": 300,
                        "health_check_type": "HTTP",
                        "health_check_path": "/health"
                    },
                    dependencies=["rds", "elasticache"]
                )
            },
            database=DatabaseConfiguration(
                engine="postgresql",
                host="${RDS_ENDPOINT}",
                port=5432,
                database="causal_eval",
                username="postgres",
                password_env_var="RDS_PASSWORD",
                ssl_mode="require",
                pool_settings=settings['database'],
                migration_settings={
                    "auto_migrate": True,
                    "backup_retention": 7,
                    "multi_az": environment == DeploymentEnvironment.PRODUCTION
                }
            ),
            cache=CacheConfiguration(
                engine="redis",
                host="${ELASTICACHE_ENDPOINT}",
                port=6379,
                password_env_var="ELASTICACHE_AUTH_TOKEN",
                database=0,
                settings={
                    **settings['cache'],
                    "node_type": "cache.t3.micro" if environment == DeploymentEnvironment.DEVELOPMENT else "cache.r6g.large",
                    "num_cache_nodes": 1 if environment == DeploymentEnvironment.DEVELOPMENT else 3
                }
            ),
            monitoring=self._get_monitoring_config(environment, aws=True),
            security=self._get_security_config(environment, aws=True),
            scaling={
                "auto_scaling": True,
                "target_tracking": {
                    "cpu_utilization": 70,
                    "request_count": 1000
                },
                "step_scaling": True
            },
            networking={
                "vpc_config": {
                    "subnets": ["private", "public"],
                    "security_groups": ["api", "database", "cache"]
                },
                "load_balancer": {
                    "type": "application",
                    "scheme": "internet-facing",
                    "ssl_policy": "ELBSecurityPolicy-TLS-1-2-2017-01"
                }
            },
            storage={
                "s3_buckets": ["model-artifacts", "evaluation-results", "backups"],
                "ebs_optimization": True,
                "backup_enabled": True
            }
        )
    
    def _generate_gcp_config(
        self, 
        environment: DeploymentEnvironment, 
        region: str,
        settings: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate GCP deployment configuration."""
        
        machine_type = "e2-standard-2" if environment == DeploymentEnvironment.DEVELOPMENT else "c2-standard-4"
        
        return DeploymentConfiguration(
            environment=environment,
            target=DeploymentTarget.GCP,
            region=region,
            services={
                "api": ServiceConfiguration(
                    name="causal-eval-api",
                    image="gcr.io/PROJECT_ID/causal-eval-bench:latest",
                    version="1.0.0",
                    port=8000,
                    replicas=2 if environment == DeploymentEnvironment.DEVELOPMENT else 3,
                    resources={
                        "machine_type": machine_type,
                        "disk_size": "20GB",
                        "disk_type": "pd-ssd"
                    },
                    environment={
                        "ENVIRONMENT": environment.value,
                        "DATABASE_URL": "${DATABASE_URL}",
                        "REDIS_URL": "${REDIS_URL}",
                        "GOOGLE_CLOUD_PROJECT": "${GOOGLE_CLOUD_PROJECT}",
                        "LOG_LEVEL": "INFO"
                    },
                    health_check={
                        "type": "HTTP",
                        "request_path": "/health",
                        "port": 8000,
                        "check_interval_sec": 30,
                        "timeout_sec": 10
                    },
                    dependencies=["cloud-sql", "memorystore"]
                )
            },
            database=DatabaseConfiguration(
                engine="postgresql",
                host="${CLOUD_SQL_CONNECTION_NAME}",
                port=5432,
                database="causal_eval",
                username="postgres",
                password_env_var="CLOUD_SQL_PASSWORD",
                ssl_mode="require",
                pool_settings=settings['database'],
                migration_settings={
                    "auto_migrate": True,
                    "backup_enabled": True,
                    "high_availability": environment == DeploymentEnvironment.PRODUCTION
                }
            ),
            cache=CacheConfiguration(
                engine="redis",
                host="${MEMORYSTORE_HOST}",
                port=6379,
                password_env_var="MEMORYSTORE_AUTH_STRING",
                database=0,
                settings={
                    **settings['cache'],
                    "memory_size_gb": 1 if environment == DeploymentEnvironment.DEVELOPMENT else 4,
                    "tier": "BASIC" if environment == DeploymentEnvironment.DEVELOPMENT else "STANDARD_HA"
                }
            ),
            monitoring=self._get_monitoring_config(environment, gcp=True),
            security=self._get_security_config(environment, gcp=True),
            scaling={
                "auto_scaling": True,
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 0.7,
                "cool_down_period": 90
            },
            networking={
                "vpc_config": {
                    "network": "default",
                    "firewall_rules": ["allow-http", "allow-https", "allow-health-check"]
                },
                "load_balancer": {
                    "type": "EXTERNAL",
                    "protocol": "HTTP",
                    "ssl_certificates": True
                }
            },
            storage={
                "cloud_storage_buckets": ["artifacts", "results", "backups"],
                "persistent_disks": True,
                "backup_schedule": "daily"
            }
        )
    
    def _generate_azure_config(
        self, 
        environment: DeploymentEnvironment, 
        region: str,
        settings: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate Azure deployment configuration."""
        
        vm_size = "Standard_B2s" if environment == DeploymentEnvironment.DEVELOPMENT else "Standard_D4s_v3"
        
        return DeploymentConfiguration(
            environment=environment,
            target=DeploymentTarget.AZURE,
            region=region,
            services={
                "api": ServiceConfiguration(
                    name="causal-eval-api",
                    image="causaleval.azurecr.io/causal-eval-bench:latest",
                    version="1.0.0",
                    port=8000,
                    replicas=2 if environment == DeploymentEnvironment.DEVELOPMENT else 3,
                    resources={
                        "vm_size": vm_size,
                        "disk_size": "30GB",
                        "disk_type": "Premium_LRS"
                    },
                    environment={
                        "ENVIRONMENT": environment.value,
                        "DATABASE_URL": "${DATABASE_URL}",
                        "REDIS_URL": "${REDIS_URL}",
                        "AZURE_SUBSCRIPTION_ID": "${AZURE_SUBSCRIPTION_ID}",
                        "LOG_LEVEL": "INFO"
                    },
                    health_check={
                        "type": "HTTP",
                        "path": "/health",
                        "port": 8000,
                        "interval": 30,
                        "timeout": 10
                    },
                    dependencies=["azure-database", "azure-cache"]
                )
            },
            database=DatabaseConfiguration(
                engine="postgresql",
                host="${AZURE_DATABASE_HOST}",
                port=5432,
                database="causal_eval",
                username="postgres",
                password_env_var="AZURE_DATABASE_PASSWORD",
                ssl_mode="require",
                pool_settings=settings['database'],
                migration_settings={
                    "auto_migrate": True,
                    "backup_retention_days": 7,
                    "geo_redundant_backup": environment == DeploymentEnvironment.PRODUCTION
                }
            ),
            cache=CacheConfiguration(
                engine="redis",
                host="${AZURE_CACHE_HOST}",
                port=6380,
                password_env_var="AZURE_CACHE_KEY",
                database=0,
                settings={
                    **settings['cache'],
                    "sku_name": "Basic" if environment == DeploymentEnvironment.DEVELOPMENT else "Standard",
                    "sku_capacity": 1 if environment == DeploymentEnvironment.DEVELOPMENT else 3
                }
            ),
            monitoring=self._get_monitoring_config(environment, azure=True),
            security=self._get_security_config(environment, azure=True),
            scaling={
                "auto_scaling": True,
                "scale_set_config": {
                    "min_capacity": 2,
                    "max_capacity": 10,
                    "default_capacity": 2
                },
                "scaling_policy": {
                    "metric": "cpu_percent",
                    "threshold": 70,
                    "direction": "Increase"
                }
            },
            networking={
                "virtual_network": {
                    "address_space": "10.0.0.0/16",
                    "subnets": ["api", "database", "cache"]
                },
                "load_balancer": {
                    "type": "Standard",
                    "sku": "Standard",
                    "frontend_ip_configuration": "public"
                }
            },
            storage={
                "storage_accounts": ["artifacts", "results", "backups"],
                "managed_disks": True,
                "backup_policy": "daily"
            }
        )
    
    def _generate_local_config(
        self, 
        environment: DeploymentEnvironment, 
        region: str,
        settings: Dict[str, Any]
    ) -> DeploymentConfiguration:
        """Generate local development configuration."""
        
        return DeploymentConfiguration(
            environment=environment,
            target=DeploymentTarget.LOCAL,
            region=region,
            services={
                "api": ServiceConfiguration(
                    name="causal-eval-api",
                    image="local",
                    version="dev",
                    port=8000,
                    replicas=1,
                    resources={
                        "memory": "2Gi",
                        "cpu": "1"
                    },
                    environment={
                        "ENVIRONMENT": environment.value,
                        "DATABASE_URL": "sqlite:///./causal_eval.db",
                        "REDIS_URL": "redis://localhost:6379/0",
                        "LOG_LEVEL": "DEBUG",
                        "RELOAD": "true"
                    },
                    health_check={
                        "enabled": False
                    },
                    dependencies=[]
                )
            },
            database=DatabaseConfiguration(
                engine="sqlite",
                host="localhost",
                port=0,
                database="causal_eval.db",
                username="",
                password_env_var="",
                ssl_mode="disable",
                pool_settings={"pool_size": 1, "max_overflow": 0},
                migration_settings={"auto_migrate": True}
            ),
            cache=CacheConfiguration(
                engine="memory",
                host="localhost",
                port=0,
                password_env_var=None,
                database=0,
                settings={"max_size": 100}
            ),
            monitoring=self._get_monitoring_config(environment, local=True),
            security=self._get_security_config(environment, local=True),
            scaling={"auto_scaling": False},
            networking={"enable_cors": True, "cors_origins": ["*"]},
            storage={"local_storage": True, "backup_enabled": False}
        )
    
    def _get_monitoring_config(self, environment: DeploymentEnvironment, **kwargs) -> MonitoringConfiguration:
        """Get monitoring configuration."""
        
        is_production = environment == DeploymentEnvironment.PRODUCTION
        
        return MonitoringConfiguration(
            metrics_enabled=True,
            metrics_port=9090,
            logging_level="INFO" if is_production else "DEBUG",
            tracing_enabled=is_production,
            health_check_interval=30,
            prometheus_config={
                "scrape_interval": "15s",
                "retention": "30d" if is_production else "7d",
                "external_labels": {"environment": environment.value}
            },
            grafana_config={
                "enable_dashboards": True,
                "alert_manager": is_production,
                "datasources": ["prometheus", "loki"]
            }
        )
    
    def _get_security_config(self, environment: DeploymentEnvironment, **kwargs) -> SecurityConfiguration:
        """Get security configuration."""
        
        is_production = environment == DeploymentEnvironment.PRODUCTION
        is_local = kwargs.get('local', False)
        
        return SecurityConfiguration(
            enable_https=is_production and not is_local,
            ssl_cert_path="/etc/ssl/certs/server.crt" if is_production else None,
            ssl_key_path="/etc/ssl/private/server.key" if is_production else None,
            enable_cors=True,
            cors_origins=["*"] if not is_production else ["https://causal-eval.com"],
            api_rate_limiting={
                "enabled": is_production,
                "requests_per_minute": 100 if is_production else 1000,
                "burst_size": 20
            },
            authentication={
                "enabled": is_production,
                "method": "jwt",
                "token_expiry": 3600
            },
            data_encryption={
                "encrypt_at_rest": is_production,
                "encrypt_in_transit": is_production,
                "key_rotation": is_production
            }
        )


class DeploymentManager:
    """Manages deployment configurations and operations."""
    
    def __init__(self, config_generator: Optional[DeploymentConfigurationGenerator] = None):
        self.config_generator = config_generator or DeploymentConfigurationGenerator()
    
    def export_configuration(
        self, 
        config: DeploymentConfiguration, 
        format: str = "yaml",
        output_path: Optional[Path] = None
    ) -> Union[str, Dict[str, Any]]:
        """Export configuration to various formats."""
        
        config_dict = asdict(config)
        
        if format.lower() == "yaml":
            yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2)
            if output_path:
                output_path.write_text(yaml_content)
            return yaml_content
        
        elif format.lower() == "json":
            json_content = json.dumps(config_dict, indent=2)
            if output_path:
                output_path.write_text(json_content)
            return json_content
        
        elif format.lower() == "docker-compose":
            return self._export_docker_compose(config, output_path)
        
        elif format.lower() == "kubernetes":
            return self._export_kubernetes_manifests(config, output_path)
        
        else:
            return config_dict
    
    def _export_docker_compose(self, config: DeploymentConfiguration, output_path: Optional[Path] = None) -> str:
        """Export as docker-compose.yml."""
        
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {"causal-eval-network": {"driver": "bridge"}},
            "volumes": {}
        }
        
        # Add services
        for service_name, service in config.services.items():
            compose_config["services"][service_name] = {
                "image": service.image,
                "ports": [f"{service.port}:{service.port}"],
                "environment": service.environment,
                "depends_on": service.dependencies,
                "networks": ["causal-eval-network"],
                "restart": "unless-stopped"
            }
            
            if service.health_check and service.health_check.get("test"):
                compose_config["services"][service_name]["healthcheck"] = service.health_check
        
        # Add database
        if config.database.engine == "postgresql":
            compose_config["services"]["db"] = {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_DB": config.database.database,
                    "POSTGRES_USER": config.database.username,
                    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}"
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "networks": ["causal-eval-network"],
                "restart": "unless-stopped"
            }
            compose_config["volumes"]["postgres_data"] = {}
        
        # Add cache
        if config.cache.engine == "redis":
            compose_config["services"]["redis"] = {
                "image": "redis:6-alpine",
                "command": "redis-server --appendonly yes",
                "volumes": ["redis_data:/data"],
                "networks": ["causal-eval-network"],
                "restart": "unless-stopped"
            }
            compose_config["volumes"]["redis_data"] = {}
        
        yaml_content = yaml.dump(compose_config, default_flow_style=False, indent=2)
        
        if output_path:
            output_path.write_text(yaml_content)
        
        return yaml_content
    
    def _export_kubernetes_manifests(self, config: DeploymentConfiguration, output_path: Optional[Path] = None) -> Dict[str, str]:
        """Export as Kubernetes manifests."""
        
        manifests = {}
        
        # Deployment manifest
        for service_name, service in config.services.items():
            deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"{service_name}-deployment",
                    "labels": {"app": service_name}
                },
                "spec": {
                    "replicas": service.replicas,
                    "selector": {"matchLabels": {"app": service_name}},
                    "template": {
                        "metadata": {"labels": {"app": service_name}},
                        "spec": {
                            "containers": [{
                                "name": service_name,
                                "image": service.image,
                                "ports": [{"containerPort": service.port}],
                                "env": [{"name": k, "value": v} for k, v in service.environment.items()],
                                "resources": service.resources
                            }]
                        }
                    }
                }
            }
            
            if service.health_check:
                deployment["spec"]["template"]["spec"]["containers"][0]["livenessProbe"] = service.health_check
                deployment["spec"]["template"]["spec"]["containers"][0]["readinessProbe"] = service.health_check
            
            manifests[f"{service_name}-deployment.yaml"] = yaml.dump(deployment, default_flow_style=False, indent=2)
            
            # Service manifest
            svc = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{service_name}-service",
                    "labels": {"app": service_name}
                },
                "spec": {
                    "selector": {"app": service_name},
                    "ports": [{"port": service.port, "targetPort": service.port}],
                    "type": "ClusterIP"
                }
            }
            
            manifests[f"{service_name}-service.yaml"] = yaml.dump(svc, default_flow_style=False, indent=2)
        
        if output_path and output_path.is_dir():
            for filename, content in manifests.items():
                (output_path / filename).write_text(content)
        
        return manifests
    
    def validate_configuration(self, config: DeploymentConfiguration) -> Dict[str, Any]:
        """Validate deployment configuration."""
        
        issues = []
        warnings = []
        
        # Check required fields
        if not config.services:
            issues.append("No services defined")
        
        # Validate service configurations
        for service_name, service in config.services.items():
            if not service.image:
                issues.append(f"Service {service_name} missing image")
            
            if service.port <= 0 or service.port > 65535:
                issues.append(f"Service {service_name} has invalid port: {service.port}")
            
            if service.replicas <= 0:
                warnings.append(f"Service {service_name} has no replicas")
        
        # Validate database configuration
        if not config.database.host:
            issues.append("Database host not specified")
        
        # Validate security for production
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.security.enable_https:
                warnings.append("HTTPS not enabled for production")
            
            if not config.security.authentication.get("enabled"):
                warnings.append("Authentication not enabled for production")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "score": max(0, 100 - len(issues) * 20 - len(warnings) * 5)
        }


# Export main classes
__all__ = [
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