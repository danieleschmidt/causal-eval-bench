#!/usr/bin/env python3
"""
Monitoring and observability setup script for Causal Eval Bench.
Configures Prometheus, Grafana, and alerting.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import requests
import yaml


class MonitoringManager:
    """Manages monitoring stack setup and configuration."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.prometheus_url = "http://localhost:9090"
        self.grafana_url = "http://localhost:3000"
        self.grafana_credentials = ("admin", "admin123")
        
    def check_service_health(self, service_url: str, timeout: int = 30) -> bool:
        """Check if a service is healthy and responding."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{service_url}/api/v1/query", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False
    
    def wait_for_prometheus(self, timeout: int = 60) -> bool:
        """Wait for Prometheus to be ready."""
        click.echo("Waiting for Prometheus to be ready...")
        if self.check_service_health(self.prometheus_url, timeout):
            click.echo("✅ Prometheus is ready")
            return True
        else:
            click.echo("❌ Prometheus failed to start within timeout")
            return False
    
    def wait_for_grafana(self, timeout: int = 60) -> bool:
        """Wait for Grafana to be ready."""
        click.echo("Waiting for Grafana to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.grafana_url}/api/health", timeout=5)
                if response.status_code == 200:
                    click.echo("✅ Grafana is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        click.echo("❌ Grafana failed to start within timeout")
        return False
    
    def configure_grafana_datasource(self) -> bool:
        """Configure Prometheus datasource in Grafana."""
        click.echo("Configuring Grafana datasource...")
        
        datasource_config = {
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://prometheus:9090",
            "access": "proxy",
            "isDefault": True,
            "jsonData": {
                "timeInterval": "15s",
                "queryTimeout": "60s",
                "httpMethod": "POST"
            }
        }
        
        try:
            response = requests.post(
                f"{self.grafana_url}/api/datasources",
                json=datasource_config,
                auth=self.grafana_credentials,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 409]:  # 409 = already exists
                click.echo("✅ Grafana datasource configured")
                return True
            else:
                click.echo(f"❌ Failed to configure datasource: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            click.echo(f"❌ Error configuring datasource: {e}")
            return False
    
    def import_grafana_dashboards(self) -> bool:
        """Import dashboards into Grafana."""
        click.echo("Importing Grafana dashboards...")
        
        dashboards_dir = self.project_root / "docker" / "grafana" / "dashboards"
        if not dashboards_dir.exists():
            click.echo("❌ Dashboards directory not found")
            return False
        
        success_count = 0
        total_count = 0
        
        for dashboard_file in dashboards_dir.glob("*.json"):
            total_count += 1
            try:
                with open(dashboard_file, 'r') as f:
                    dashboard_data = json.load(f)
                
                import_payload = {
                    "dashboard": dashboard_data["dashboard"],
                    "overwrite": True,
                    "inputs": [],
                    "folderId": 0
                }
                
                response = requests.post(
                    f"{self.grafana_url}/api/dashboards/db",
                    json=import_payload,
                    auth=self.grafana_credentials,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    click.echo(f"✅ Imported dashboard: {dashboard_file.name}")
                    success_count += 1
                else:
                    click.echo(f"❌ Failed to import {dashboard_file.name}: {response.text}")
                    
            except Exception as e:
                click.echo(f"❌ Error importing {dashboard_file.name}: {e}")
        
        click.echo(f"Imported {success_count}/{total_count} dashboards")
        return success_count == total_count
    
    def create_health_endpoint_config(self) -> Dict[str, any]:
        """Create health check endpoint configuration."""
        return {
            "health_checks": {
                "database": {
                    "enabled": True,
                    "timeout": 5,
                    "query": "SELECT 1"
                },
                "redis": {
                    "enabled": True,
                    "timeout": 3,
                    "command": "ping"
                },
                "external_apis": {
                    "enabled": True,
                    "timeout": 10,
                    "endpoints": [
                        "https://api.openai.com/v1/models",
                        "https://api.anthropic.com/v1/messages"
                    ]
                }
            },
            "metrics": {
                "prometheus": {
                    "enabled": True,
                    "port": 9090,
                    "path": "/metrics"
                },
                "custom_metrics": [
                    "evaluation_requests_total",
                    "evaluation_duration_seconds",
                    "question_generation_total",
                    "model_performance_score",
                    "api_errors_total"
                ]
            }
        }


@click.group()
def cli():
    """Monitoring setup and management for Causal Eval Bench."""
    pass


@cli.command()
@click.option('--timeout', default=120, help='Timeout in seconds')
def setup(timeout):
    """Setup complete monitoring stack."""
    project_root = Path(__file__).parent.parent
    manager = MonitoringManager(project_root)
    
    click.echo("🔧 Setting up monitoring stack...")
    
    # Create monitoring config
    config = manager.create_health_endpoint_config()
    
    # Create monitoring config directory
    monitoring_dir = project_root / "config" / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    # Write health check config
    with open(monitoring_dir / "health_checks.json", "w") as f:
        json.dump(config, f, indent=2)
    
    click.echo("✅ Monitoring configuration created")


@cli.command()
def validate():
    """Validate monitoring configuration."""
    project_root = Path(__file__).parent.parent
    
    click.echo("🔍 Validating monitoring configuration...")
    
    success = True
    
    # Validate dashboard files
    dashboards_dir = project_root / "docker" / "grafana" / "dashboards"
    if dashboards_dir.exists():
        for dashboard_file in dashboards_dir.glob("*.json"):
            try:
                with open(dashboard_file, 'r') as f:
                    json.load(f)
                click.echo(f"✅ Dashboard valid: {dashboard_file.name}")
            except json.JSONDecodeError as e:
                click.echo(f"❌ Dashboard invalid: {dashboard_file.name} - {e}")
                success = False
    
    if success:
        click.echo("\n✅ All monitoring configurations are valid")
    else:
        click.echo("\n❌ Some configurations are invalid")
        sys.exit(1)


if __name__ == "__main__":
    cli()
