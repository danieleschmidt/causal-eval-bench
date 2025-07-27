#!/usr/bin/env python3
"""Monitoring and observability setup for Causal Eval Bench."""

import json
import os
from pathlib import Path
from typing import Dict, Any

def create_health_endpoint_config() -> Dict[str, Any]:
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

def main():
    """Setup monitoring configuration."""
    config = create_health_endpoint_config()
    
    # Create monitoring config directory
    monitoring_dir = Path("config/monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    # Write health check config
    with open(monitoring_dir / "health_checks.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Monitoring configuration created")

if __name__ == "__main__":
    main()
