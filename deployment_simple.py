#\!/usr/bin/env python3
"""Simplified deployment readiness checker."""

import sys
import os
from pathlib import Path
from enum import Enum

class ReadinessLevel(Enum):
    NOT_READY = "not_ready" 
    BASIC = "basic"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"

def check_deployment_readiness():
    """Run simplified deployment readiness checks."""
    print("🚀 Deployment Readiness Assessment")
    print("=" * 45)
    
    project_root = Path(".")
    checks = []
    
    # Core checks
    dockerfile = (project_root / "Dockerfile").exists()
    readme = (project_root / "README.md").exists()
    pyproject = (project_root / "pyproject.toml").exists()
    tests = len(list(project_root.rglob("test_*.py"))) > 0
    health = len(list(project_root.rglob("*health*"))) > 0
    makefile = (project_root / "Makefile").exists()
    
    checks = [
        ("Dockerfile", dockerfile, True),
        ("README.md", readme, True), 
        ("pyproject.toml", pyproject, True),
        ("Test files", tests, True),
        ("Health endpoints", health, True),
        ("Build automation", makefile, True)
    ]
    
    print("📋 Deployment Checks:")
    passed = 0
    total = len(checks)
    
    for name, status, required in checks:
        icon = "✅" if status else "❌"
        req_text = " (Required)" if required else ""
        print(f"  {icon} {name}{req_text}: {'PASS' if status else 'FAIL'}")
        if status:
            passed += 1
    
    score = (passed / total) * 100
    
    print(f"
📊 Results: {passed}/{total} checks passed ({score:.1f}%)")
    
    if score >= 80:
        print("🚀 DEPLOYMENT STATUS: PRODUCTION READY")
        return True
    elif score >= 60:
        print("🟡 DEPLOYMENT STATUS: BASIC READY") 
        return True
    else:
        print("❌ DEPLOYMENT STATUS: NOT READY")
        return False

if __name__ == "__main__":
    success = check_deployment_readiness()
    sys.exit(0 if success else 1)
