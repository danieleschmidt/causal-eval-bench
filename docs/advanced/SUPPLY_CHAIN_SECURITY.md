# Supply Chain Security for Causal Eval Bench

This guide covers comprehensive supply chain security practices for Causal Eval Bench, ensuring the integrity and security of dependencies, build processes, and distribution channels.

## 🔒 Supply Chain Security Overview

Supply chain security for evaluation frameworks involves protecting against:

- **Dependency Confusion**: Malicious packages with similar names
- **Typosquatting**: Packages exploiting common typos
- **Compromised Dependencies**: Legitimate packages that become compromised
- **Build System Attacks**: Malicious code injection during builds
- **Distribution Channel Attacks**: Compromised package repositories

## 📦 Dependency Management Security

### Secure Dependency Resolution

Implement secure dependency management practices:

```toml
# pyproject.toml - Enhanced dependency security
[tool.poetry.dependencies]
python = "^3.9"

# Pin exact versions for critical security dependencies
fastapi = "0.104.0"  # Exact version
uvicorn = "0.24.0"   # Exact version
pydantic = "2.5.0"   # Exact version

# Use version ranges for less critical dependencies
httpx = "^0.25.0"
pandas = "^2.1.0"

[tool.poetry.group.security.dependencies]
# Security scanning tools
bandit = "^1.7.5"
safety = "^2.3.0" 
pip-audit = "^2.6.0"
cyclonedx-bom = "^4.0.0"  # SBOM generation

[tool.poetry.group.verification.dependencies]
# Package verification tools
sigstore = "^2.0.0"
in-toto = "^2.0.0"
```

### Dependency Verification

Implement comprehensive dependency verification:

```python
# scripts/verify_dependencies.py
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

class DependencyVerifier:
    def __init__(self, lockfile_path: str = "poetry.lock"):
        self.lockfile_path = Path(lockfile_path)
        self.known_hashes = self.load_known_hashes()
        self.trusted_publishers = self.load_trusted_publishers()
    
    def verify_all_dependencies(self) -> Dict[str, bool]:
        """Verify all project dependencies."""
        
        verification_results = {}
        dependencies = self.parse_lockfile()
        
        for dep_name, dep_info in dependencies.items():
            results = {
                "hash_verification": self.verify_package_hash(dep_name, dep_info),
                "signature_verification": self.verify_package_signature(dep_name, dep_info),
                "publisher_verification": self.verify_publisher(dep_name, dep_info),
                "vulnerability_scan": self.scan_for_vulnerabilities(dep_name, dep_info)
            }
            
            verification_results[dep_name] = all(results.values())
            
            if not verification_results[dep_name]:
                self.log_verification_failure(dep_name, results)
        
        return verification_results
    
    def verify_package_hash(self, package_name: str, package_info: dict) -> bool:
        """Verify package integrity using hashes."""
        
        expected_hash = package_info.get("hash")
        if not expected_hash:
            self.log_warning(f"No hash available for {package_name}")
            return False
        
        # Download and verify package hash
        actual_hash = self.compute_package_hash(package_name, package_info["version"])
        
        if actual_hash != expected_hash:
            self.log_security_alert(
                f"Hash mismatch for {package_name}: expected {expected_hash}, got {actual_hash}"
            )
            return False
        
        return True
    
    def verify_package_signature(self, package_name: str, package_info: dict) -> bool:
        """Verify package signatures using Sigstore."""
        
        try:
            # Use sigstore to verify package signature
            result = subprocess.run([
                "python", "-m", "sigstore", "verify",
                "--package", f"{package_name}=={package_info['version']}",
                "--trusted-root", "pypi"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log_warning(f"Signature verification failed for {package_name}")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Signature verification error for {package_name}: {e}")
            return False
    
    def verify_publisher(self, package_name: str, package_info: dict) -> bool:
        """Verify package publisher against trusted list."""
        
        publisher_info = self.get_publisher_info(package_name)
        
        if package_name in self.trusted_publishers:
            expected_publisher = self.trusted_publishers[package_name]
            actual_publisher = publisher_info.get("publisher")
            
            if actual_publisher != expected_publisher:
                self.log_security_alert(
                    f"Publisher mismatch for {package_name}: "
                    f"expected {expected_publisher}, got {actual_publisher}"
                )
                return False
        
        return True
    
    def scan_for_vulnerabilities(self, package_name: str, package_info: dict) -> bool:
        """Scan package for known vulnerabilities."""
        
        try:
            # Use safety to check for vulnerabilities
            result = subprocess.run([
                "safety", "check", "--json", "--package", 
                f"{package_name}=={package_info['version']}"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                vulnerabilities = json.loads(result.stdout)
                self.log_security_alert(
                    f"Vulnerabilities found in {package_name}: {vulnerabilities}"
                )
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Vulnerability scan error for {package_name}: {e}")
            return False

# Enhanced dependency pinning strategy
class SecureDependencyManager:
    def __init__(self):
        self.security_policy = self.load_security_policy()
    
    def load_security_policy(self) -> dict:
        """Load dependency security policy."""
        return {
            "critical_dependencies": [
                "fastapi", "uvicorn", "pydantic", "sqlalchemy",
                "cryptography", "jwt", "passlib"
            ],
            "pin_exact_versions": True,
            "auto_update_security": True,
            "vulnerability_threshold": "medium",
            "trusted_publishers": {
                "fastapi": "tiangolo",
                "pydantic": "pydantic",
                "sqlalchemy": "sqlalchemy"
            }
        }
    
    def generate_secure_requirements(self) -> str:
        """Generate requirements with security constraints."""
        
        requirements = []
        dependencies = self.get_project_dependencies()
        
        for dep_name, dep_version in dependencies.items():
            if dep_name in self.security_policy["critical_dependencies"]:
                # Pin exact versions for critical dependencies
                requirements.append(f"{dep_name}=={dep_version}")
            else:
                # Use compatible version ranges for others
                requirements.append(f"{dep_name}~={dep_version}")
        
        return "\n".join(requirements)
```

### Software Bill of Materials (SBOM)

Generate and maintain comprehensive SBOMs:

```python
# scripts/generate_sbom.py
from cyclonedx.builder import build_sbom_from_requirements
from cyclonedx.output import make_outputter
from cyclonedx.schema import SchemaVersion
import json
from datetime import datetime

class SBOMGenerator:
    def __init__(self):
        self.component_name = "causal-eval-bench"
        self.component_version = self.get_version()
    
    def generate_sbom(self, output_format: str = "json") -> str:
        """Generate comprehensive SBOM."""
        
        # Build SBOM from requirements
        sbom = build_sbom_from_requirements(
            requirements_file="requirements.txt",
            component_name=self.component_name,
            component_version=self.component_version
        )
        
        # Add custom metadata
        sbom.metadata.timestamp = datetime.utcnow()
        sbom.metadata.tools.add_tool("causal-eval-sbom-generator", "1.0.0")
        
        # Add vulnerability data
        self.add_vulnerability_data(sbom)
        
        # Add license information
        self.add_license_information(sbom)
        
        # Add supplier information
        self.add_supplier_information(sbom)
        
        # Output SBOM
        outputter = make_outputter(
            output_format=output_format,
            schema_version=SchemaVersion.V1_4
        )
        
        return outputter.output_as_string(sbom)
    
    def add_vulnerability_data(self, sbom):
        """Add vulnerability information to SBOM."""
        
        vulnerability_scanner = VulnerabilityScanner()
        
        for component in sbom.components:
            vulnerabilities = vulnerability_scanner.scan_component(
                component.name, component.version
            )
            
            for vuln in vulnerabilities:
                sbom.vulnerabilities.add_vulnerability(
                    id=vuln.id,
                    source=vuln.source,
                    description=vuln.description,
                    severity=vuln.severity,
                    affected_components=[component]
                )
    
    def add_license_information(self, sbom):
        """Add comprehensive license information."""
        
        license_scanner = LicenseScanner()
        
        for component in sbom.components:
            license_info = license_scanner.get_license_info(
                component.name, component.version
            )
            
            component.licenses.add_license_choice(license_info)
    
    def add_supplier_information(self, sbom):
        """Add supplier/publisher information."""
        
        supplier_verifier = SupplierVerifier()
        
        for component in sbom.components:
            supplier_info = supplier_verifier.get_supplier_info(
                component.name
            )
            
            component.supplier = supplier_info
    
    def validate_sbom(self, sbom_content: str) -> bool:
        """Validate SBOM completeness and accuracy."""
        
        try:
            sbom_data = json.loads(sbom_content)
            
            validation_checks = [
                self.validate_required_fields(sbom_data),
                self.validate_component_integrity(sbom_data),
                self.validate_vulnerability_data(sbom_data),
                self.validate_license_compliance(sbom_data)
            ]
            
            return all(validation_checks)
            
        except Exception as e:
            self.log_error(f"SBOM validation failed: {e}")
            return False

# Automated SBOM monitoring
class SBOMMonitor:
    def __init__(self):
        self.baseline_sbom = self.load_baseline_sbom()
    
    def monitor_sbom_changes(self, new_sbom: str) -> dict:
        """Monitor changes in SBOM."""
        
        changes = {
            "added_components": [],
            "removed_components": [],
            "updated_components": [],
            "new_vulnerabilities": [],
            "resolved_vulnerabilities": []
        }
        
        new_sbom_data = json.loads(new_sbom)
        baseline_data = json.loads(self.baseline_sbom)
        
        # Compare components
        baseline_components = {c["name"]: c for c in baseline_data["components"]}
        new_components = {c["name"]: c for c in new_sbom_data["components"]}
        
        # Identify changes
        for name, component in new_components.items():
            if name not in baseline_components:
                changes["added_components"].append(component)
            elif baseline_components[name]["version"] != component["version"]:
                changes["updated_components"].append({
                    "name": name,
                    "old_version": baseline_components[name]["version"],
                    "new_version": component["version"]
                })
        
        for name in baseline_components:
            if name not in new_components:
                changes["removed_components"].append(baseline_components[name])
        
        # Compare vulnerabilities
        self.compare_vulnerabilities(baseline_data, new_sbom_data, changes)
        
        return changes
    
    def generate_change_report(self, changes: dict) -> str:
        """Generate human-readable change report."""
        
        report = ["SBOM Change Report", "=" * 50, ""]
        
        if changes["added_components"]:
            report.append("New Components:")
            for component in changes["added_components"]:
                report.append(f"  + {component['name']} {component['version']}")
            report.append("")
        
        if changes["updated_components"]:
            report.append("Updated Components:")
            for component in changes["updated_components"]:
                report.append(
                    f"  ~ {component['name']}: "
                    f"{component['old_version']} → {component['new_version']}"
                )
            report.append("")
        
        if changes["new_vulnerabilities"]:
            report.append("New Vulnerabilities:")
            for vuln in changes["new_vulnerabilities"]:
                report.append(f"  ! {vuln['id']}: {vuln['severity']}")
            report.append("")
        
        return "\n".join(report)
```

## 🔐 Build Security

### Secure Build Pipeline

Implement secure build practices:

```yaml
# .github/workflows/secure-build.yml
name: Secure Build Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  dependency-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Verify Poetry installation
        run: poetry --version
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pypoetry
          key: ${{ runner.os }}-poetry-${{ hashFiles('**/poetry.lock') }}
      
      - name: Install dependencies
        run: poetry install --no-dev
      
      - name: Run dependency security scan
        run: |
          poetry run safety check --json --output safety-report.json
          poetry run bandit -r causal_eval/ -f json -o bandit-report.json
          poetry run pip-audit --format=json --output=audit-report.json
      
      - name: Generate SBOM
        run: |
          poetry run cyclonedx-py -o sbom.json
      
      - name: Verify dependency signatures
        run: |
          poetry run python scripts/verify_dependencies.py
      
      - name: Upload security artifacts
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json
            audit-report.json
            sbom.json

  build-verification:
    runs-on: ubuntu-latest
    needs: dependency-security
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      
      - name: Build package
        run: |
          poetry build
      
      - name: Verify build artifacts
        run: |
          # Verify wheel and tarball integrity
          python scripts/verify_build_artifacts.py dist/
      
      - name: Sign build artifacts
        env:
          SIGSTORE_ID_TOKEN: ${{ secrets.SIGSTORE_ID_TOKEN }}
        run: |
          # Sign using Sigstore
          python -m sigstore sign dist/*
      
      - name: Upload signed artifacts
        uses: actions/upload-artifact@v3
        with:
          name: signed-packages
          path: |
            dist/
            *.sig

  container-security:
    runs-on: ubuntu-latest
    needs: build-verification
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: |
          docker build -t causal-eval-bench:${{ github.sha }} .
      
      - name: Scan container for vulnerabilities
        uses: anchore/scan-action@v3
        with:
          image: causal-eval-bench:${{ github.sha }}
          fail-build: true
          severity-cutoff: medium
      
      - name: Generate container SBOM
        run: |
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            anchore/syft:latest causal-eval-bench:${{ github.sha }} \
            -o json > container-sbom.json
      
      - name: Sign container image
        env:
          COSIGN_PRIVATE_KEY: ${{ secrets.COSIGN_PRIVATE_KEY }}
          COSIGN_PASSWORD: ${{ secrets.COSIGN_PASSWORD }}
        run: |
          cosign sign --key env://COSIGN_PRIVATE_KEY \
            causal-eval-bench:${{ github.sha }}
```

### Build Reproducibility

Ensure reproducible builds:

```python
# scripts/reproducible_build.py
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

class ReproducibleBuildValidator:
    def __init__(self):
        self.build_config = self.load_build_config()
    
    def validate_reproducibility(self, artifact_path: str) -> bool:
        """Validate build reproducibility."""
        
        # Build artifact multiple times
        build_hashes = []
        
        for attempt in range(3):
            # Clean build environment
            self.clean_build_environment()
            
            # Rebuild artifact
            self.rebuild_artifact()
            
            # Calculate hash
            artifact_hash = self.calculate_artifact_hash(artifact_path)
            build_hashes.append(artifact_hash)
        
        # Verify all builds produce identical results
        if len(set(build_hashes)) == 1:
            self.log_success("Build is reproducible")
            return True
        else:
            self.log_error(f"Build not reproducible. Hashes: {build_hashes}")
            return False
    
    def clean_build_environment(self):
        """Clean build environment to ensure consistency."""
        
        # Remove build artifacts
        build_dirs = ["build/", "dist/", "*.egg-info/"]
        for pattern in build_dirs:
            subprocess.run(["rm", "-rf"] + glob.glob(pattern))
        
        # Clear Python cache
        subprocess.run(["find", ".", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"])
        subprocess.run(["find", ".", "-name", "*.pyc", "-delete"])
        
        # Reset environment variables that might affect build
        build_env = os.environ.copy()
        build_env.pop("SOURCE_DATE_EPOCH", None)
        build_env["SOURCE_DATE_EPOCH"] = "1609459200"  # Fixed timestamp
        
        return build_env
    
    def rebuild_artifact(self):
        """Rebuild artifact with controlled environment."""
        
        env = self.clean_build_environment()
        
        # Build with fixed timestamp
        subprocess.run(
            ["poetry", "build"],
            env=env,
            check=True
        )
    
    def calculate_artifact_hash(self, artifact_path: str) -> str:
        """Calculate SHA256 hash of build artifact."""
        
        hasher = hashlib.sha256()
        
        with open(artifact_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def generate_build_provenance(self) -> dict:
        """Generate build provenance information."""
        
        return {
            "builder": {
                "id": "github-actions",
                "version": os.environ.get("GITHUB_SHA", "unknown")
            },
            "buildType": "poetry-build",
            "invocation": {
                "configSource": {
                    "uri": f"git+{os.environ.get('GITHUB_REPOSITORY', '')}",
                    "digest": {"sha1": os.environ.get("GITHUB_SHA", "")}
                }
            },
            "metadata": {
                "buildStartedOn": self.get_build_start_time(),
                "buildFinishedOn": self.get_build_finish_time(),
                "completeness": {
                    "parameters": True,
                    "environment": True,
                    "materials": True
                },
                "reproducible": self.validate_reproducibility("dist/")
            },
            "materials": self.get_build_materials()
        }
```

## 🛡️ Runtime Security

### Container Security

Implement comprehensive container security:

```dockerfile
# Dockerfile - Security-hardened
FROM python:3.11-slim as builder

# Create non-root user
RUN groupadd -r causaleval && useradd -r -g causaleval causaleval

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies in virtual environment
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

# Production stage
FROM python:3.11-slim as production

# Security hardening
RUN groupadd -r causaleval && useradd -r -g causaleval causaleval

# Install security updates only
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=causaleval:causaleval . /app
WORKDIR /app

# Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -exec rm -rf {} + && \
    rm -rf /app/tests /app/docs

# Set security options
USER causaleval
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Use non-root user
CMD ["uvicorn", "causal_eval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Runtime Attestation

Implement runtime security attestation:

```python
# causal_eval/security/attestation.py
from in_toto import runlib
from in_toto.models.link import Link
from in_toto.models.metadata import Metablock
import json
import os

class RuntimeAttestationManager:
    def __init__(self):
        self.attestation_key = self.load_attestation_key()
        self.policy = self.load_attestation_policy()
    
    def generate_deployment_attestation(self, deployment_info: dict) -> str:
        """Generate deployment attestation."""
        
        link = Link(
            name="deploy-causal-eval-bench",
            command=deployment_info["command"],
            byproducts=deployment_info["artifacts"],
            environment=self.get_deployment_environment()
        )
        
        # Sign the link metadata
        metablock = Metablock(signed=link)
        metablock.sign(self.attestation_key)
        
        return json.dumps(metablock.to_dict())
    
    def verify_deployment_attestation(self, attestation: str) -> bool:
        """Verify deployment attestation."""
        
        try:
            metablock = Metablock.from_dict(json.loads(attestation))
            
            # Verify signature
            if not metablock.verify_signature(self.attestation_key):
                return False
            
            # Verify attestation policy compliance
            return self.verify_policy_compliance(metablock.signed)
            
        except Exception as e:
            self.log_error(f"Attestation verification failed: {e}")
            return False
    
    def generate_runtime_attestation(self) -> str:
        """Generate runtime environment attestation."""
        
        runtime_info = {
            "container_id": os.environ.get("HOSTNAME"),
            "image_digest": self.get_container_image_digest(),
            "running_processes": self.get_running_processes(),
            "network_connections": self.get_network_connections(),
            "file_integrity": self.verify_file_integrity(),
            "dependency_versions": self.get_dependency_versions()
        }
        
        link = Link(
            name="runtime-attestation",
            environment=runtime_info
        )
        
        metablock = Metablock(signed=link)
        metablock.sign(self.attestation_key)
        
        return json.dumps(metablock.to_dict())
    
    def continuous_attestation_monitoring(self):
        """Monitor runtime environment continuously."""
        
        baseline_attestation = self.generate_runtime_attestation()
        
        while True:
            current_attestation = self.generate_runtime_attestation()
            
            if not self.compare_attestations(baseline_attestation, current_attestation):
                self.alert_security_team("Runtime attestation mismatch detected")
            
            time.sleep(300)  # Check every 5 minutes

# SLSA (Supply Chain Levels for Software Artifacts) compliance
class SLSACompliance:
    def __init__(self):
        self.slsa_level = 3  # Target SLSA level
    
    def generate_slsa_provenance(self, build_info: dict) -> dict:
        """Generate SLSA provenance information."""
        
        provenance = {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicateType": "https://slsa.dev/provenance/v0.2",
            "subject": [
                {
                    "name": build_info["artifact_name"],
                    "digest": {
                        "sha256": build_info["artifact_hash"]
                    }
                }
            ],
            "predicate": {
                "builder": {
                    "id": "https://github.com/actions/runner"
                },
                "buildType": "https://github.com/actions/workflow",
                "invocation": {
                    "configSource": {
                        "uri": build_info["repository_uri"],
                        "digest": {
                            "sha1": build_info["commit_sha"]
                        }
                    }
                },
                "metadata": {
                    "buildInvocationId": build_info["build_id"],
                    "buildStartedOn": build_info["build_start_time"],
                    "buildFinishedOn": build_info["build_finish_time"],
                    "completeness": {
                        "parameters": True,
                        "environment": True,
                        "materials": True
                    },
                    "reproducible": build_info["reproducible"]
                },
                "materials": build_info["materials"]
            }
        }
        
        return provenance
    
    def validate_slsa_compliance(self, provenance: dict) -> dict:
        """Validate SLSA compliance level."""
        
        compliance_checks = {
            "source_tracked": self.check_source_tracking(provenance),
            "build_service": self.check_build_service(provenance),
            "build_integrity": self.check_build_integrity(provenance),
            "provenance_authenticated": self.check_provenance_auth(provenance),
            "provenance_service_generated": self.check_service_generated(provenance)
        }
        
        compliance_level = self.determine_compliance_level(compliance_checks)
        
        return {
            "compliance_level": compliance_level,
            "checks": compliance_checks,
            "meets_target": compliance_level >= self.slsa_level
        }
```

## 🚨 Security Monitoring

### Supply Chain Monitoring

Implement continuous supply chain monitoring:

```python
# causal_eval/security/monitoring.py
from typing import Dict, List, Optional
import requests
import json
from datetime import datetime, timedelta

class SupplyChainMonitor:
    def __init__(self):
        self.monitoring_config = self.load_monitoring_config()
        self.alert_channels = self.setup_alert_channels()
    
    def monitor_dependencies(self):
        """Continuously monitor dependencies for security issues."""
        
        dependencies = self.get_project_dependencies()
        
        for dep_name, dep_version in dependencies.items():
            # Check for new vulnerabilities
            vulnerabilities = self.check_vulnerabilities(dep_name, dep_version)
            if vulnerabilities:
                self.alert_new_vulnerabilities(dep_name, vulnerabilities)
            
            # Check for package updates
            updates = self.check_for_updates(dep_name, dep_version)
            if updates and self.is_security_update(updates):
                self.alert_security_updates(dep_name, updates)
            
            # Monitor package integrity
            if not self.verify_package_integrity(dep_name, dep_version):
                self.alert_integrity_failure(dep_name, dep_version)
    
    def monitor_build_pipeline(self):
        """Monitor build pipeline for security issues."""
        
        build_history = self.get_build_history(limit=10)
        
        for build in build_history:
            # Check build integrity
            if not self.verify_build_integrity(build):
                self.alert_build_compromise(build)
            
            # Monitor for anomalous build behavior
            anomalies = self.detect_build_anomalies(build)
            if anomalies:
                self.alert_build_anomalies(build, anomalies)
    
    def monitor_distribution_channels(self):
        """Monitor distribution channels for integrity."""
        
        distribution_points = [
            "pypi.org",
            "github.com/releases",
            "docker.io"
        ]
        
        for channel in distribution_points:
            if not self.verify_distribution_integrity(channel):
                self.alert_distribution_compromise(channel)
    
    def generate_security_dashboard(self) -> dict:
        """Generate security monitoring dashboard data."""
        
        return {
            "dependency_security": {
                "vulnerable_packages": self.count_vulnerable_packages(),
                "outdated_packages": self.count_outdated_packages(),
                "security_updates_available": self.count_security_updates()
            },
            "build_security": {
                "build_integrity_score": self.calculate_build_integrity_score(),
                "recent_build_anomalies": self.get_recent_build_anomalies(),
                "attestation_compliance": self.check_attestation_compliance()
            },
            "distribution_security": {
                "channel_integrity": self.check_distribution_integrity(),
                "signature_verification": self.check_signature_verification(),
                "sbom_coverage": self.calculate_sbom_coverage()
            }
        }

class SecurityIncidentResponse:
    def __init__(self):
        self.incident_playbooks = self.load_incident_playbooks()
        self.response_teams = self.load_response_teams()
    
    def handle_dependency_compromise(self, package_name: str, threat_info: dict):
        """Handle compromised dependency incident."""
        
        # Immediate response
        self.quarantine_package(package_name)
        self.stop_deployments()
        self.alert_security_team("CRITICAL", f"Dependency compromise: {package_name}")
        
        # Assessment phase
        impact_assessment = self.assess_compromise_impact(package_name, threat_info)
        
        # Response actions based on impact
        if impact_assessment["severity"] == "critical":
            self.execute_emergency_response(package_name, impact_assessment)
        else:
            self.execute_standard_response(package_name, impact_assessment)
        
        # Recovery actions
        self.plan_recovery_actions(package_name, impact_assessment)
    
    def handle_build_compromise(self, build_id: str, compromise_indicators: list):
        """Handle build pipeline compromise."""
        
        # Immediate containment
        self.isolate_build_environment(build_id)
        self.revoke_build_artifacts(build_id)
        self.rotate_build_credentials()
        
        # Investigation
        investigation_report = self.investigate_build_compromise(
            build_id, compromise_indicators
        )
        
        # Remediation
        remediation_plan = self.create_remediation_plan(investigation_report)
        self.execute_remediation(remediation_plan)
        
        # Recovery
        self.restore_secure_build_environment()
        self.verify_build_integrity()
    
    def generate_incident_report(self, incident_id: str) -> dict:
        """Generate comprehensive incident report."""
        
        incident = self.get_incident(incident_id)
        
        return {
            "incident_summary": incident.summary,
            "timeline": incident.timeline,
            "impact_assessment": incident.impact,
            "response_actions": incident.response_actions,
            "lessons_learned": incident.lessons_learned,
            "preventive_measures": incident.preventive_measures,
            "compliance_implications": incident.compliance_impact
        }
```

This comprehensive supply chain security framework provides robust protection against various attack vectors while maintaining transparency and compliance with industry standards. Regular monitoring and incident response procedures ensure rapid detection and mitigation of security threats.