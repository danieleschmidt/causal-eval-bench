#!/usr/bin/env python3
"""Security scan and vulnerability assessment for Causal Evaluation Bench."""

import sys
import os
import re
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))

class SecurityLevel(Enum):
    """Security risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityFinding:
    """Security assessment finding."""
    category: str
    level: SecurityLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    cve_reference: Optional[str] = None

class SecurityScanner:
    """Comprehensive security scanner for the codebase."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.findings: List[SecurityFinding] = []
        self.excluded_paths = {
            ".git", "__pycache__", ".pytest_cache", "node_modules", 
            "venv", ".venv", "build", "dist", ".tox"
        }
    
    def scan_all(self) -> List[SecurityFinding]:
        """Run comprehensive security scan."""
        print("🔒 Starting Security Assessment...")
        print("=" * 50)
        
        # Run all security checks
        self.scan_hardcoded_secrets()
        self.scan_sql_injection()
        self.scan_xss_vulnerabilities() 
        self.scan_insecure_dependencies()
        self.scan_file_permissions()
        self.scan_input_validation()
        self.scan_authentication_flaws()
        self.scan_configuration_security()
        self.scan_logging_security()
        self.scan_crypto_usage()
        
        return self.findings
    
    def scan_hardcoded_secrets(self):
        """Scan for hardcoded secrets and credentials."""
        print("🔍 Scanning for hardcoded secrets...")
        
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password", SecurityLevel.HIGH),
            (r'api_key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded API key", SecurityLevel.HIGH),
            (r'secret_key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret key", SecurityLevel.HIGH),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token", SecurityLevel.HIGH),
            (r'private_key\s*=\s*["\'][^"\']{32,}["\']', "Hardcoded private key", SecurityLevel.CRITICAL),
            (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', "Private key in code", SecurityLevel.CRITICAL),
            (r'[A-Za-z0-9+/]{40,}={0,2}', "Base64 encoded potential secret", SecurityLevel.MEDIUM)
        ]
        
        py_files = list(self.project_root.rglob("*.py"))
        
        for file_path in py_files:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Skip obvious test files or examples
                        if any(test_indicator in str(file_path).lower() 
                               for test_indicator in ['test_', 'example', 'demo', 'sample']):
                            continue
                            
                        self.findings.append(SecurityFinding(
                            category="Secrets Management",
                            level=level,
                            title=description,
                            description=f"Potential hardcoded secret found: {match.group()[:50]}...",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Use environment variables or secure vault for secrets"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def scan_sql_injection(self):
        """Scan for SQL injection vulnerabilities."""
        print("🔍 Scanning for SQL injection vulnerabilities...")
        
        sql_patterns = [
            (r'execute\([^)]*%[sf][^)]*\)', "String formatting in SQL", SecurityLevel.HIGH),
            (r'execute\([^)]*\+[^)]*\)', "String concatenation in SQL", SecurityLevel.HIGH),
            (r'query\([^)]*%[sf][^)]*\)', "String formatting in query", SecurityLevel.HIGH),
            (r'SELECT.*WHERE.*=.*%[sf]', "Direct string interpolation in WHERE", SecurityLevel.HIGH),
            (r'INSERT.*VALUES.*%[sf]', "Direct string interpolation in INSERT", SecurityLevel.HIGH),
            (r'UPDATE.*SET.*=.*%[sf]', "Direct string interpolation in UPDATE", SecurityLevel.HIGH)
        ]
        
        py_files = list(self.project_root.rglob("*.py"))
        
        for file_path in py_files:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in sql_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="SQL Injection",
                            level=level,
                            title=description,
                            description=f"Potential SQL injection vulnerability: {match.group()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Use parameterized queries or ORM with proper escaping"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def scan_xss_vulnerabilities(self):
        """Scan for Cross-Site Scripting vulnerabilities."""
        print("🔍 Scanning for XSS vulnerabilities...")
        
        xss_patterns = [
            (r'render_template_string\([^)]*\+[^)]*\)', "Template injection risk", SecurityLevel.HIGH),
            (r'\.format\([^)]*request\.[^)]*\)', "Direct request data in format", SecurityLevel.MEDIUM),
            (r'f"[^"]*{request\.[^}]*}', "Request data in f-string", SecurityLevel.MEDIUM),
            (r'innerHTML.*=.*request\.', "Direct request data to innerHTML", SecurityLevel.HIGH),
            (r'document\.write\([^)]*request\.[^)]*\)', "Request data in document.write", SecurityLevel.HIGH)
        ]
        
        files_to_scan = list(self.project_root.rglob("*.py")) + list(self.project_root.rglob("*.html")) + list(self.project_root.rglob("*.js"))
        
        for file_path in files_to_scan:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in xss_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="Cross-Site Scripting (XSS)",
                            level=level,
                            title=description,
                            description=f"Potential XSS vulnerability: {match.group()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Sanitize and escape user input before rendering"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def scan_insecure_dependencies(self):
        """Scan for insecure dependencies."""
        print("🔍 Scanning dependencies for known vulnerabilities...")
        
        # Check pyproject.toml for dependencies
        pyproject_path = self.project_root / "pyproject.toml"
        requirements_path = self.project_root / "requirements.txt"
        
        insecure_packages = {
            "django": {"version": "<3.2.19", "cve": "CVE-2023-31047", "severity": SecurityLevel.HIGH},
            "flask": {"version": "<2.2.2", "cve": "CVE-2023-30861", "severity": SecurityLevel.MEDIUM},
            "requests": {"version": "<2.31.0", "cve": "CVE-2023-32681", "severity": SecurityLevel.MEDIUM},
            "pyyaml": {"version": "<6.0.1", "cve": "CVE-2022-38749", "severity": SecurityLevel.HIGH},
            "pillow": {"version": "<9.3.0", "cve": "CVE-2022-45198", "severity": SecurityLevel.HIGH}
        }
        
        # Scan pyproject.toml
        if pyproject_path.exists():
            try:
                content = pyproject_path.read_text()
                
                for package, vuln_info in insecure_packages.items():
                    if package in content:
                        self.findings.append(SecurityFinding(
                            category="Dependency Vulnerability",
                            level=vuln_info["severity"],
                            title=f"Potentially vulnerable {package} dependency",
                            description=f"Package {package} may be vulnerable to {vuln_info['cve']}",
                            file_path=str(pyproject_path),
                            recommendation=f"Update {package} to version {vuln_info['version']} or later",
                            cve_reference=vuln_info["cve"]
                        ))
            
            except Exception as e:
                print(f"Error scanning pyproject.toml: {e}")
        
        # Scan requirements.txt
        if requirements_path.exists():
            try:
                content = requirements_path.read_text()
                
                for package, vuln_info in insecure_packages.items():
                    if package in content:
                        self.findings.append(SecurityFinding(
                            category="Dependency Vulnerability", 
                            level=vuln_info["severity"],
                            title=f"Potentially vulnerable {package} dependency",
                            description=f"Package {package} may be vulnerable to {vuln_info['cve']}",
                            file_path=str(requirements_path),
                            recommendation=f"Update {package} to version {vuln_info['version']} or later",
                            cve_reference=vuln_info["cve"]
                        ))
            
            except Exception as e:
                print(f"Error scanning requirements.txt: {e}")
    
    def scan_file_permissions(self):
        """Scan for insecure file permissions."""
        print("🔍 Scanning file permissions...")
        
        sensitive_files = [
            "config.py", "settings.py", ".env", "secret.key",
            "private.pem", "id_rsa", "database.db"
        ]
        
        for file_pattern in sensitive_files:
            matching_files = list(self.project_root.rglob(file_pattern))
            
            for file_path in matching_files:
                if any(excluded in str(file_path) for excluded in self.excluded_paths):
                    continue
                
                try:
                    stat = file_path.stat()
                    permissions = oct(stat.st_mode)[-3:]
                    
                    # Check for world-readable sensitive files
                    if permissions[-1] in ['4', '5', '6', '7']:  # World readable
                        self.findings.append(SecurityFinding(
                            category="File Permissions",
                            level=SecurityLevel.MEDIUM,
                            title="World-readable sensitive file",
                            description=f"Sensitive file {file_path.name} has world-readable permissions ({permissions})",
                            file_path=str(file_path),
                            recommendation="Restrict file permissions to owner only (600 or 700)"
                        ))
                    
                    # Check for world-writable files
                    if permissions[-1] in ['2', '3', '6', '7']:  # World writable
                        self.findings.append(SecurityFinding(
                            category="File Permissions",
                            level=SecurityLevel.HIGH,
                            title="World-writable file",
                            description=f"File {file_path.name} has world-writable permissions ({permissions})",
                            file_path=str(file_path),
                            recommendation="Remove world-writable permissions"
                        ))
                
                except Exception as e:
                    print(f"Error checking permissions for {file_path}: {e}")
    
    def scan_input_validation(self):
        """Scan for input validation issues."""
        print("🔍 Scanning input validation...")
        
        validation_patterns = [
            (r'request\.(args|form|json)\[[^]]+\](?!\s*\.)(?![^{]*})', "Direct request data usage", SecurityLevel.MEDIUM),
            (r'int\(request\.(args|form)\[[^]]+\]\)', "Unsafe integer conversion", SecurityLevel.MEDIUM),
            (r'eval\([^)]*request\.[^)]*\)', "eval() with user input", SecurityLevel.CRITICAL),
            (r'exec\([^)]*request\.[^)]*\)', "exec() with user input", SecurityLevel.CRITICAL),
            (r'open\([^)]*request\.[^)]*\)', "File operations with user input", SecurityLevel.HIGH),
            (r'subprocess\.[^(]*\([^)]*request\.[^)]*\)', "Command execution with user input", SecurityLevel.CRITICAL)
        ]
        
        py_files = list(self.project_root.rglob("*.py"))
        
        for file_path in py_files:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in validation_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="Input Validation",
                            level=level,
                            title=description,
                            description=f"Potential input validation issue: {match.group()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Validate and sanitize all user input before processing"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def scan_authentication_flaws(self):
        """Scan for authentication and authorization flaws.""" 
        print("🔍 Scanning authentication mechanisms...")
        
        auth_patterns = [
            (r'password\s*==\s*["\'][^"\']*["\']', "Hardcoded password comparison", SecurityLevel.HIGH),
            (r'if.*password.*==.*request\.', "Plain text password comparison", SecurityLevel.HIGH),
            (r'md5\([^)]*password[^)]*\)', "MD5 for password hashing", SecurityLevel.HIGH),
            (r'sha1\([^)]*password[^)]*\)', "SHA1 for password hashing", SecurityLevel.MEDIUM),
            (r'session\[[^]]*\]\s*=\s*True', "Session manipulation", SecurityLevel.MEDIUM),
            (r'@.*_required.*\n\s*def.*\n(?!.*check.*auth)', "Missing auth check", SecurityLevel.MEDIUM)
        ]
        
        py_files = list(self.project_root.rglob("*.py"))
        
        for file_path in py_files:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in auth_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="Authentication",
                            level=level,
                            title=description,
                            description=f"Potential authentication flaw: {match.group()[:100]}...",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Use secure authentication methods and proper password hashing"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def scan_configuration_security(self):
        """Scan for insecure configurations."""
        print("🔍 Scanning configuration security...")
        
        config_files = [
            self.project_root / "docker-compose.yml",
            self.project_root / "Dockerfile", 
            self.project_root / "config" / "settings.py",
            self.project_root / ".env.example"
        ]
        
        insecure_configs = [
            (r'DEBUG\s*=\s*True', "Debug mode enabled", SecurityLevel.HIGH),
            (r'ALLOWED_HOSTS\s*=\s*\[\s*\]', "Empty ALLOWED_HOSTS", SecurityLevel.MEDIUM),
            (r'SECRET_KEY\s*=\s*["\'][^"\']{1,20}["\']', "Weak secret key", SecurityLevel.HIGH),
            (r'privileged:\s*true', "Privileged Docker container", SecurityLevel.HIGH),
            (r'--privileged', "Privileged Docker run", SecurityLevel.HIGH),
            (r'user:\s*root', "Running as root in container", SecurityLevel.MEDIUM)
        ]
        
        for config_file in config_files:
            if not config_file.exists():
                continue
                
            try:
                content = config_file.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in insecure_configs:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="Configuration Security",
                            level=level,
                            title=description,
                            description=f"Insecure configuration: {match.group()}",
                            file_path=str(config_file),
                            line_number=line_num,
                            recommendation="Review and secure configuration settings"
                        ))
            
            except Exception as e:
                print(f"Error scanning {config_file}: {e}")
    
    def scan_logging_security(self):
        """Scan for logging security issues."""
        print("🔍 Scanning logging practices...")
        
        logging_patterns = [
            (r'log.*password', "Password in logs", SecurityLevel.HIGH),
            (r'print\([^)]*password[^)]*\)', "Password in print statement", SecurityLevel.MEDIUM),
            (r'log.*token', "Token in logs", SecurityLevel.HIGH),
            (r'log.*secret', "Secret in logs", SecurityLevel.HIGH),
            (r'log.*key', "Key in logs", SecurityLevel.MEDIUM)
        ]
        
        py_files = list(self.project_root.rglob("*.py"))
        
        for file_path in py_files:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in logging_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="Logging Security",
                            level=level,
                            title=description,
                            description=f"Sensitive data in logs: {match.group()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Avoid logging sensitive information"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def scan_crypto_usage(self):
        """Scan for cryptographic issues."""
        print("🔍 Scanning cryptographic usage...")
        
        crypto_patterns = [
            (r'Random\(\)', "Insecure random number generation", SecurityLevel.MEDIUM),
            (r'random\.random\(\)', "Insecure random for crypto", SecurityLevel.MEDIUM), 
            (r'DES\.new\(', "Insecure DES encryption", SecurityLevel.HIGH),
            (r'RC4\.new\(', "Insecure RC4 encryption", SecurityLevel.HIGH),
            (r'MD5\.new\(', "Insecure MD5 hashing", SecurityLevel.MEDIUM),
            (r'SHA\.new\(', "Insecure SHA-1 hashing", SecurityLevel.MEDIUM),
            (r'ssl\.create_default_context\([^)]*check_hostname=False', "SSL hostname verification disabled", SecurityLevel.HIGH),
            (r'verify=False', "SSL verification disabled", SecurityLevel.HIGH)
        ]
        
        py_files = list(self.project_root.rglob("*.py"))
        
        for file_path in py_files:
            if any(excluded in str(file_path) for excluded in self.excluded_paths):
                continue
                
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, description, level in crypto_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        self.findings.append(SecurityFinding(
                            category="Cryptography",
                            level=level,
                            title=description,
                            description=f"Cryptographic issue: {match.group()}",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Use secure cryptographic algorithms and proper random number generation"
                        ))
            
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Group findings by category and severity
        by_category = {}
        by_severity = {level: 0 for level in SecurityLevel}
        
        for finding in self.findings:
            if finding.category not in by_category:
                by_category[finding.category] = []
            by_category[finding.category].append(finding)
            by_severity[finding.level] += 1
        
        # Calculate security score
        severity_weights = {
            SecurityLevel.CRITICAL: 10,
            SecurityLevel.HIGH: 7,
            SecurityLevel.MEDIUM: 4,
            SecurityLevel.LOW: 1
        }
        
        total_weight = sum(by_severity[level] * severity_weights[level] for level in SecurityLevel)
        max_possible = len(self.findings) * severity_weights[SecurityLevel.CRITICAL]
        
        if max_possible > 0:
            security_score = max(0, 100 - (total_weight / max_possible) * 100)
        else:
            security_score = 100  # No findings = perfect score
        
        return {
            "security_score": security_score,
            "total_findings": len(self.findings),
            "by_severity": {level.value: count for level, count in by_severity.items()},
            "by_category": {cat: len(findings) for cat, findings in by_category.items()},
            "findings": [
                {
                    "category": f.category,
                    "level": f.level.value,
                    "title": f.title,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "recommendation": f.recommendation,
                    "cve_reference": f.cve_reference
                }
                for f in self.findings
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        severity_counts = {level: 0 for level in SecurityLevel}
        for finding in self.findings:
            severity_counts[finding.level] += 1
        
        if severity_counts[SecurityLevel.CRITICAL] > 0:
            recommendations.append("🚨 Address CRITICAL security issues immediately")
            
        if severity_counts[SecurityLevel.HIGH] > 0:
            recommendations.append("⚠️ Fix HIGH priority security vulnerabilities")
            
        categories = set(f.category for f in self.findings)
        
        if "Secrets Management" in categories:
            recommendations.append("🔐 Implement proper secrets management (environment variables, vaults)")
            
        if "Input Validation" in categories:
            recommendations.append("🛡️ Implement comprehensive input validation and sanitization")
            
        if "Authentication" in categories:
            recommendations.append("🔒 Review and strengthen authentication mechanisms")
            
        if "Dependency Vulnerability" in categories:
            recommendations.append("📦 Update vulnerable dependencies to latest secure versions")
            
        if "Configuration Security" in categories:
            recommendations.append("⚙️ Review and secure configuration settings")
        
        if not recommendations:
            recommendations.append("✅ No critical security issues found - maintain current security practices")
        
        return recommendations


def main():
    """Run security assessment."""
    print("🔒 Causal Evaluation Bench - Security Assessment")
    print("=" * 55)
    
    scanner = SecurityScanner()
    findings = scanner.scan_all()
    report = scanner.generate_report()
    
    print(f"\n📊 SECURITY ASSESSMENT RESULTS:")
    print(f"  Security Score: {report['security_score']:.1f}/100")
    print(f"  Total Findings: {report['total_findings']}")
    print(f"  Critical: {report['by_severity']['critical']}")
    print(f"  High: {report['by_severity']['high']}")
    print(f"  Medium: {report['by_severity']['medium']}")
    print(f"  Low: {report['by_severity']['low']}")
    
    if report['by_category']:
        print(f"\n📋 FINDINGS BY CATEGORY:")
        for category, count in report['by_category'].items():
            print(f"  {category}: {count} issues")
    
    print(f"\n💡 SECURITY RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Detailed findings (show first 10)
    if report['findings']:
        print(f"\n🔍 DETAILED FINDINGS (showing first 10):")
        for i, finding in enumerate(report['findings'][:10], 1):
            print(f"  {i}. [{finding['level'].upper()}] {finding['title']}")
            print(f"     {finding['description']}")
            if finding['file_path']:
                location = f"{finding['file_path']}"
                if finding['line_number']:
                    location += f":{finding['line_number']}"
                print(f"     Location: {location}")
            print(f"     Fix: {finding['recommendation']}")
            print()
    
    # Determine overall security status
    security_score = report['security_score']
    critical_issues = report['by_severity']['critical']
    high_issues = report['by_severity']['high']
    
    print("=" * 55)
    
    if critical_issues > 0:
        print("🚨 SECURITY STATUS: CRITICAL - Immediate action required")
        security_status = False
    elif high_issues > 3:
        print("⚠️  SECURITY STATUS: HIGH RISK - Address vulnerabilities soon")  
        security_status = False
    elif security_score >= 80:
        print("✅ SECURITY STATUS: GOOD - Minor issues to address")
        security_status = True
    elif security_score >= 60:
        print("🟡 SECURITY STATUS: ACCEPTABLE - Some improvements needed")
        security_status = True
    else:
        print("🔴 SECURITY STATUS: POOR - Significant security work required")
        security_status = False
    
    # Save report to file
    report_path = Path("security_assessment_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return security_status


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)