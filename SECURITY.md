# Security Policy

## Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Causal Eval Bench, please follow these steps:

### 1. **Do NOT** create a public GitHub issue

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Report privately

Send your vulnerability report to our security team:

- **Email**: security@causal-eval-bench.org
- **Subject**: [SECURITY] Brief description of the issue

### 3. Include the following information

- **Description**: A clear description of the vulnerability
- **Impact**: How the vulnerability could be exploited and its potential impact
- **Reproduction**: Step-by-step instructions to reproduce the issue
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have ideas for how to fix the issue

### 4. Response timeline

- **Acknowledgment**: We will acknowledge receipt of your report within 24 hours
- **Initial assessment**: We will provide an initial assessment within 72 hours
- **Regular updates**: We will send updates on our progress every 7 days
- **Resolution**: We aim to resolve critical vulnerabilities within 90 days

## Security Measures

### Code Security

- **Static Analysis**: All code is scanned with Bandit for security vulnerabilities
- **Dependency Scanning**: Dependencies are regularly scanned with Safety
- **Secret Detection**: Secrets are detected and prevented from being committed
- **Code Review**: All changes require security-focused code review

### Infrastructure Security

- **Container Security**: Docker images are scanned for vulnerabilities with Trivy
- **Network Security**: Services communicate over encrypted channels
- **Access Control**: Principle of least privilege is enforced
- **Monitoring**: Security events are logged and monitored

### Data Security

- **Encryption at Rest**: Sensitive data is encrypted when stored
- **Encryption in Transit**: All API communications use HTTPS/TLS
- **Input Validation**: All user inputs are validated and sanitized
- **Output Encoding**: All outputs are properly encoded to prevent injection

### API Security

- **Authentication**: JWT-based authentication for API access
- **Authorization**: Role-based access control (RBAC)
- **Rate Limiting**: API endpoints are rate-limited to prevent abuse
- **Input Validation**: All API inputs are validated against schemas

## Security Best Practices for Users

### For Developers

1. **Keep Dependencies Updated**: Regularly update dependencies to get security patches
2. **Use Environment Variables**: Never hardcode API keys or secrets in code
3. **Secure Configuration**: Follow our security configuration guidelines
4. **Monitor Logs**: Regularly review application logs for suspicious activity

### For Deployments

1. **Use HTTPS**: Always deploy with HTTPS enabled
2. **Secure Headers**: Configure appropriate security headers
3. **Database Security**: Use strong passwords and encrypted connections
4. **Container Security**: Keep base images and containers updated
5. **Network Security**: Use firewalls and VPNs where appropriate

### For API Keys

1. **Rotation**: Regularly rotate API keys and credentials
2. **Least Privilege**: Use API keys with minimal required permissions
3. **Monitoring**: Monitor API key usage for unusual patterns
4. **Revocation**: Immediately revoke compromised keys

## Known Security Considerations

### Model API Keys

- **Risk**: API keys for model providers (OpenAI, Anthropic, etc.) have access to external services
- **Mitigation**: Use environment variables, rotate keys regularly, monitor usage
- **Recommendation**: Use separate keys for development and production

### Evaluation Data

- **Risk**: Evaluation questions might contain sensitive information
- **Mitigation**: Sanitize inputs, avoid processing personal data
- **Recommendation**: Review evaluation datasets for sensitive content

### Result Storage

- **Risk**: Evaluation results might reveal model capabilities or biases
- **Mitigation**: Implement proper access controls and encryption
- **Recommendation**: Consider data retention policies

## Security Monitoring

We actively monitor for:

- **Unusual API usage patterns**
- **Failed authentication attempts**
- **Suspicious evaluation requests**
- **Performance anomalies that might indicate attacks**
- **Dependency vulnerabilities**

## Incident Response

In case of a security incident:

1. **Containment**: Immediately contain the incident to prevent further damage
2. **Assessment**: Assess the scope and impact of the incident
3. **Communication**: Notify affected users within 24 hours
4. **Resolution**: Implement fixes and validate effectiveness
5. **Post-mortem**: Conduct post-incident review and improve processes

## Security Updates

- **Critical**: Immediate release and notification
- **High**: Release within 72 hours
- **Medium**: Release with next scheduled update
- **Low**: Release with next minor version

## Compliance

This project follows security practices aligned with:

- **OWASP Top 10**: Web application security risks
- **SANS Top 25**: Most dangerous software errors
- **ISO 27001**: Information security management
- **SOC 2**: Security and availability principles

## Security Tools

We use the following tools for security:

- **Bandit**: Python security linter
- **Safety**: Python dependency vulnerability scanner
- **detect-secrets**: Prevent secrets in code
- **Trivy**: Container vulnerability scanner
- **OWASP ZAP**: Web application security testing

## Security Training

Our development team receives training on:

- **Secure coding practices**
- **Common vulnerability patterns**
- **Security testing techniques**
- **Incident response procedures**

## Contact Information

- **Security Team**: security@causal-eval-bench.org
- **General Contact**: contact@causal-eval-bench.org
- **Documentation**: https://docs.causal-eval-bench.org/security

## Acknowledgments

We thank the security researchers who have responsibly disclosed vulnerabilities:

- (No reported vulnerabilities yet)

---

**Last Updated**: January 27, 2025
**Next Review**: April 27, 2025