# Security Policy

## Supported Versions

We actively support the following versions of Causal Eval Bench with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of Causal Eval Bench seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Private Disclosure Process

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please send an email to security@terragon-labs.com with the following information:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact of the vulnerability
- Any suggested fixes or mitigations
- Your contact information for follow-up

You should receive a response within 48 hours. If for some reason you do not, please follow up to ensure we received your original message.

### What to Expect

When you report a vulnerability to us, here's what you can expect:

1. **Acknowledgment**: We'll acknowledge receipt of your vulnerability report within 48 hours.

2. **Investigation**: We'll investigate the issue and determine its severity and impact.

3. **Resolution**: We'll work on a fix and coordinate the release timeline with you.

4. **Disclosure**: We'll publicly disclose the vulnerability after a fix is released, giving you credit for the discovery (unless you prefer to remain anonymous).

### Security Update Process

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release new versions as quickly as possible
5. Publish a security advisory with details of the vulnerability

## Security Considerations

### Data Handling

Causal Eval Bench processes evaluation data that may contain sensitive information:

- **Model Responses**: Store model responses securely and consider data retention policies
- **API Keys**: Never log or store API keys for external services
- **Evaluation Results**: Protect evaluation results that may contain proprietary information

### API Security

Our API implements several security measures:

- **Authentication**: JWT-based authentication for API access
- **Rate Limiting**: Protection against abuse and DoS attacks
- **Input Validation**: Strict validation of all input parameters
- **Output Sanitization**: Sanitization of responses to prevent injection attacks

### Infrastructure Security

- **Container Security**: Regular updates of base images and dependencies
- **Secrets Management**: Secure handling of secrets and configuration
- **Network Security**: Proper isolation and firewall configurations
- **Monitoring**: Security monitoring and alerting

### Dependencies

We regularly audit our dependencies for known vulnerabilities:

- Automated dependency scanning with Dependabot
- Regular security audits with `safety` and `bandit`
- Pinned dependency versions for reproducible builds
- Regular updates to address security issues

## Security Best Practices for Users

When using Causal Eval Bench:

### API Keys and Secrets

- Store API keys securely using environment variables or secret management systems
- Never commit API keys or secrets to version control
- Rotate API keys regularly
- Use principle of least privilege for API access

### Data Protection

- Be mindful of data privacy when evaluating models
- Implement proper access controls for evaluation results
- Consider data anonymization for sensitive datasets
- Follow your organization's data handling policies

### Network Security

- Use HTTPS for all API communications
- Implement proper firewall rules
- Consider VPN or private networks for sensitive deployments
- Monitor network traffic for anomalies

### Monitoring and Logging

- Enable security logging and monitoring
- Set up alerts for suspicious activities
- Regular review of access logs
- Implement log retention policies

## Responsible Disclosure

We believe in responsible disclosure and ask that you:

- Give us reasonable time to address the issue before public disclosure
- Make a good faith effort to avoid privacy violations and disruption
- Only interact with accounts you own or with explicit permission
- Do not access or modify data that doesn't belong to you

## Recognition

We appreciate the security research community's efforts to keep our users safe. Security researchers who responsibly disclose vulnerabilities will be:

- Acknowledged in our security advisories (unless anonymity is requested)
- Listed in our hall of fame (with permission)
- Considered for our security researcher recognition program

## Contact Information

For security-related inquiries:
- Email: security@terragon-labs.com
- PGP Key: [Available on request]
- Response Time: 48 hours for initial response

For general support:
- Email: support@terragon-labs.com
- GitHub Issues: For non-security related issues only

## Security Advisories

Stay informed about security updates:
- Subscribe to our security advisories on GitHub
- Follow [@TerragonLabs](https://twitter.com/TerragonLabs) for security announcements
- Check our [security page](https://terragon-labs.com/security) for updates

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.org/dev/security/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [API Security Best Practices](https://owasp.org/www-project-api-security/)

---

Last updated: January 29, 2025