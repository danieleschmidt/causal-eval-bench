# Incident Response Runbook

This runbook provides step-by-step procedures for responding to and resolving incidents in the Causal Eval Bench system.

## Incident Classification

### Severity Levels

| Level | Description | Impact | Response Time |
|-------|-------------|---------|---------------|
| **P0 - Critical** | Complete service outage | All users affected | 15 minutes |
| **P1 - High** | Major functionality unavailable | Most users affected | 1 hour |
| **P2 - Medium** | Minor functionality affected | Some users affected | 4 hours |
| **P3 - Low** | Performance degradation | Limited impact | 24 hours |

### Examples

#### P0 Critical
- API completely unresponsive
- Database corruption or total failure
- Security breach with data exposure

#### P1 High
- Evaluation service completely down
- Authentication system failure
- Mass evaluation failures (>50%)

#### P2 Medium
- Single model API integration failing
- Intermittent database connectivity issues
- Performance degradation (>5s response times)

#### P3 Low
- Minor UI bugs
- Non-critical feature unavailable
- Performance slightly degraded (2-5s response times)

## General Response Procedure

### 1. Initial Response (0-5 minutes)

1. **Acknowledge the Incident**
   ```bash
   # Check if incident is already being handled
   # Post in #incidents channel
   "🚨 Incident Response Started - [BRIEF DESCRIPTION] - [YOUR NAME] responding"
   ```

2. **Initial Assessment**
   ```bash
   # Quick health check
   make health
   
   # Check service status
   curl -f http://localhost:8000/health
   
   # Check recent logs
   make logs-app | tail -100
   ```

3. **Classify Severity**
   - Determine P0-P3 level based on impact
   - Update incident channel with severity
   - Tag appropriate team members

### 2. Investigation (5-15 minutes)

1. **Gather Information**
   ```bash
   # Check system metrics
   # Prometheus: http://localhost:9090
   # Look for spikes in error rates, latency, resource usage
   
   # Check recent deployments
   git log --oneline -n 10
   
   # Check infrastructure status
   docker-compose ps
   docker stats
   ```

2. **Identify Root Cause**
   - Recent code changes
   - Infrastructure changes
   - External service issues
   - Resource exhaustion
   - Database issues

### 3. Mitigation (15+ minutes)

1. **Immediate Actions**
   - Scale resources if needed
   - Restart services if appropriate
   - Rollback recent deployments if suspected
   - Implement temporary workarounds

2. **Communication**
   - Update status page
   - Notify stakeholders for P0/P1
   - Regular updates every 30 minutes

### 4. Resolution

1. **Implement Fix**
   - Deploy permanent solution
   - Verify fix resolves issue
   - Monitor for regression

2. **Post-Incident**
   - Update status page with resolution
   - Schedule post-incident review
   - Document lessons learned

## Specific Incident Types

### API Service Down (P0)

#### Symptoms
- Health check endpoint returning 503/404
- All API requests failing
- Prometheus showing `up{job="causal-eval-api"} == 0`

#### Investigation Steps
```bash
# 1. Check container status
docker-compose ps app

# 2. Check application logs
make logs-app | tail -50

# 3. Check resource usage
docker stats causal-eval-app

# 4. Check dependencies
make health
```

#### Common Causes & Solutions

**Application Crash**
```bash
# Check exit code and logs
docker-compose logs app

# Restart service
docker-compose restart app

# If recurring, check for memory leaks or uncaught exceptions
```

**Resource Exhaustion**
```bash
# Check memory usage
docker stats

# Scale up resources
docker-compose up --scale app=2

# Or increase resource limits in docker-compose.yml
```

**Database Connection Issues**
```bash
# Check database connectivity
make shell-db

# Check connection pool status
# Look for "remaining connection slots are reserved" errors

# Restart database if needed
docker-compose restart postgres
```

### High Error Rate (P1)

#### Symptoms
- Error rate >5% for sustained period
- Prometheus alert: `HighErrorRate`
- Increased 5xx responses in logs

#### Investigation Steps
```bash
# 1. Check error distribution
# Grafana dashboard: API Overview

# 2. Check recent deployments
git log --oneline --since="2 hours ago"

# 3. Analyze error types
make logs-app | grep ERROR | tail -20

# 4. Check external dependencies
# Test model API endpoints
curl -f https://api.openai.com/v1/models
```

#### Common Causes & Solutions

**Code Bug in Recent Deployment**
```bash
# Rollback to previous version
git revert <commit-hash>
make docker
docker-compose up -d

# Or use previous image
docker-compose down
docker-compose up -d --image previous-tag
```

**External API Issues**
```bash
# Check model API status pages
# Implement circuit breaker if not already done
# Use cached responses temporarily

# Disable problematic integrations temporarily
# Set environment variable to skip failing APIs
```

**Database Performance Issues**
```bash
# Check for long-running queries
make shell-db
SELECT * FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '30 seconds';

# Kill problematic queries if needed
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE ...;
```

### Database Issues (P0/P1)

#### Symptoms
- Database health check failing
- Connection timeout errors
- `database_connection_pool_exhausted` alert

#### Investigation Steps
```bash
# 1. Check database container
docker-compose ps postgres

# 2. Check database logs
docker-compose logs postgres

# 3. Test connectivity
make shell-db

# 4. Check active connections
SELECT count(*) FROM pg_stat_activity;
```

#### Common Causes & Solutions

**Connection Pool Exhausted**
```bash
# Check current connections
SELECT count(*), state FROM pg_stat_activity GROUP BY state;

# Kill idle connections
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE state = 'idle' AND query_start < NOW() - INTERVAL '1 hour';

# Increase connection pool size (temporary)
# Edit docker-compose.yml postgres command
```

**Database Corruption**
```bash
# Check database integrity
REINDEX DATABASE causal_eval_bench;

# If corruption detected, restore from backup
docker-compose down
# Restore backup procedure
docker-compose up -d
```

**Disk Space Issues**
```bash
# Check disk usage
df -h

# Clean up old logs
docker system prune -f

# Expand disk if needed (cloud provider specific)
```

### Performance Degradation (P2)

#### Symptoms
- Response times >2 seconds (95th percentile)
- `HighLatency` alert firing
- User complaints about slow responses

#### Investigation Steps
```bash
# 1. Check system resources
docker stats

# 2. Check database performance
# Look for slow queries in logs
make logs-app | grep "duration.*ms" | sort -k5 -nr | head -10

# 3. Check cache performance
# Grafana: Cache Hit Rate dashboard

# 4. Check external API response times
# Grafana: Model API Performance dashboard
```

#### Common Causes & Solutions

**High CPU Usage**
```bash
# Scale horizontally
docker-compose up --scale app=3

# Or vertically (increase CPU limits)
# Edit docker-compose.yml resources section
```

**Database Performance**
```bash
# Check for missing indexes
EXPLAIN ANALYZE SELECT ... ;

# Add indexes for frequently queried columns
CREATE INDEX idx_evaluations_created_at ON evaluations(created_at);

# Update statistics
ANALYZE;
```

**Cache Issues**
```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Clear cache if needed (temporary solution)
docker-compose exec redis redis-cli flushall

# Increase cache memory limit
# Edit docker/redis/redis.conf
```

### External API Failures (P2)

#### Symptoms
- Model API requests failing
- `model_api_rate_limit_errors` increasing
- Specific model evaluations failing

#### Investigation Steps
```bash
# 1. Check API status pages
# OpenAI: https://status.openai.com/
# Anthropic: https://status.anthropic.com/

# 2. Test API endpoints directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# 3. Check rate limiting
make logs-app | grep "rate.*limit"

# 4. Check API key validity
```

#### Common Causes & Solutions

**Rate Limiting**
```bash
# Implement exponential backoff
# Check current rate limit settings
# Distribute load across multiple API keys if available

# Temporary: Reduce concurrent requests
# Set MAX_CONCURRENT_API_CALLS=5 in environment
```

**API Key Issues**
```bash
# Rotate API keys
# Update environment variables
# Restart services to pick up new keys
docker-compose restart app worker
```

**Service Outage**
```bash
# Check official status pages
# Implement circuit breaker pattern
# Fall back to alternative models if configured

# Enable cached responses temporarily
# Set ENABLE_RESPONSE_CACHE=true
```

## Communication Templates

### Initial Incident Report
```
🚨 INCIDENT ALERT - P[X] - [TITLE]

Status: Investigating
Started: [TIME]
Impact: [DESCRIPTION]
Responder: [NAME]

Initial findings: [BRIEF DESCRIPTION]

Updates will be posted every 30 minutes.
```

### Status Update
```
🔄 INCIDENT UPDATE - P[X] - [TITLE]

Status: [Investigating/Mitigating/Resolved]
Duration: [TIME since start]

Progress: [WHAT'S BEEN DONE]
Current action: [WHAT'S BEING DONE NOW]
ETA: [ESTIMATED RESOLUTION TIME]

Next update in 30 minutes.
```

### Resolution Notice
```
✅ INCIDENT RESOLVED - P[X] - [TITLE]

Status: Resolved
Duration: [TOTAL TIME]
Root Cause: [BRIEF DESCRIPTION]

Resolution: [WHAT WAS DONE]
Monitoring: [ONGOING MONITORING]

Post-incident review scheduled for [DATE/TIME].
```

## Emergency Contacts

### Team Contacts
- **On-call Engineer**: [Phone/Slack]
- **Team Lead**: [Phone/Slack]
- **Infrastructure**: [Phone/Slack]
- **Security Team**: [Phone/Slack]

### External Contacts
- **Cloud Provider Support**: [Support ticket system]
- **Database Admin**: [Contact info]
- **Security Incident Response**: [Contact info]

## Escalation Procedures

### When to Escalate

1. **Automatic Escalation**
   - P0 incidents after 30 minutes
   - P1 incidents after 2 hours
   - Any security-related incident

2. **Manual Escalation Triggers**
   - Unable to determine root cause
   - Fix requires specialized knowledge
   - External dependencies involved
   - Customer data potentially compromised

### Escalation Process

1. **Internal Escalation**
   ```
   # Tag team lead in incident channel
   @team-lead - Need escalation for P[X] incident
   
   # Provide context
   - What's been tried
   - Current status
   - Why escalation needed
   ```

2. **External Escalation**
   - Contact cloud provider support
   - Engage external security team
   - Notify legal team if data involved

## Tools and Resources

### Monitoring Dashboards
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Application Health**: http://localhost:8000/health

### Useful Commands
```bash
# Quick system overview
make health

# Service logs
make logs-app
make logs-db

# Resource usage
docker stats

# Database access
make shell-db

# Application shell
make shell

# Restart services
docker-compose restart [service]

# Scale services
docker-compose up --scale app=3
```

### External Resources
- **OpenAI Status**: https://status.openai.com/
- **Anthropic Status**: https://status.anthropic.com/
- **AWS Status**: https://status.aws.amazon.com/
- **GitHub Status**: https://www.githubstatus.com/

## Post-Incident Procedures

### Immediately After Resolution

1. **Update Status Page**
   - Mark incident as resolved
   - Provide brief resolution summary

2. **Internal Communication**
   - Post resolution in incident channel
   - Thank team members who helped
   - Schedule post-incident review

3. **Customer Communication**
   - Notify affected customers (P0/P1)
   - Provide explanation and next steps

### Post-Incident Review

Schedule within 48 hours of resolution:

1. **Timeline Review**
   - When incident started
   - When detected
   - Response actions taken
   - When resolved

2. **Root Cause Analysis**
   - Technical root cause
   - Process failures
   - Communication issues

3. **Action Items**
   - Technical improvements
   - Process improvements
   - Monitoring enhancements
   - Documentation updates

### Follow-up Actions

1. **Implement Fixes**
   - Deploy permanent solutions
   - Update monitoring/alerting
   - Improve documentation

2. **Share Learnings**
   - Update runbooks
   - Train team members
   - Share with other teams

---

Remember: The goal is to restore service quickly while maintaining clear communication and learning from each incident.