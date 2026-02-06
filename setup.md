# Visual-ML Production Setup Guide

Complete guide for setting up Visual-ML in development and production environments.

## 🔧 Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | ≥ 3.10 | Backend runtime |
| **Node.js** | ≥ 18.x | Frontend build & runtime |
| **PostgreSQL** | ≥ 14.x | Production database |
| **Redis** | ≥ 7.x | Task queue & caching |
| **Docker** | ≥ 24.x | Container orchestration (recommended) |

---

## 🚀 Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/Vizuara-AI-Lab/Visual-ML
cd Visual-ML
```

### 2. Backend Setup

#### Install Python Dependencies

Using uv (Recommended - Faster)**

```bash
cd server
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

#### Configure Environment

#### Initialize Database

```bash
# Create database tables
alembic upgrade head

# Verify migration
alembic current
```

#### Start Backend Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd client

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env and add your VITE_GOOGLE_CLIENT_ID

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

### 4. Redis Setup (For Email & Background Tasks)

 Docker (Recommended)

```bash
docker run -d -p 6379:6379 --name redis-visualml redis:latest
```


```

### 5. Start Celery Worker (For Email Tasks)

```bash
cd server

# Windows
celery -A app.core.celery_app worker --loglevel=info --pool=solo

# Linux/macOS
celery -A app.core.celery_app worker --loglevel=info
```

### 6. Verify Setup

✅ Backend: `http://localhost:8000/docs`  
✅ Frontend: `http://localhost:5173`  
✅ Redis: `redis-cli ping` (should return `PONG`)  
✅ Celery: Check worker logs for registered tasks


```python
# In Python shell
from app.services.email_service import EmailService

email_service = EmailService()
email_service.send_verification_otp("test@example.com", "123456")
```

---

## 🔄 Background Tasks (Celery)

### Redis Setup

**Production Redis Configuration** (`/etc/redis/redis.conf`):

```conf
# Security
requirepass your_secure_redis_password
bind 127.0.0.1

# Persistence
save 900 1
save 300 10
save 60 10000

# Memory
maxmemory 256mb
maxmemory-policy allkeys-lru
```

### Start Celery Worker

**Development:**

```bash
celery -A app.core.celery_app worker --loglevel=info --pool=solo
```

**Production (Supervisor):**

Already configured in supervisor config above.

### Monitor Celery

**Install Flower (Web UI):**

```bash
pip install flower
celery -A app.core.celery_app flower --port=5555
```

Access at: `http://localhost:5555`

**CLI Monitoring:**

```bash
# Check worker status
celery -A app.core.celery_app inspect active

# Check registered tasks
celery -A app.core.celery_app inspect registered

# Purge all tasks
celery -A app.core.celery_app purge
```

---

## 📊 Monitoring & Logging

### Application Logs

**Backend logs** (using Loguru):

```bash
# Development
tail -f server/logs/app.log

# Production (Supervisor)
tail -f /var/log/visualml/backend.out.log
tail -f /var/log/visualml/celery.out.log
```

### Database Monitoring

```bash
# PostgreSQL connections
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# Database size
sudo -u postgres psql -c "SELECT pg_size_pretty(pg_database_size('visualml_prod'));"
```

### Redis Monitoring

```bash
# Connect to Redis CLI
redis-cli -a your_password

# Monitor commands in real-time
MONITOR

# Get info
INFO

# Check memory usage
INFO memory
```

### Nginx Logs

```bash
# Access logs
tail -f /var/log/nginx/access.log

# Error logs
tail -f /var/log/nginx/error.log
```

### Health Checks

Create health check endpoints:

```python
# In main.py
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Error

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U visualml_user -d visualml_prod -h localhost

# Check DATABASE_URL in .env
```

#### 2. Redis Connection Error

```bash
# Check Redis is running
sudo systemctl status redis

# Test connection
redis-cli ping

# Check REDIS_URL in .env
```

#### 3. Celery Worker Not Starting

```bash
# Check Redis connection
redis-cli ping

# Check Celery logs
tail -f /var/log/visualml/celery.err.log

# Restart worker
sudo supervisorctl restart visualml-celery
```

#### 4. Email Not Sending

```bash
# Check Brevo credentials in .env
# Check Celery worker logs
# Test SMTP connection:
telnet smtp-relay.brevo.com 587
```

#### 5. Frontend Build Fails

```bash
# Clear cache
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be ≥18

# Build with verbose output
npm run build -- --debug
```

#### 6. Migration Fails

```bash
# Check current version
alembic current

# Rollback and retry
alembic downgrade -1
alembic upgrade head

# If stuck, stamp to specific version
alembic stamp head
```

### Performance Optimization

#### Backend

```env
# Increase workers (CPU cores * 2 + 1)
uvicorn main:app --workers 4

# Enable caching
ENABLE_CACHE=True
CACHE_TTL=3600
```

#### Database

```sql
-- Create indexes for frequently queried fields
CREATE INDEX idx_student_email ON students(emailId);
CREATE INDEX idx_project_student ON projects(studentId);
```

#### Frontend

```bash
# Enable production build optimizations
npm run build

# Analyze bundle size
npm install -D rollup-plugin-visualizer
```

---

## 🔒 Security Checklist

- [ ] Change all default passwords (SECRET_KEY, ADMIN_PASSWORD, DB_PASSWORD)
- [ ] Enable HTTPS with SSL certificate
- [ ] Configure CORS properly (no wildcard in production)
- [ ] Set up firewall (ufw/iptables)
- [ ] Enable rate limiting
- [ ] Regular security updates (`apt update && apt upgrade`)
- [ ] Backup database regularly
- [ ] Use environment variables (never commit secrets)
- [ ] Enable Redis password authentication
- [ ] Configure PostgreSQL to only accept local connections
- [ ] Set up monitoring and alerting

---

## 📦 Backup & Recovery

### Database Backup

```bash
# Backup PostgreSQL
pg_dump -U visualml_user visualml_prod > backup_$(date +%Y%m%d).sql

# Restore
psql -U visualml_user visualml_prod < backup_20260122.sql

# Automated daily backups (crontab)
0 2 * * * pg_dump -U visualml_user visualml_prod > /backups/db_$(date +\%Y\%m\%d).sql
```

### File Backup

```bash
# Backup uploads and models
tar -czf uploads_backup.tar.gz server/uploads server/models

# Restore
tar -xzf uploads_backup.tar.gz
```

---

## 🎯 Quick Start Commands

### Development

```bash
# Terminal 1: Backend
cd server && uvicorn main:app --reload

# Terminal 2: Frontend
cd client && npm run dev

# Terminal 3: Redis
docker run -d -p 6379:6379 redis:latest

# Terminal 4: Celery
cd server && celery -A app.core.celery_app worker --loglevel=info --pool=solo
```

### Production (Docker)

```bash
# Build and start all services
docker-compose -f docker-compose.prod.yml up -d --build

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop all services
docker-compose -f docker-compose.prod.yml down
```

### Production (Traditional)

```bash
# Start all services
sudo supervisorctl start all
sudo systemctl start nginx

# Check status
sudo supervisorctl status
sudo systemctl status nginx

# Restart services
sudo supervisorctl restart visualml-backend
sudo supervisorctl restart visualml-celery
```

---

## 📞 Support

For issues or questions:
- Check logs first
- Review this setup guide
- Check GitHub issues
- Contact: admin@visualml.com

---

**Last Updated**: January 2026  
**Version**: 1.0.0



two method graph base and the previous output method
