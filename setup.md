# Visual-ML Production Setup Guide

Complete guide for setting up Visual-ML in development and production environments.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Production Deployment](#production-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Email Service Setup](#email-service-setup)
7. [Background Tasks (Celery)](#background-tasks-celery)
8. [Monitoring & Logging](#monitoring--logging)
9. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | ≥ 3.10 | Backend runtime |
| **Node.js** | ≥ 18.x | Frontend build & runtime |
| **PostgreSQL** | ≥ 14.x | Production database |
| **Redis** | ≥ 7.x | Task queue & caching |
| **Docker** | ≥ 24.x | Container orchestration (recommended) |
| **Nginx** | Latest | Reverse proxy (production) |

### Optional Tools

- **uv** - Fast Python package installer (recommended)
- **PM2** - Process manager for Node.js
- **Supervisor** - Process manager for Python workers
- **Certbot** - SSL certificate management

---

## 🚀 Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd Visual-ML
```

### 2. Backend Setup

#### Install Python Dependencies

**Option A: Using uv (Recommended - Faster)**

```bash
cd server
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Option B: Using pip**

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

#### Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# IMPORTANT: Change SECRET_KEY, ADMIN_PASSWORD, and database credentials
```

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

**Option A: Docker (Recommended)**

```bash
docker run -d -p 6379:6379 --name redis-visualml redis:latest
```

**Option B: Local Installation**

```bash
# Ubuntu/Debian
sudo apt-get install redis-server
redis-server

# macOS
brew install redis
brew services start redis

# Windows (WSL)
sudo apt-get install redis-server
redis-server
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

---

## 🏭 Production Deployment

### Architecture Overview

```
Internet → Nginx (SSL) → Frontend (Static Files)
                       ↓
                    Backend API (Uvicorn/Gunicorn)
                       ↓
                    PostgreSQL Database
                       ↓
                    Redis (Cache + Queue)
                       ↓
                    Celery Workers
```

### Option 1: Docker Deployment (Recommended)

#### 1. Create Docker Compose Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: visualml-postgres
    environment:
      POSTGRES_DB: visualml_prod
      POSTGRES_USER: visualml_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U visualml_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    container_name: visualml-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Backend API
  backend:
    build:
      context: ./server
      dockerfile: Dockerfile
    container_name: visualml-backend
    environment:
      - DATABASE_URL=postgresql://visualml_user:${DB_PASSWORD}@postgres:5432/visualml_prod
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
      - DEBUG=False
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./server/uploads:/app/uploads
      - ./server/models:/app/models
    restart: unless-stopped

  # Celery Worker
  celery-worker:
    build:
      context: ./server
      dockerfile: Dockerfile
    container_name: visualml-celery
    command: celery -A app.core.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://visualml_user:${DB_PASSWORD}@postgres:5432/visualml_prod
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  # Frontend
  frontend:
    build:
      context: ./client
      dockerfile: Dockerfile
    container_name: visualml-frontend
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### 2. Create Backend Dockerfile

Create `server/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads models

# Expose port
EXPOSE 8000

# Run migrations and start server
CMD alembic upgrade head && \
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 3. Create Frontend Dockerfile

Create `client/Dockerfile`:

```dockerfile
# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

#### 4. Create Nginx Configuration

Create `client/nginx.conf`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Frontend routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

#### 5. Deploy with Docker Compose

```bash
# Create .env file for Docker Compose
cat > .env << EOF
DB_PASSWORD=your_secure_db_password
REDIS_PASSWORD=your_secure_redis_password
SECRET_KEY=your_super_secret_key_min_32_chars
EOF

# Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Run migrations
docker exec visualml-backend alembic upgrade head
```

### Option 2: Traditional Server Deployment

#### 1. Server Setup (Ubuntu 22.04)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip \
    postgresql postgresql-contrib redis-server nginx \
    git supervisor certbot python3-certbot-nginx
```

#### 2. Database Setup

```bash
# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE visualml_prod;
CREATE USER visualml_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE visualml_prod TO visualml_user;
\q
EOF
```

#### 3. Backend Deployment

```bash
# Create application user
sudo useradd -m -s /bin/bash visualml
sudo su - visualml

# Clone repository
git clone <repository-url> /home/visualml/app
cd /home/visualml/app/server

# Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure environment
cp .env.example .env
nano .env  # Edit with production values

# Run migrations
alembic upgrade head

# Exit to root user
exit
```

#### 4. Configure Supervisor (Backend + Celery)

Create `/etc/supervisor/conf.d/visualml.conf`:

```ini
[program:visualml-backend]
command=/home/visualml/app/server/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
directory=/home/visualml/app/server
user=visualml
autostart=true
autorestart=true
stderr_logfile=/var/log/visualml/backend.err.log
stdout_logfile=/var/log/visualml/backend.out.log

[program:visualml-celery]
command=/home/visualml/app/server/.venv/bin/celery -A app.core.celery_app worker --loglevel=info
directory=/home/visualml/app/server
user=visualml
autostart=true
autorestart=true
stderr_logfile=/var/log/visualml/celery.err.log
stdout_logfile=/var/log/visualml/celery.out.log
```

```bash
# Create log directory
sudo mkdir -p /var/log/visualml
sudo chown visualml:visualml /var/log/visualml

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status
```

#### 5. Frontend Deployment

```bash
cd /home/visualml/app/client

# Install dependencies and build
npm ci
npm run build

# Copy build to nginx directory
sudo cp -r dist/* /var/www/visualml/
sudo chown -R www-data:www-data /var/www/visualml
```

#### 6. Configure Nginx

Create `/etc/nginx/sites-available/visualml`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    root /var/www/visualml;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # API Backend
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for ML operations
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # File upload size
    client_max_body_size 100M;
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/visualml /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 7. SSL Certificate (Let's Encrypt)

```bash
sudo certbot --nginx -d your-domain.com
sudo systemctl reload nginx
```

---

## ⚙️ Environment Configuration

### Backend (.env)

**Critical Production Settings:**

```env
# Application
APP_NAME=Visual-ML
ENVIRONMENT=production
DEBUG=False

# Database (PostgreSQL for production)
DATABASE_URL=postgresql://visualml_user:password@localhost:5432/visualml_prod

# Security (MUST CHANGE!)
SECRET_KEY=generate-a-secure-random-key-min-32-characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_MINUTES=10080

# CORS
ALLOWED_ORIGINS=["https://your-domain.com"]

# Redis
REDIS_URL=redis://:password@localhost:6379/0
ENABLE_CACHE=True

# Celery
CELERY_BROKER_URL=redis://:password@localhost:6379/0
CELERY_RESULT_BACKEND=redis://:password@localhost:6379/0
ENABLE_BACKGROUND_TASKS=True

# Email (Brevo SMTP)
BREVO_SMTP_SERVER=smtp-relay.brevo.com
BREVO_SMTP_PORT=587
BREVO_SMTP_USER=your-brevo-email@example.com
BREVO_SMTP_PASSWORD=your-brevo-smtp-key
BREVO_SENDER_EMAIL=noreply@your-domain.com
BREVO_SENDER_NAME=Visual ML

# Admin
ADMIN_EMAIL=admin@your-domain.com
ADMIN_PASSWORD=change-this-secure-password

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
```

### Frontend (.env)

```env
VITE_API_URL=https://your-domain.com/api/v1
VITE_GOOGLE_CLIENT_ID=your-google-oauth-client-id
```

### Generate Secure SECRET_KEY

```bash
# Python method
python -c "import secrets; print(secrets.token_urlsafe(32))"

# OpenSSL method
openssl rand -base64 32
```

---

## 🗄️ Database Setup

### Development (SQLite)

```bash
# Already configured in .env
DATABASE_URL=sqlite:///./visual_ml.db

# Run migrations
alembic upgrade head
```

### Production (PostgreSQL)

```bash
# Install PostgreSQL client
pip install psycopg2-binary

# Update .env
DATABASE_URL=postgresql://user:password@localhost:5432/visualml_prod

# Run migrations
alembic upgrade head

# Create admin user (if needed)
python -c "from app.services.auth_service import create_admin; create_admin()"
```

### Migration Commands

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Check current version
alembic current

# View migration history
alembic history
```

---

## 📧 Email Service Setup

### Brevo (Recommended)

1. **Sign up**: https://www.brevo.com/
2. **Get SMTP credentials**:
   - Go to Settings → SMTP & API
   - Copy SMTP server, port, login, and password
3. **Configure sender**:
   - Verify your domain or email
   - Set up SPF/DKIM records for better deliverability

### Environment Variables

```env
BREVO_SMTP_SERVER=smtp-relay.brevo.com
BREVO_SMTP_PORT=587
BREVO_SMTP_USER=your-email@example.com
BREVO_SMTP_PASSWORD=your-smtp-key
BREVO_SENDER_EMAIL=noreply@your-domain.com
BREVO_SENDER_NAME=Visual ML
```

### Test Email Delivery

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
