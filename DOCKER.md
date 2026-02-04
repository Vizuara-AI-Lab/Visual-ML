# Docker Deployment Guide for Visual-ML

Complete guide for running Visual-ML with Docker in development and production environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

---

## 🚀 Quick Start

### Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.0+
- 4GB+ RAM available
- 10GB+ disk space

### Development (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/Vizuara-AI-Lab/Visual-ML
cd Visual-ML

# 2. Start all services
docker-compose up -d

# 3. Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000/docs
# Redis: localhost:6379
# PostgreSQL: localhost:5432
```

### Production (10 minutes)

```bash
# 1. Create environment file
cp .env.docker.example .env
# Edit .env with your production credentials

# 2. Build and start services
docker-compose -f docker-compose.prod.yml up -d --build

# 3. Run database migrations
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head

# 4. Access the application
# Frontend: http://localhost
# Backend API: http://localhost/api/v1/docs
```

---

## 🔧 Development Setup

### Starting Services

```bash
# Start all services in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | - |
| Backend API | http://localhost:8000/docs | - |
| PostgreSQL | localhost:5432 | user: `visualml_user`<br>pass: `visualml_dev_password`<br>db: `visualml_dev` |
| Redis | localhost:6379 | No password |

### Hot Reload

The development setup includes volume mounts for hot-reload:

- **Backend**: Code changes trigger automatic reload via `uvicorn --reload`
- **Frontend**: Vite dev server with HMR (Hot Module Replacement)

### Running Commands

```bash
# Backend commands
docker-compose exec backend python -m pytest
docker-compose exec backend alembic upgrade head
docker-compose exec backend python -m app.scripts.seed_data

# Frontend commands
docker-compose exec frontend npm run lint
docker-compose exec frontend npm run build

# Database commands
docker-compose exec postgres psql -U visualml_user -d visualml_dev
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v
```

---

## 🏭 Production Deployment

### 1. Server Setup

**Minimum Requirements:**
- 2 CPU cores
- 4GB RAM
- 20GB disk space
- Ubuntu 22.04 LTS (recommended)

**Install Docker:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Application Setup

```bash
# Create application directory
sudo mkdir -p /opt/visual-ml
sudo chown $USER:$USER /opt/visual-ml
cd /opt/visual-ml

# Clone repository
git clone https://github.com/Vizuara-AI-Lab/Visual-ML .

# Create environment file
cp .env.docker.example .env
nano .env  # Edit with production values
```

### 3. Environment Configuration

Edit `.env` with production values:

```bash
# Database - Use strong passwords!
POSTGRES_PASSWORD=your-secure-password-here
DATABASE_URL=postgresql://visualml_user:your-secure-password-here@postgres:5432/visualml_prod

# Redis - Enable authentication
REDIS_PASSWORD=your-redis-password-here

# Application Security - Generate with: openssl rand -hex 32
SECRET_KEY=your-64-character-secret-key-here

# Email Service
BREVO_API_KEY=your-brevo-api-key

# OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# AWS S3 (if using)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
S3_BUCKET=your-bucket-name

# Frontend URL
FRONTEND_URL=https://yourdomain.com
```

### 4. Build and Deploy

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Run database migrations
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head

# Check service health
docker-compose -f docker-compose.prod.yml ps
```

### 5. SSL/HTTPS Setup (Optional)

For production, use a reverse proxy like Nginx or Traefik with Let's Encrypt:

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal is configured automatically
```

### 6. Monitoring

```bash
# View all service logs
docker-compose -f docker-compose.prod.yml logs -f

# Monitor resource usage
docker stats

# Check service health
docker-compose -f docker-compose.prod.yml ps
curl http://localhost/api/v1/
```

---

## ⚙️ Configuration

### Environment Variables

See [.env.docker.example](.env.docker.example) for all available options.

**Critical Variables:**

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | ✅ |
| `SECRET_KEY` | JWT signing key (32+ chars) | ✅ |
| `REDIS_PASSWORD` | Redis authentication | ✅ Production |
| `BREVO_API_KEY` | Email service API key | ✅ |
| `GOOGLE_CLIENT_ID` | OAuth client ID | ⚠️ Optional |

### Scaling Services

```bash
# Scale Celery workers
docker-compose -f docker-compose.prod.yml up -d --scale celery-worker=4

# Scale backend (requires load balancer)
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

### Backup and Restore

**Database Backup:**

```bash
# Backup
docker-compose exec postgres pg_dump -U visualml_user visualml_prod > backup_$(date +%Y%m%d).sql

# Restore
cat backup_20260129.sql | docker-compose exec -T postgres psql -U visualml_user visualml_prod
```

**Volume Backup:**

```bash
# Backup uploads and models
docker run --rm -v visual-ml_backend_uploads:/data -v $(pwd):/backup \
  alpine tar czf /backup/uploads_backup.tar.gz -C /data .

# Restore
docker run --rm -v visual-ml_backend_uploads:/data -v $(pwd):/backup \
  alpine tar xzf /backup/uploads_backup.tar.gz -C /data
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Services Not Starting

```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs backend

# Verify environment variables
docker-compose config
```

#### 2. Database Connection Errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres pg_isready -U visualml_user

# Check DATABASE_URL in .env
docker-compose exec backend env | grep DATABASE_URL
```

#### 3. Redis Connection Errors

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping

# If password protected
docker-compose exec redis redis-cli -a your-password ping
```

#### 4. Frontend Not Loading

```bash
# Check nginx logs
docker-compose logs frontend

# Verify backend is accessible
curl http://localhost:8000/api/v1/

# Check nginx configuration
docker-compose exec frontend nginx -t
```

#### 5. Celery Tasks Not Running

```bash
# Check worker logs
docker-compose logs celery-worker

# Verify Redis connection
docker-compose exec celery-worker python -c "from app.core.celery_app import celery_app; print(celery_app.control.inspect().active())"

# Check registered tasks
docker-compose exec celery-worker celery -A app.core.celery_app inspect registered
```

### Port Conflicts

If ports are already in use:

```bash
# Find process using port
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Change ports in docker-compose.yml
# Example: "8001:8000" instead of "8000:8000"
```

### Clean Restart

```bash
# Stop everything
docker-compose down

# Remove volumes (⚠️ deletes data)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

---

## 🚄 Performance Tuning

### Backend Optimization

**Increase Uvicorn Workers:**

Edit `docker-compose.prod.yml`:

```yaml
backend:
  command: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 8
```

**Celery Concurrency:**

```yaml
celery-worker:
  command: celery -A app.core.celery_app worker --loglevel=info --concurrency=8
```

### Database Optimization

**Connection Pooling:**

Edit `.env`:

```bash
DATABASE_URL=postgresql://user:pass@postgres:5432/db?pool_size=20&max_overflow=10
```

**PostgreSQL Configuration:**

Create `postgres.conf`:

```conf
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
```

Mount in `docker-compose.prod.yml`:

```yaml
postgres:
  volumes:
    - ./postgres.conf:/etc/postgresql/postgresql.conf
  command: postgres -c config_file=/etc/postgresql/postgresql.conf
```

### Redis Optimization

Edit `docker-compose.prod.yml`:

```yaml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Resource Limits

Add to `docker-compose.prod.yml`:

```yaml
backend:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 2G
      reservations:
        cpus: '1'
        memory: 1G
```

---

## 📊 Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/

# Frontend health
curl http://localhost/

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

### Resource Usage

```bash
# Real-time stats
docker stats

# Service-specific
docker stats visualml-backend-prod
```

### Logs

```bash
# All services
docker-compose logs -f --tail=100

# Specific service
docker-compose logs -f backend

# Save logs to file
docker-compose logs > logs_$(date +%Y%m%d).txt
```

---

## 🔒 Security Best Practices

1. **Change all default passwords** in `.env`
2. **Use strong SECRET_KEY** (64+ characters)
3. **Enable Redis password** in production
4. **Use HTTPS** with SSL certificates
5. **Limit exposed ports** (only 80/443 public)
6. **Regular updates**: `docker-compose pull && docker-compose up -d`
7. **Backup regularly** (database + volumes)
8. **Monitor logs** for suspicious activity

---

## 📞 Support

- **Documentation**: [setup.md](setup.md)
- **Issues**: [GitHub Issues](https://github.com/Vizuara-AI-Lab/Visual-ML/issues)
- **Email**: admin@visualml.com

---

**Last Updated**: January 2026  
**Version**: 1.0.0
