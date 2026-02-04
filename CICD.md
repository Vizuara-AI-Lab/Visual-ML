# CI/CD Setup Guide

Complete guide for setting up automated CI/CD pipelines with GitHub Actions.

## 📋 Overview

The Visual-ML project includes three GitHub Actions workflows:

1. **CI (Continuous Integration)**: Automated testing and linting
2. **Build & Push**: Build and push Docker images to registry
3. **Deploy**: Automated deployment to staging/production

---

## 🔧 Prerequisites

### 1. GitHub Repository

Ensure your code is pushed to GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/Visual-ML.git
git push -u origin main
```

### 2. Docker Hub Account

Create an account at [hub.docker.com](https://hub.docker.com) and create two repositories:
- `visual-ml-backend`
- `visual-ml-frontend`

---

## ⚙️ GitHub Secrets Configuration

Navigate to: **Repository Settings → Secrets and variables → Actions**

### Required Secrets

#### Docker Registry

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DOCKER_USERNAME` | Docker Hub username | `youruser` |
| `DOCKER_PASSWORD` | Docker Hub access token | `dckr_pat_xxxxx` |

**How to get Docker Hub token:**
1. Go to [Docker Hub Account Settings](https://hub.docker.com/settings/security)
2. Click "New Access Token"
3. Name it "GitHub Actions" and copy the token

#### Database

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `POSTGRES_USER` | PostgreSQL username | `visualml_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `secure_password_123` |
| `POSTGRES_DB` | Database name | `visualml_prod` |
| `DATABASE_URL` | Full database connection string | `postgresql://user:pass@host:5432/db` |

#### Redis

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `REDIS_PASSWORD` | Redis authentication password | `redis_secure_pass` |

#### Application Security

| Secret Name | Description | How to Generate |
|-------------|-------------|-----------------|
| `SECRET_KEY` | JWT signing key | `openssl rand -hex 32` |

#### Email Service (Brevo)

| Secret Name | Description | Where to Find |
|-------------|-------------|---------------|
| `BREVO_API_KEY` | Brevo API key | [Brevo Dashboard](https://app.brevo.com/settings/keys/api) |

#### OAuth (Optional)

| Secret Name | Description | Where to Find |
|-------------|-------------|---------------|
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) |
| `GOOGLE_CLIENT_SECRET` | Google OAuth secret | Google Cloud Console |

#### AWS S3 (Optional)

| Secret Name | Description | Where to Find |
|-------------|-------------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS access key | [AWS IAM Console](https://console.aws.amazon.com/iam/) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | AWS IAM Console |
| `S3_BUCKET` | S3 bucket name | `visual-ml-datasets` |

#### Deployment Server

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `SERVER_HOST` | Server IP or hostname | `123.45.67.89` |
| `SERVER_USER` | SSH username | `ubuntu` |
| `SSH_PRIVATE_KEY` | SSH private key | Contents of `~/.ssh/id_rsa` |
| `FRONTEND_URL` | Production domain | `https://visualml.com` |

#### Build Configuration

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `VITE_API_URL` | Backend API URL for frontend | `https://api.visualml.com` |

#### Notifications (Optional)

| Secret Name | Description | Where to Find |
|-------------|-------------|---------------|
| `SLACK_WEBHOOK` | Slack webhook for notifications | [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks) |

---

## 🚀 Workflow Details

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Jobs:**
- **Backend Tests**: Linting, testing, coverage
- **Frontend Tests**: Linting, type checking, build verification
- **Security Scan**: Vulnerability scanning with Trivy

**No secrets required** - runs automatically on every push/PR.

### 2. Build & Push Workflow (`.github/workflows/build-push.yml`)

**Triggers:**
- Push to `main` branch
- Git tags matching `v*.*.*`
- Manual dispatch

**Required Secrets:**
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `VITE_API_URL` (for frontend build)

**What it does:**
1. Builds Docker images for backend and frontend
2. Tags images with:
   - `latest` (for main branch)
   - Git commit SHA
   - Semantic version (for tags)
3. Pushes to Docker Hub
4. Caches layers for faster builds

### 3. Deploy Workflow (`.github/workflows/deploy.yml`)

**Triggers:**
- Manual dispatch (with environment selection)
- Successful completion of Build & Push workflow

**Required Secrets:** All secrets listed above

**What it does:**
1. Connects to deployment server via SSH
2. Creates `.env` file with production secrets
3. Pulls latest Docker images
4. Runs database migrations
5. Deploys with zero-downtime rolling update
6. Performs health checks
7. Rolls back on failure
8. Sends Slack notification (if configured)

---

## 📝 Usage Examples

### Running CI Tests

CI runs automatically on every push and PR. To trigger manually:

1. Go to **Actions** tab in GitHub
2. Select **CI - Continuous Integration**
3. Click **Run workflow**

### Building and Pushing Images

**Automatic:** Push to `main` branch

```bash
git checkout main
git pull origin main
git push origin main
```

**Manual:** Trigger workflow

1. Go to **Actions** → **Build and Push Docker Images**
2. Click **Run workflow**
3. Select branch and click **Run**

**Release:** Create a version tag

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Deploying to Production

**Option 1: Manual Deployment**

1. Go to **Actions** → **Deploy to Production**
2. Click **Run workflow**
3. Select environment (staging/production)
4. Click **Run workflow**

**Option 2: Automatic Deployment**

Deployment runs automatically after successful image build on `main` branch.

---

## 🖥️ Server Setup for Deployment

### 1. Create Deployment User

```bash
# On your server
sudo useradd -m -s /bin/bash deploy
sudo usermod -aG docker deploy
sudo mkdir -p /opt/visual-ml
sudo chown deploy:deploy /opt/visual-ml
```

### 2. Generate SSH Key

```bash
# On your local machine
ssh-keygen -t ed25519 -C "github-actions" -f ~/.ssh/github_actions

# Copy public key to server
ssh-copy-id -i ~/.ssh/github_actions.pub deploy@your-server-ip

# Add private key to GitHub Secrets as SSH_PRIVATE_KEY
cat ~/.ssh/github_actions
```

### 3. Install Docker on Server

```bash
# On your server
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker deploy
```

### 4. Create Application Directory

```bash
# On your server as deploy user
mkdir -p /opt/visual-ml
cd /opt/visual-ml
```

---

## 🔍 Monitoring Workflows

### View Workflow Runs

1. Go to **Actions** tab in GitHub
2. Click on a workflow to see runs
3. Click on a run to see details and logs

### Check Deployment Status

```bash
# SSH into server
ssh deploy@your-server-ip

# Check running containers
docker ps

# View logs
cd /opt/visual-ml
docker-compose logs -f
```

### Health Checks

```bash
# Backend
curl https://your-domain.com/api/v1/

# Frontend
curl https://your-domain.com/
```

---

## 🐛 Troubleshooting

### CI Tests Failing

**Check logs:**
1. Go to Actions → Failed workflow
2. Click on failed job
3. Expand failed step to see error

**Common issues:**
- Linting errors: Run `npm run lint` or `ruff check .` locally
- Test failures: Run `pytest` or `npm test` locally
- Type errors: Run `npx tsc --noEmit` locally

### Build Failing

**Docker Hub authentication:**
- Verify `DOCKER_USERNAME` and `DOCKER_PASSWORD` are correct
- Ensure Docker Hub token has write permissions

**Build errors:**
- Check Dockerfile syntax
- Verify all dependencies are in `pyproject.toml` or `package.json`

### Deployment Failing

**SSH connection:**
```bash
# Test SSH connection locally
ssh -i ~/.ssh/github_actions deploy@your-server-ip
```

**Server issues:**
```bash
# On server, check Docker
docker ps
docker-compose ps

# Check disk space
df -h

# Check logs
docker-compose logs
```

**Environment variables:**
- Verify all required secrets are set in GitHub
- Check `.env` file on server

---

## 🔄 Rollback Procedure

### Manual Rollback

```bash
# SSH into server
ssh deploy@your-server-ip
cd /opt/visual-ml

# Pull previous image version
docker pull youruser/visual-ml-backend:previous-tag
docker pull youruser/visual-ml-frontend:previous-tag

# Update docker-compose.yml to use specific tags
# Then restart
docker-compose down
docker-compose up -d
```

### Automatic Rollback

The deploy workflow includes automatic rollback on failure. If health checks fail, it will:
1. Stop new containers
2. Restart previous containers
3. Notify via Slack (if configured)

---

## 📊 Best Practices

### 1. Branch Strategy

- `main`: Production-ready code
- `develop`: Development branch
- `feature/*`: Feature branches

**Workflow:**
```bash
# Create feature branch
git checkout -b feature/new-feature develop

# Make changes and push
git push origin feature/new-feature

# Create PR to develop
# After review, merge to develop

# When ready for production, merge develop to main
git checkout main
git merge develop
git push origin main
```

### 2. Versioning

Use semantic versioning for releases:

```bash
# Major release (breaking changes)
git tag -a v2.0.0 -m "Major release with breaking changes"

# Minor release (new features)
git tag -a v1.1.0 -m "Added new features"

# Patch release (bug fixes)
git tag -a v1.0.1 -m "Bug fixes"

git push origin --tags
```

### 3. Environment Management

- **Staging**: Test deployments before production
- **Production**: Stable, user-facing environment

Use GitHub Environments for approval gates:
1. Go to **Settings → Environments**
2. Create `staging` and `production` environments
3. Add protection rules (required reviewers, wait timer)

### 4. Monitoring

- Enable Slack notifications for deployment status
- Set up uptime monitoring (e.g., UptimeRobot, Pingdom)
- Configure log aggregation (e.g., Papertrail, Loggly)

---

## 🔐 Security Considerations

1. **Rotate secrets regularly** (every 90 days)
2. **Use environment-specific secrets** (staging vs production)
3. **Limit SSH key permissions** (deploy user only)
4. **Enable 2FA** on GitHub and Docker Hub
5. **Review workflow logs** for sensitive data leaks
6. **Use secret scanning** (GitHub Advanced Security)

---

## 📞 Support

- **GitHub Actions Docs**: [docs.github.com/actions](https://docs.github.com/actions)
- **Docker Hub**: [hub.docker.com](https://hub.docker.com)
- **Issues**: [GitHub Issues](https://github.com/Vizuara-AI-Lab/Visual-ML/issues)

---

**Last Updated**: January 2026  
**Version**: 1.0.0
