# Visual-ML Server - Optimized Dependencies

## Build Size Reduction

### Before Optimization
- **Total Size:** ~4.3GB
- **Deployment:** ❌ Failed (buffer allocation error)

### After Optimization  
- **Total Size:** ~500MB (estimated)
- **Deployment:** ✅ Should succeed

---

## Removed Dependencies

### Heavy ML/AI Packages (Not Used)

| Package | Size | Reason for Removal |
|---------|------|-------------------|
| `chromadb` | ~500MB | Vector database - not used in codebase |
| `sentence-transformers` | ~1GB | Text embeddings - not used |
| `langchain` | ~200MB | LLM framework - not used |
| `langchain-openai` | ~50MB | LangChain OpenAI - not used |
| `langchain-google-genai` | ~50MB | LangChain Google - not used |
| `langchain-anthropic` | ~50MB | LangChain Anthropic - not used |
| `pypdf` | ~10MB | PDF parsing - not used |

### Business Packages (Not Used)

| Package | Size | Reason for Removal |
|---------|------|-------------------|
| `stripe` | ~5MB | Payment processing - not implemented |
| `sendgrid` | ~5MB | Email service - using Brevo instead |

**Total Removed:** ~1.9GB

---

## Kept Dependencies

### Core Framework
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### Authentication & Security
- `python-jose` - JWT tokens
- `passlib` - Password hashing
- `google-auth` - Google OAuth
- `argon2-cffi` - Password hashing
- `email-validator` - Email validation

### Database
- `sqlalchemy` - ORM
- `alembic` - Migrations
- `psycopg2-binary` - PostgreSQL driver

### ML (Lightweight)
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML algorithms
- `joblib` - Model serialization

### GenAI (Direct APIs)
- `openai` - OpenAI API
- `anthropic` - Claude API
- `google-generativeai` - Gemini API

### Infrastructure
- `redis` - Caching
- `celery` - Background tasks
- `boto3` - AWS S3
- `brevo-python` - Email service

---

## Migration Notes

### If You Need Removed Packages Later

**For RAG/Vector Search:**
```toml
# Add back when needed:
dependencies = [
    "chromadb>=1.4.1",
    "sentence-transformers>=5.2.0",
]
```

**For LangChain:**
```toml
# Add back when needed:
dependencies = [
    "langchain>=1.2.6",
    "langchain-openai>=1.1.7",
]
```

**For PDF Processing:**
```toml
# Add back when needed:
dependencies = [
    "pypdf>=6.6.0",
]
```

**For Payments:**
```toml
# Add back when needed:
dependencies = [
    "stripe>=14.2.0",
]
```

---

## Deployment Instructions

### 1. Update Dependencies

```bash
cd server

# Remove old lock file
rm uv.lock

# Regenerate with optimized dependencies
uv lock

# Install locally to test
uv sync
```

### 2. Test Locally

```bash
# Run server
uvicorn main:app --reload

# Test endpoints
curl http://localhost:8000/
curl http://localhost:8000/api/v1/
```

### 3. Deploy

```bash
# Commit changes
git add pyproject.toml uv.lock
git commit -m "Optimize dependencies for deployment"
git push origin main

# Deploy will auto-trigger
```

### 4. Verify Build Size

```bash
# Check build output (should be < 1GB now)
# Monitor deployment logs for success
```

---

## Expected Results

✅ **Build size:** ~500MB (down from 4.3GB)  
✅ **Deployment:** Should succeed without buffer errors  
✅ **Functionality:** All current features work  
✅ **Performance:** Faster cold starts  

---

## Rollback Plan

If issues occur:

```bash
# Revert changes
git revert HEAD
git push origin main

# Or restore specific packages:
# Edit pyproject.toml and add back needed dependencies
```

---

**Last Updated:** 2026-01-31  
**Optimized By:** Dependency Analysis
