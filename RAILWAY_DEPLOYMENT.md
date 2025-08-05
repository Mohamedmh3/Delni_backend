# Railway Deployment Guide for DELNI Backend

## 🚀 Quick Deploy to Railway

### 1. Prerequisites
- Railway account
- MongoDB Atlas database
- PostgreSQL database (Railway will provide this)

### 2. Deploy to Railway

1. **Connect your GitHub repository to Railway**
2. **Create a new service** from your GitHub repo
3. **Add a PostgreSQL database** in Railway dashboard
4. **Set environment variables** (see below)

### 3. Required Environment Variables

Set these in Railway's environment variables section:

```bash
# Django Settings
DJANGO_SECRET_KEY=your-super-secret-key-here
DJANGO_DEBUG=False
ENVIRONMENT=production

# MongoDB Settings (from MongoDB Atlas)
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority
MONGODB_DATABASE=bus
MONGODB_COLLECTION=bus

# CORS Settings
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Host Settings (optional - Railway auto-detection handles this)
ALLOWED_HOSTS=delnibackend-production.up.railway.app,.railway.app

# Optional Settings
SENTRY_DSN=your-sentry-dsn-if-using
CELERY_ENABLED=False
DOCKER_ENABLED=False
KUBERNETES_ENABLED=False
```

### 4. Railway Auto-Configuration

Railway will automatically set:
- `DATABASE_URL` - PostgreSQL connection string
- `RAILWAY_ENVIRONMENT` - Set to 'production'
- `RAILWAY_SERVICE_NAME` - Your service name

### 5. Health Check Endpoints

After deployment, test these endpoints:

- **Root**: `https://your-app.railway.app/`
- **Health Check**: `https://your-app.railway.app/health/`
- **API Health**: `https://your-app.railway.app/api/health/`
- **Swagger Docs**: `https://your-app.railway.app/swagger/`

### 6. Troubleshooting

#### Common Issues:

1. **psycopg2 error**: Fixed by adding `psycopg2-binary` to requirements.txt
2. **Static files**: Fixed by adding whitenoise middleware
3. **Environment detection**: Fixed by auto-detecting Railway environment
4. **Database connection**: Fixed by parsing Railway's DATABASE_URL

#### Debug Endpoints:

- `/debug/` - Check settings configuration
- `/test-mongo/` - Test MongoDB connection
- `/inspect-db/` - Inspect MongoDB database

### 7. Files Added/Modified

- ✅ `requirements.txt` - Added PostgreSQL and production dependencies
- ✅ `Procfile` - Railway deployment configuration
- ✅ `runtime.txt` - Python version specification
- ✅ `settings.py` - Railway auto-detection and PostgreSQL configuration
- ✅ `urls.py` - Added health check endpoints
- ✅ `RAILWAY_DEPLOYMENT.md` - This deployment guide

### 8. Next Steps

1. Deploy to Railway
2. Set environment variables
3. Test health endpoints
4. Configure custom domain (optional)
5. Set up monitoring (optional)

## 🎉 Your app should now deploy successfully on Railway! 