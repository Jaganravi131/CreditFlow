# Deployment Guide for CreditFlow

This guide explains how to deploy the CreditFlow Conversion Attribution Simulator to various cloud platforms.

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - Free & Easy)

Streamlit Community Cloud offers free hosting for Streamlit apps with zero configuration.

**Steps:**

1. **Push your code to GitHub** (already done!)

2. **Visit [share.streamlit.io](https://share.streamlit.io)**

3. **Sign in with GitHub**

4. **Click "New app"**

5. **Fill in the details:**
   - Repository: `Jaganravi131/CreditFlow`
   - Branch: `main` (or your preferred branch)
   - Main file path: `conversion_simulator.py`

6. **Click "Deploy"**

That's it! Streamlit will:
- Automatically detect `requirements.txt`
- Install all dependencies
- Deploy your app
- Give you a public URL like `https://creditflow-your-app.streamlit.app`

**Advantages:**
- ✅ Completely free
- ✅ Zero configuration needed
- ✅ Automatic SSL/HTTPS
- ✅ Easy updates (just push to GitHub)
- ✅ Perfect for demos and prototypes

---

### Option 2: Heroku

Heroku is a popular cloud platform that supports Python apps. The `Procfile` is already configured.

**Steps:**

1. **Install Heroku CLI** (if not already installed):
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create a new Heroku app**:
   ```bash
   heroku create creditflow-demo
   ```

4. **Add Python buildpack**:
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Deploy**:
   ```bash
   git push heroku main
   ```

6. **Open your app**:
   ```bash
   heroku open
   ```

**Configuration:**

The `Procfile` is already set up:
```
web: streamlit run conversion_simulator.py --server.port=$PORT --server.address=0.0.0.0
```

**Advantages:**
- ✅ Production-ready platform
- ✅ Easy scaling
- ✅ Built-in monitoring
- ✅ Custom domains supported

**Note:** Heroku's free tier was discontinued, but they offer affordable paid plans starting at $5/month.

---

### Option 3: Docker (Any Platform)

Deploy as a Docker container to AWS, Google Cloud, Azure, or any Docker-compatible platform.

**Create a Dockerfile:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "conversion_simulator.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run locally:**
```bash
docker build -t creditflow .
docker run -p 8501:8501 creditflow
```

**Deploy to cloud:**
- **AWS ECS/Fargate**: Push to ECR, create a service
- **Google Cloud Run**: `gcloud run deploy creditflow --source .`
- **Azure Container Instances**: Deploy from Docker Hub or ACR

---

### Option 4: Railway

Railway offers simple deployment with GitHub integration.

**Steps:**

1. **Visit [railway.app](https://railway.app)**

2. **Sign in with GitHub**

3. **Click "New Project"**

4. **Select "Deploy from GitHub repo"**

5. **Choose the CreditFlow repository**

6. **Railway will auto-detect and deploy**

**Advantages:**
- ✅ Free tier available ($5 credit/month)
- ✅ Automatic deployments from GitHub
- ✅ Simple dashboard
- ✅ Fast setup

---

### Option 5: Render

Render is another platform-as-a-service option similar to Heroku.

**Steps:**

1. **Visit [render.com](https://render.com)**

2. **Sign up/Login with GitHub**

3. **Click "New +" → "Web Service"**

4. **Connect your GitHub repository**

5. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run conversion_simulator.py --server.port=$PORT --server.address=0.0.0.0`

6. **Click "Create Web Service"**

**Advantages:**
- ✅ Free tier available
- ✅ Automatic SSL
- ✅ Easy scaling
- ✅ Good performance

---

## 🔧 Environment Variables

If you need to configure the app differently for production, you can use environment variables:

```python
import os

# Example: Different port for production
port = int(os.environ.get('PORT', 8501))
```

For most deployments, the default configuration works perfectly.

---

## 🚀 Quick Comparison

| Platform | Cost | Setup Difficulty | Best For |
|----------|------|------------------|----------|
| **Streamlit Cloud** | Free | ⭐ Very Easy | Demos, MVPs |
| **Heroku** | $5+/month | ⭐⭐ Easy | Small apps |
| **Railway** | $5+/month | ⭐⭐ Easy | Startups |
| **Render** | Free tier + paid | ⭐⭐ Easy | Production apps |
| **Docker/Cloud** | Varies | ⭐⭐⭐ Moderate | Enterprise |

---

## 📊 Monitoring

After deployment, monitor your app's performance:

- **Streamlit Cloud**: Built-in logs and analytics
- **Heroku**: `heroku logs --tail`
- **Railway/Render**: Dashboard with logs and metrics
- **Docker**: Use cloud provider's monitoring tools

---

## 🔒 Security Notes

1. **Secrets**: Never commit sensitive data. Use environment variables.
2. **Authentication**: For production, consider adding authentication (Streamlit supports this).
3. **HTTPS**: All recommended platforms provide automatic SSL/HTTPS.
4. **Rate Limiting**: Consider implementing rate limiting for public deployments.

---

## 💡 Recommendations

**For this project (CreditFlow demo):**
- ✅ **Use Streamlit Community Cloud** - It's free, easy, and perfect for demos
- The app is already optimized for Streamlit Cloud
- No configuration changes needed
- Deploys in under 5 minutes

**Next steps after deployment:**
1. Share the public URL with stakeholders
2. Monitor usage and feedback
3. Iterate based on user needs
4. Consider adding authentication if needed

---

## 🆘 Troubleshooting

**Issue**: App takes too long to load
**Solution**: Models are large. Consider model compression or caching strategies.

**Issue**: Memory errors
**Solution**: Most free tiers have 512MB RAM. Upgrade to paid tier if needed.

**Issue**: Port conflicts
**Solution**: Streamlit uses port 8501 by default. Cloud platforms handle this automatically.

---

For more help, refer to:
- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Railway Docs](https://docs.railway.app/)
