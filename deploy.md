# 🚀 Vercel Deployment Guide for ReRisk AI

## Important Note About Streamlit on Vercel

**Streamlit apps have limitations on Vercel** because:
- Vercel is designed for serverless functions, not long-running web servers
- Streamlit requires persistent connections for real-time interactivity
- The app may timeout or have performance issues

## Alternative Deployment Options (Recommended)

### 1. **Streamlit Cloud (Recommended)**
- **Free hosting** specifically for Streamlit apps
- **Direct GitHub integration**
- **No configuration needed**

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy automatically

### 2. **Heroku**
- Better suited for Streamlit apps
- Free tier available
- Persistent connections

### 3. **Railway**
- Modern alternative to Heroku
- Good for Python apps
- Free tier available

## If You Still Want to Try Vercel

### Prerequisites
1. Install Vercel CLI: `npm i -g vercel`
2. Have a Vercel account

### Deployment Steps

1. **Initialize Vercel project:**
   ```bash
   vercel login
   vercel init
   ```

2. **Configure environment variables:**
   ```bash
   vercel env add STREAMLIT_SERVER_PORT
   # Enter: 8501
   
   vercel env add STREAMLIT_SERVER_ADDRESS  
   # Enter: 0.0.0.0
   ```

3. **Deploy:**
   ```bash
   vercel --prod
   ```

### Vercel Configuration Files Created

- **`vercel.json`**: Vercel deployment configuration
- **`Procfile`**: Process definition for web servers
- **`runtime.txt`**: Python version specification

## Expected Issues on Vercel

1. **Cold Starts**: App may be slow to load initially
2. **Timeouts**: Long-running computations may timeout
3. **Memory Limits**: Large ML models may exceed memory limits
4. **Session State**: May not persist between requests

## Recommended Solution: Streamlit Cloud

For the best experience, use **Streamlit Cloud**:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/rerisk-ai.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Test on different port (simulate Vercel)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Performance Optimizations for Deployment

The app includes several optimizations:
- `@st.cache_data` for data loading
- Efficient ML model training
- Memory-conscious data processing
- Error handling for missing data

## Monitoring

Once deployed, monitor:
- App startup time
- Memory usage
- User interactions
- Error logs

## Troubleshooting

### Common Issues:
1. **Import errors**: Check all dependencies in requirements.txt
2. **Memory issues**: Reduce model complexity or data size
3. **Timeout errors**: Optimize long-running computations
4. **File not found**: Ensure data files are in the correct location

### Debug Commands:
```bash
# Check Vercel logs
vercel logs

# Local testing with production settings
VERCEL=1 streamlit run app.py
```

## Final Recommendation

**Use Streamlit Cloud for the best experience** - it's specifically designed for Streamlit apps and provides the most reliable hosting for this type of application.
