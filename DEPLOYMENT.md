# Deployment Guide

## Quick Start

### 1. Pre-Deployment Checklist
- [ ] Docker build tested locally (`docker build -t sw-ai-6-api .`)
- [ ] Local container runs successfully (`docker run -p 8000:8000 sw-ai-6-api`)
- [ ] UI accessible at `http://localhost:8000/api/v1/ui`
- [ ] Model weights exist in `model_weights/` directory
- [ ] Knowledge corpus CSV exists (optional: `knowledge_corpus.csv`)

### 2. Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DEVICE` | Device for inference | `cpu` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `ALLOW_ORIGINS` | CORS allowed origins | `*` | No (set in prod) |
| `HOST` | Server bind host | `0.0.0.0` | No |
| `PORT` | Server port | `8000` | No |
| `MODEL_TSN_PATH` | TSN model path | `model_weights/tsn_optical_flow.pth` | No |
| `MODEL_SGCN_PATH` | SGCN model path | `model_weights/sgcn_2d.pth` | No |
| `MODEL_STGCN_PATH` | ST-GCN model path | `model_weights/stgcn_3d.pth` | No |
| `MODEL_FUSION_PATH` | Fusion model path | `model_weights/fusion.pkl` | No |
| `KNOWLEDGE_CORPUS_PATH` | Knowledge CSV path | `knowledge_corpus.csv` | No |
| `SEVERITY_BANDS` | Severity mapping | `0-2:Minimal,2-4:Low,4-7:Moderate,7-10:High` | No |

### 3. Platform-Specific Deployment

#### Railway (Recommended for Simplicity)
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Railway auto-detects Dockerfile
4. Add environment variables in dashboard
5. Deploy!

#### Google Cloud Run
```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/sw-ai-6-api

# Deploy
gcloud run deploy sw-ai-6-api \
  --image gcr.io/YOUR_PROJECT_ID/sw-ai-6-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DEVICE=cpu,LOG_LEVEL=INFO \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10
```

#### AWS ECS Fargate
1. Push image to ECR:
   ```bash
   aws ecr create-repository --repository-name sw-ai-6-api
   docker tag sw-ai-6-api:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/sw-ai-6-api:latest
   docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/sw-ai-6-api:latest
   ```
2. Create Task Definition (2 vCPU, 4GB RAM recommended)
3. Create ECS Service with Application Load Balancer
4. Set environment variables in task definition

#### Render
1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Select Docker as environment
5. Set environment variables
6. Deploy

### 4. Post-Deployment Verification

```bash
# Health check
curl https://your-domain.com/health

# Readiness check
curl https://your-domain.com/ready

# UI access
open https://your-domain.com/api/v1/ui
```

### 5. Monitoring & Logs

- **Health endpoints**: `/health` (liveness), `/ready` (readiness)
- **Logs**: Check platform-specific log viewer
- **Metrics**: Monitor CPU, memory, request latency

### 6. Storage Considerations

- **Uploads**: Ephemeral in container (consider external storage)
- **Reports**: Ephemeral in container (consider S3/GCS/Azure Blob)
- **Model weights**: Should be in image or mounted volume

### 7. Scaling

- **Horizontal**: Use platform auto-scaling (CPU/memory-based)
- **Vertical**: Increase instance size if processing is slow
- **Recommended**: 2 vCPU, 4GB RAM per instance for production

### 8. Security

- [ ] Set `ALLOW_ORIGINS` to specific domain (not `*`)
- [ ] Use HTTPS (handled by platform load balancer)
- [ ] Consider API key authentication for production
- [ ] Review file upload size limits (default: 100MB)

### 9. Cost Optimization

- Use CPU-only instances (no GPU needed)
- Set appropriate instance sizes
- Enable auto-scaling to zero when idle (Cloud Run/Railway)
- Consider spot instances for development

## Troubleshooting

### Build fails
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Ensure model weights are in repository

### Runtime errors
- Check logs: `docker logs <container_id>`
- Verify model paths exist
- Check environment variables

### Slow inference
- Increase instance CPU/memory
- Consider GPU instances (if available)
- Check network latency

