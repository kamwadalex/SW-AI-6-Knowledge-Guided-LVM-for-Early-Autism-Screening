## SW-AI-6 FastAPI Service

Knowledge-Guided LVM API for early autism screening. Includes video upload UI, on-the-fly feature extraction (optical flow, 2D pose, 3D pose with ROMP fallback), three-model inference (TSN/SGCN/ST-GCN), score fusion, knowledge-guided explanations, and PDF report generation.

### Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
set APP_NAME="SW-AI-6 API"
set ENVIRONMENT=development
set LOG_LEVEL=INFO
set ALLOW_ORIGINS=*
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/api/v1/ui` for the web UI, or `http://localhost:8000/docs` for interactive docs.

Notes:
- Upload size limit: 100 MB
- GPU recommended; install torch/torchvision matching your CUDA on target

### Endpoints

- `GET /health` – liveness
- `GET /ready` – readiness
- `GET /api/v1/ui` – web interface (upload video, view results, download PDF)
- `POST /api/v1/infer` – run inference on uploaded video (multipart)
- `POST /api/v1/report/{report_id}` – generate PDF from a summary JSON
- `GET /api/v1/report/{report_id}` – download generated PDF

### Container build and run

```powershell
docker build -t sw-ai-6-api .
docker run --rm -p 8000:8000 -e LOG_LEVEL=INFO sw-ai-6-api
```

### Configuration (env vars)

- `APP_NAME` – service name
- `ENVIRONMENT` – e.g. `development` | `production`
- `LOG_LEVEL` – `DEBUG` | `INFO` | `WARNING` | `ERROR`
- `ALLOW_ORIGINS` – comma-separated origins for CORS
- `HOST`, `PORT` – server bind
- `DEVICE` – `auto` | `cpu` | `cuda`
- `MODEL_TSN_PATH` – default `model_weights/tsn_optical_flow.pth`
- `MODEL_SGCN_PATH` – default `model_weights/sgcn_2d.pth`
- `MODEL_STGCN_PATH` – default `model_weights/stgcn_3d.pth`
- `MODEL_FUSION_PATH` – default `model_weights/fusion.pkl`
- `KNOWLEDGE_CORPUS_PATH` – optional CSV with columns: `Linked_Model,Category,Description,Severity_Indicator,Clinical_References`
- `SEVERITY_BANDS` – default `0-2:Minimal,3-4:Low,5-7:Moderate,8-10:High`

### Deploying to cloud (generic)

1. Build the image and push to your registry.
2. Deploy a container service (Render, Fly.io, Azure Container Apps, AWS ECS/Fargate, etc.).
3. Set env vars in your provider.
4. Expose port `8000` with HTTP load balancer.

Torch/vision on GPU:
- The `requirements.txt` does not pin torch/torchvision. Install them separately to match your cloud GPU/CUDA, e.g. (CUDA 12.1):
```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
For CPU-only:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Project structure

```text
app/
  api/
    v1/
      endpoints/
        example.py
        health.py
        inference.py
        report.py
        ui.py
      __init__.py
      api.py
    __init__.py
  core/
    config.py
    logging.py
  middleware/
    limits.py
  services/
    feature_extraction.py
    fusion.py
    knowledge.py
    knowledge_guidance.py
    models/
      tsn.py
      sgcn.py
      stgcn.py
    pipeline.py
  __init__.py
  main.py
Dockerfile
requirements.txt
.dockerignore
README.md
model_weights/
knowledge_corpus.csv (optional)
```

### Using the UI
1) Navigate to `http://localhost:8000/api/v1/ui`
2) Choose a video (≤ 100 MB). Optionally tick “Use mock scores” for a quick demo.
3) Click Run Inference. The JSON results render inline.
4) A “Download PDF report” link appears after PDF generation.

### Knowledge Guidance
- If `KNOWLEDGE_CORPUS_PATH` is set, clinical domains and references are read from the CSV.
- Otherwise, sensible defaults are used to map TSN/SGCN/ST-GCN to domains.

### Notes on ROMP
- 3D pose uses ROMP if available. If ROMP import fails, a safe fallback keeps the pipeline running. You can later integrate a lighter 3D estimator if needed.


