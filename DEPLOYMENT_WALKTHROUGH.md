# Autism Screening API - Deployment Walkthrough & System Architecture

---

## System Architecture & Flow

### 📁 File Structure
```
app/
├── main.py                 # FastAPI entry point
├── api/
│   ├── endpoints.py        # Basic screening endpoints
│   └── enhanced_endpoints.py  # Enhanced endpoints with explanations
├── core/
│   ├── config.py           # Application settings
│   ├── models.py           # Pydantic models (CREATED)
│   └── enhanced_models.py  # Enhanced response models
├── models/
│   ├── predictor.py        # Main prediction pipeline
│   ├── enhanced_predictor.py  # Enhanced predictor with knowledge guidance
│   ├── model_loader.py     # Loads trained models
│   ├── stgcn.py           # ST-GCN model architecture
│   └── graph_utils.py     # Graph utilities for skeleton processing
├── services/
│   ├── video_processor.py  # Video preprocessing & feature extraction
│   ├── optical_flow.py     # Optical flow computation
│   └── knowledge_guide.py  # Knowledge-guided explanations
└── static/
    └── index.html          # Web UI
```

---

## System Flow (Request to Response)

### 1. Application Startup (`app/main.py`)

**Flow:**
```
Startup → Load Configuration → Initialize Services → Ready to Serve
```

**Code Path:**
```python
# Startup sequence
@app.on_event("startup")
async def startup_event():
    # 1. Loads settings from app/core/config.py
    # 2. Initializes predictor (loads 4 models)
    # 3. Initializes video processor
    # 4. API ready to receive requests
```

**Models Loaded:**
- TSN (Temporal Segment Network) - Optical flow analysis
- SGCN (Spatial GCN) - 2D skeleton analysis
- STGCN (Spatial-Temporal GCN) - 3D skeleton analysis
- Fusion model - Combines all predictions

---

### 2. Request Handling (`app/api/endpoints.py`)

**Endpoint: POST `/api/v1/screen`**

**Flow:**
```
Upload Video → Validate → Save Temp File → Process → Predict → Return → Cleanup
```

**Code Path:**
```python
@router.post("/screen")
async def screen_video(file: UploadFile):
    # 1. Validate file type and size
    _validate_video_file(file)
    
    # 2. Save to temporary file
    temp_filepath = save_to_temp(file)
    
    # 3. Process video
    model_inputs = video_processor.process_video(temp_filepath)
    
    # 4. Run prediction
    result = predictor.predict(model_inputs)
    
    # 5. Return formatted response
    return ScreeningResponse(...)
    
    # 6. Cleanup (background task)
    background_tasks.add_task(_cleanup_temp_file, temp_filepath)
```

---

### 3. Video Processing (`app/services/video_processor.py`)

**Method: `process_video(video_path)`**

**Flow:**
```
Extract Frames → Optical Flow (TSN) → 2D Skeleton (SGCN) → 3D Skeleton (STGCN) → Return Tensors
```

**Details:**
- **Extracts 64 frames** from video (uniform sampling)
- **For TSN (10 frames):** Computes optical flow between consecutive frames
- **For SGCN (4 frames):** Extracts 2D skeleton using MediaPipe Pose (24 joints)
- **For STGCN (32 frames):** Extracts 3D skeleton using ROMP/MediaPipe (24 joints)

**Output:**
```python
{
    'tsn': torch.Tensor,    # [1, 10, 3, 224, 224]
    'sgcn': torch.Tensor,   # [1, 4, 24, 2]
    'stgcn': torch.Tensor  # [1, 3, 32, 24]
}
```

---

### 4. Prediction Pipeline (`app/models/predictor.py`)

**Method: `predict(model_inputs)`**

**Flow:**
```
TSN Prediction → SGCN Prediction → STGCN Prediction → Fusion → Interpretation → Return
```

**Code Path:**
```python
def predict(self, model_inputs):
    # 1. Run individual models
    tsn_score = self.predict_tsn(model_inputs['tsn'])
    sgcn_score = self.predict_sgcn(model_inputs['sgcn'])
    stgcn_score = self.predict_stgcn(model_inputs['stgcn'])
    
    # 2. Fuse predictions
    fusion_result = self.fuse_predictions(tsn_score, sgcn_score, stgcn_score)
    
    # 3. Interpret score
    interpretation = self.interpret_score(fusion_result['final_score'])
    
    # 4. Return result
    return {
        'prediction': {...},
        'component_analysis': {...},
        'interpretation': {...},
        'processing_metadata': {...}
    }
```

**Score Interpretation:**
- **1-3:** Minimal evidence
- **4-5:** Mild symptoms
- **6-7:** Moderate symptoms
- **8-10:** Severe symptoms

---

### 5. Enhanced Endpoint (`app/api/enhanced_endpoints.py`)

**Endpoint: POST `/api/v1/screen-with-explanation`**

**Additional Features:**
- Uses `EnhancedAutismScreeningPredictor`
- Adds knowledge-guided explanations
- Maps predictions to clinical domains

**Flow:**
```
Base Prediction → Knowledge Guidance → Domain Mapping → Clinical Recommendations → Return
```

**Knowledge Guidance (`app/services/knowledge_guide.py`):**
- Identifies dominant models (which are driving the prediction)
- Maps to diagnostic domains (Social Communication, Motor, etc.)
- Provides clinical recommendations based on score and domains

---

## Model Loading (`app/models/model_loader.py`)

**Singleton Pattern:** Loads models once at startup

**Models:**
1. **TSN** (`model_weights/tsn_optical_flow.pth`)
2. **SGCN** (`model_weights/sgcn_2d.pth`)
3. **STGCN** (`model_weights/stgcn_3d.pth`) - Uses custom architecture
4. **Fusion** (`model_weights/fusion.pkl`) - GradientBoostingRegressor

**Important:** Models must exist in `model_weights/` directory

---

## Configuration (`app/core/config.py`)

**Key Settings:**
- Device: `cuda` (GPU) or `cpu`
- Frame size: `(224, 224)`
- Max video size: `100MB`
- Allowed formats: `.mp4, .avi, .mov, .mkv`

---

## Deployment Checklist for Crane Cloud

### 📦 Required Files
1. **Model weights** (must be present in `model_weights/`):
   - `tsn_optical_flow.pth`
   - `sgcn_2d.pth`
   - `stgcn_3d.pth`
   - `fusion.pkl`

2. **Knowledge corpus**:
   - `knowledge_corpus.csv` (now created at root)

3. **Dependencies**:
   - All packages listed in `requirements.txt`

###  Deployment Steps

1. **Build Docker Image:**
```bash
docker build -t autism-screening-api .
```

2. **Run Locally:**
```bash
docker run -p 8000:8000 autism-screening-api
```

3. **For Crane Cloud:**
   - Push to your git repository
   - Create deployment with Dockerfile
   - Ensure model files are uploaded to `model_weights/` directory
   - Set environment variables if needed (in `.env` file or Crane Cloud config)

###  Important Notes

1. **Model Files:** Make sure all `.pth` and `.pkl` files are committed to the repository or uploaded separately
2. **CUDA:** If deploying without GPU, change `DEVICE` in config.py to `"cpu"`
3. **Memory:** Model loading may require significant RAM during startup
4. **Health Check:** Endpoint available at `/health`

### 🔧 Environment Configuration

**Optional `.env` file:**
```env
DEVICE=cuda
API_TITLE=Autism Screening API
API_VERSION=1.0.0
```

---

## Testing Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Root Info
```bash
curl http://localhost:8000/
```

### 3. Screen Video (requires multipart/form-data)
```bash
curl -X POST "http://localhost:8000/api/v1/screen" \
  -F "file=@/path/to/video.mp4"
```

### 4. Get Model Info
```bash
curl http://localhost:8000/api/v1/model-info
```

### 5. API Documentation
```bash
# Open in browser
http://localhost:8000/docs
```

---

## Potential Issues & Solutions

### Issue 1: Models Not Found
**Error:** `Model {name} is not loaded`
**Solution:** Ensure model files are in `model_weights/` directory

### Issue 2: CUDA Out of Memory
**Error:** `CUDA out of memory`
**Solution:** Change `DEVICE` to `"cpu"` in `config.py` or reduce batch size

### Issue 3: MediaPipe Pose Detection Fails
**Error:** `Error initializing pose estimators`
**Solution:** This is handled gracefully - falls back to zeros if detection fails

### Issue 4: Video Processing Fails
**Error:** `Cannot open video file`
**Solution:** Ensure video format is supported and codecs are installed (ffmpeg in Docker)

---

## System Components Summary

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **API Layer** | Endpoints & request handling | `main.py`, `api/endpoints.py`, `api/enhanced_endpoints.py` |
| **Prediction Engine** | Model inference | `models/predictor.py`, `models/enhanced_predictor.py` |
| **Video Processing** | Feature extraction | `services/video_processor.py`, `services/optical_flow.py` |
| **Knowledge System** | Explanations | `services/knowledge_guide.py` + `knowledge_corpus.csv` |
| **Models** | Neural networks | `model_weights/*.pth`, `model_weights/*.pkl` |
| **Configuration** | Settings | `core/config.py` |

---

## Final Verification

Before deploying, ensure:
1. ✅ All Python files have no syntax errors
2. ✅ Model files are present in `model_weights/`
3. ✅ `knowledge_corpus.csv` exists at root
4. ✅ Exception handlers are registered on app (not router)
5. ✅ Logging is properly configured in all modules
6. ✅ Dockerfile builds successfully
7. ✅ Requirements.txt includes all dependencies

---

## Architecture Diagram

```
Client Request
    ↓
FastAPI (app/main.py)
    ↓
Endpoint Router (api/endpoints.py)
    ↓
Video Processor (services/video_processor.py)
    ├─→ Extract Frames
    ├─→ Optical Flow (services/optical_flow.py)
    ├─→ 2D Skeleton (MediaPipe)
    └─→ 3D Skeleton (ROMP/MediaPipe)
    ↓
Model Inputs (tensors)
    ↓
Predictor (models/predictor.py)
    ├─→ TSN Model
    ├─→ SGCN Model
    ├─→ STGCN Model
    └─→ Fusion Model
    ↓
Predictions & Interpretation
    ↓
Enhanced Predictor (optional - models/enhanced_predictor.py)
    └─→ Knowledge Guide (services/knowledge_guide.py)
    ↓
Response (JSON)
    ↓
Client
```

---

Your system is now **deployment-ready**! 🚀

