```markdown
# Autism Screening System

A multimodal AI system for early autism screening using computer vision and machine learning. This system analyzes video inputs through three specialized models and provides clinically-informed assessments with explainable outputs.

## 🎯 Overview

This system combines optical flow analysis, 2D/3D skeleton tracking, and knowledge-guided reasoning to screen for autism spectrum disorder indicators. It outputs ADOS (Autism Diagnostic Observation Schedule) comparison scores with clinical domain explanations.

```
System Architecture:
Raw Video → [TSN + SGCN + ST-GCN] → Fusion Meta-Regressor → Clinical Report
```

## 🏗️ System Architecture

### Model Pipeline
- **TSN (Temporal Segment Networks)**: Optical flow analysis for social communication patterns
- **SGCN (Spatial Graph CNN)**: 2D skeleton analysis for body posture and gestures  
- **ST-GCN (Spatial-Temporal Graph CNN)**: 3D skeleton analysis for complex behavioral sequences
- **Fusion Meta-Regressor**: Gradient Boosting model that combines all predictions
- **Knowledge Guidance**: Maps predictions to clinical domains with explanations

### Technical Specifications
| Model | Parameters | Input Shape | Focus Areas |
|-------|------------|-------------|-------------|
| TSN | 11.27M | [1, 10, 3, 224, 224] | Social Communication |
| SGCN | 41.7K | [1, 4, 24, 2] | Body Posture |
| ST-GCN | 501K | [1, 3, 32, 24] | Temporal Patterns |
| Fusion | 5K | [3] | Model Integration |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/autism-screening-system.git
cd autism-screening-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights**
```bash
# Place these in model_weights/ directory:
# - tsn_optical_flow.pth
# - sgcn_2d.pth  
# - stgcn_3d.pth
# - fusion.pkl
# - knowledge_corpus.csv
```

### Usage

1. **Start the API server**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. **Access the web interface**
```
http://localhost:8000
```

3. **Upload a video for analysis**
- Supported formats: MP4, AVI, MOV, MKV
- Maximum size: 100MB
- Recommended length: 5-30 seconds

## 📁 Project Structure

```
autism-screening-system/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── endpoints.py        # Basic API endpoints
│   │   └── enhanced_endpoints.py # Knowledge-guided endpoints
│   ├── models/
│   │   ├── predictor.py        # Main prediction pipeline
│   │   ├── enhanced_predictor.py # With knowledge guidance
│   │   ├── model_loader.py     # Model loading and management
│   │   ├── stgcn.py           # ST-GCN architecture
│   │   └── graph_utils.py     # Graph construction utilities
│   ├── services/
│   │   ├── video_processor.py  # Video processing and feature extraction
│   │   ├── optical_flow.py    # Optical flow computation
│   │   └── knowledge_guide.py # Clinical domain mapping
│   ├── core/
│   │   ├── config.py          # Application configuration
│   │   ├── models.py          # Pydantic models for API
│   │   └── enhanced_models.py # Enhanced response models
│   └── static/
│       └── index.html         # Web interface
├── model_weights/             # Trained model files
├── tests/                     # Test suite
├── requirements.txt           # Python dependencies
├── Dockerfile                # Container configuration
└── docker-compose.yml        # Multi-container setup
```

## 🔌 API Documentation

### Basic Screening Endpoint
```bash
POST /api/v1/screen
Content-Type: multipart/form-data

Body: video file
```

**Response:**
```json
{
  "prediction": {
    "final_score": 6.42,
    "severity": "High Risk",
    "confidence": 0.82
  },
  "component_analysis": {
    "optical_flow": 5.23,
    "skeleton_2d": 6.87,
    "skeleton_3d": 4.56
  },
  "interpretation": {
    "severity": "High Risk",
    "description": "Moderate autism symptoms",
    "recommendation": "Recommend comprehensive diagnostic assessment"
  }
}
```

### Enhanced Screening with Explanations
```bash
POST /api/v1/screen-with-explanation
```

**Enhanced response includes:**
```json
{
  "knowledge_guided_explanation": {
    "base_explanation": "Score 6.42 — driven mainly by SGCN, ST-GCN models...",
    "risk_level": "High Risk",
    "dominant_models": ["SGCN", "ST-GCN"],
    "domains": [
      {
        "domain": "Social Communication & Interaction",
        "description": "Difficulties in social-emotional reciprocity",
        "clinical_reference": "ADOS-2: Social Affect domain"
      }
    ],
    "clinical_recommendations": [
      "Recommend comprehensive diagnostic assessment",
      "Evaluate restricted and repetitive behaviors"
    ]
  }
}
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/test_models/ -v
pytest tests/test_api/ -v
pytest tests/test_services/ -v
```

## 🐳 Docker Deployment

### Using Docker Compose
```bash
docker-compose up -d
```

### Manual Docker Build
```bash
docker build -t autism-screening .
docker run -p 8000:8000 autism-screening
```

## 🏥 Clinical Interpretation

### ADOS Score Ranges
- **1.0-3.0**: Low Risk - Minimal evidence of autism symptoms
- **3.1-5.0**: Mild Risk - Mild autism symptoms present  
- **5.1-7.0**: High Risk - Moderate autism symptoms
- **7.1-10.0**: Very High Risk - Significant autism symptoms

### Clinical Domains
- **Social Communication & Interaction**: TSN, SGCN models
- **Early Social Attention**: SGCN model
- **Object & Repetitive Behaviors**: ST-GCN model
- **Motor Development**: ST-GCN model

## ⚙️ Configuration

Environment variables (`.env` file):
```env
# Model paths
TSN_MODEL_PATH=model_weights/tsn_optical_flow.pth
SGCN_MODEL_PATH=model_weights/sgcn_2d.pth
STGCN_MODEL_PATH=model_weights/stgcn_3d.pth
FUSION_MODEL_PATH=model_weights/fusion.pkl

# Server settings
DEVICE=cuda
HOST=0.0.0.0
PORT=8000

# Video processing
MAX_VIDEO_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=.mp4,.avi,.mov,.mkv
```

## 📊 Performance

- **Inference Time**: 2-5 seconds per video (GPU)
- **Accuracy**: R² ~0.65-0.75 on ADOS scores
- **Model Size**: ~50MB total
- **Supported Resolutions**: 224×224 to 1920×1080

## 🎓 Training Data

This system was trained on the **MMASD** (Multimodal Autism Screening Dataset):
- 32 children with ASD
- 1,315 video samples
- 100+ hours of therapy sessions
- ADOS comparison scores (1-10)

## ⚠️ Important Notes

- **Screening Tool**: This is an assistive screening tool, not a diagnostic system
- **Clinical Oversight**: Always consult healthcare professionals for diagnosis
- **Data Privacy**: No video data is stored permanently
- **Resource Requirements**: GPU recommended for real-time performance



## 🙏 Acknowledgments

- MMASD dataset providers
- Clinical advisors for domain expertise
- Open-source computer vision libraries
- Research collaborators

## 🆘 Support

For technical issues:
- Create an issue on GitHub
- Check existing documentation
- Review test cases for usage examples

For clinical questions:
- Consult with licensed healthcare providers
- Refer to ADOS-2 assessment manuals
- Review autism screening guidelines
```

