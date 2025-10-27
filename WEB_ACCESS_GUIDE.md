# How to Access the Web Interface After Deployment

## Access Points

### 1. **Web Application Interface** (for end users)
```
http://YOUR_DOMAIN:8000/app
```
This serves the full HTML interface with:
- Video upload functionality
- Drag-and-drop support
- Real-time analysis
- Interactive results display
- Download report button

### 2. **Interactive API Documentation** (for developers)
```
http://YOUR_DOMAIN:8000/docs
```
Swagger UI for testing API endpoints directly

### 3. **API Root** (JSON response)
```
http://YOUR_DOMAIN:8000/
```
API information in JSON format

### 4. **Health Check**
```
http://YOUR_DOMAIN:8000/health
```
Verify system status and model loading

---

## For Crane Cloud Deployment

### Expected URLs After Deployment:

1. **Web Interface:** 
   ```
   https://your-app-name.cranecodes.com/app
   ```

2. **API Documentation:**
   ```
   https://your-app-name.cranecodes.com/docs
   ```

3. **API Health:**
   ```
   https://your-app-name.cranecodes.com/health
   ```

---

## How the Web Interface Works

### 1. **Video Upload**
- Users can drag & drop video files or click to browse
- Supported formats: MP4, AVI, MOV, MKV
- Maximum file size: 100MB

### 2. **Video Submission**
- When "Analyze Video" button is clicked:
  - Video is sent to `/api/v1/screen-with-explanation`
  - Uses POST request with FormData
  - Shows loading spinner during processing
  - Displays results when complete

### 3. **Results Display**
- Final ADOS score (1-10)
- Risk level (Low/Medium/High/Very High)
- Confidence percentage
- Model contributions breakdown
- Domain analysis
- Clinical recommendations

### 4. **Report Download** ✅ (NEW FEATURE)
- **Download Full Report** button appears after analysis
- Downloads JSON file with:
  - Score and severity
  - Component scores
  - Domain analysis
  - Clinical recommendations
  - Processing metadata
  - Timestamp
- File name: `autism_screening_report_TIMESTAMP.json`

---

## Features Implemented

✅ **Video Upload:** Connected to API via `/api/v1/screen-with-explanation`
✅ **Report Download:** Downloads comprehensive JSON report
✅ **Static File Serving:** Mounted at `/static`
✅ **Web Interface:** Accessible at `/app`
✅ **Real-time Updates:** Shows loading and results
✅ **Responsive Design:** Works on mobile and desktop

---

## Testing Locally

1. Start the server:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

2. Access web interface:
   ```
   http://localhost:8000/app
   ```

3. Test upload:
   - Click or drag video file
   - Click "Analyze Video"
   - Wait for results
   - Click "Download Full Report"

---

## API Endpoints for the Web Interface

The HTML makes these API calls:

- **POST `/api/v1/screen-with-explanation`**
  - Uploads video file
  - Returns detailed analysis with explanations
  - Used by the web interface

- **GET `/static/index.html`**
  - Serves the main web interface
  - Accessible at `/app` route

- **GET `/health`**
  - Health check (can be called by web app)

---

## Important Notes

1. **Video Processing:**
   - Videos are processed server-side
   - Temporary files are cleaned up automatically
   - Processing may take 30-60 seconds

2. **Model Requirements:**
   - Ensure model files are in `model_weights/` directory
   - All 4 models must be loaded for full functionality

3. **Storage:**
   - Videos are temporarily stored during processing
   - Downloads are client-side only (JSON files)

4. **Browser Compatibility:**
   - Modern browsers (Chrome, Firefox, Edge, Safari)
   - Requires JavaScript enabled

---

## Deployment Checklist

- [x] Static files mounted (`/static`)
- [x] Web interface route added (`/app`)
- [x] Video upload connected to API
- [x] Report download functionality added
- [x] File validation (type & size)
- [x] Loading states implemented
- [x] Error handling in place

Your application is now **fully accessible via web browser**!

