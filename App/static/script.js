// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const scoreValue = document.getElementById('scoreValue');
const riskLevel = document.getElementById('riskLevel');
const confidence = document.getElementById('confidence');
const modelContributions = document.getElementById('modelContributions');
const domainAnalysis = document.getElementById('domainAnalysis');
const recommendations = document.getElementById('recommendations');
const rawDataContent = document.getElementById('rawDataContent');

let currentFile = null;
let currentResult = null;

// Drag and drop functionality
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Click upload area to open file dialog
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

function handleFileSelect(file) {
    // Validate file type
    const allowedTypes = ['.mp4', '.avi', '.mov', '.mkv'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        alert('Please select a valid video file (MP4, AVI, MOV, MKV)');
        return;
    }

    // Validate file size (100MB)
    if (file.size > 100 * 1024 * 1024) {
        alert('File size must be less than 100MB');
        return;
    }

    currentFile = file;
    
    // Display file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'block';
    analyzeBtn.disabled = false;

    // Create preview if possible
    createVideoPreview(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function createVideoPreview(file) {
    const url = URL.createObjectURL(file);
    // Could add video preview element here
}

function clearFile() {
    currentFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    analyzeBtn.disabled = true;
}

async function analyzeVideo() {
    if (!currentFile) return;

    // Show loading, hide results
    loadingSpinner.style.display = 'block';
    resultSection.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch('/api/v1/screen-with-explanation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        alert('Analysis failed: ' + error.message);
    } finally {
        loadingSpinner.style.display = 'none';
        resultSection.style.display = 'block';
        analyzeBtn.disabled = false;
    }
}

function displayResults(result) {
    // Store result for download
    currentResult = result;
    
    // Display main score
    const finalScore = result.prediction.final_score;
    scoreValue.textContent = finalScore.toFixed(2);
    
    // Set risk level and color
    const riskClass = getRiskClass(finalScore);
    scoreValue.className = `score-display ${riskClass}`;
    riskLevel.innerHTML = `<span class="badge bg-${getRiskBadgeColor(finalScore)}">${result.prediction.severity}</span>`;
    
    // Display confidence
    confidence.textContent = `Confidence: ${(result.prediction.confidence * 100).toFixed(1)}%`;

    // Display model contributions
    displayModelContributions(result.component_analysis);

    // Display domain analysis
    if (result.knowledge_guided_explanation) {
        displayDomainAnalysis(result.knowledge_guided_explanation);
        displayRecommendations(result.knowledge_guided_explanation);
    }

    // Display raw data
    rawDataContent.textContent = JSON.stringify(result, null, 2);
    
    // Show download button
    document.getElementById('downloadBtn').style.display = 'block';
}

function getRiskClass(score) {
    if (score <= 3) return 'risk-low';
    if (score <= 5) return 'risk-medium';
    if (score <= 7) return 'risk-high';
    return 'risk-very-high';
}

function getRiskBadgeColor(score) {
    if (score <= 3) return 'success';
    if (score <= 5) return 'warning';
    if (score <= 7) return 'danger';
    return 'dark';
}

function displayModelContributions(components) {
    const models = [
        { name: 'Optical Flow', key: 'optical_flow', color: 'primary' },
        { name: '2D Skeleton', key: 'skeleton_2d', color: 'success' },
        { name: '3D Skeleton', key: 'skeleton_3d', color: 'info' }
    ];

    let html = '';
    models.forEach(model => {
        const score = components[model.key];
        const percentage = (score / 10 * 100).toFixed(1);
        html += `
            <div class="mb-3">
                <div class="d-flex justify-content-between">
                    <span>${model.name}</span>
                    <span class="fw-bold">${score.toFixed(2)}</span>
                </div>
                <div class="progress">
                    <div class="progress-bar bg-${model.color}" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });
    modelContributions.innerHTML = html;
}

function displayDomainAnalysis(explanation) {
    let html = '';
    explanation.domains.forEach(domain => {
        html += `
            <div class="card domain-card">
                <div class="card-body">
                    <h6 class="card-title">${domain.domain}</h6>
                    <p class="card-text small text-muted">${domain.description}</p>
                    <div class="small">
                        <strong>Clinical Reference:</strong> ${domain.clinical_reference}
                    </div>
                </div>
            </div>
        `;
    });
    domainAnalysis.innerHTML = html;
}

function displayRecommendations(explanation) {
    let html = '';
    explanation.clinical_recommendations.forEach(rec => {
        html += `
            <div class="card recommendation-card mb-2">
                <div class="card-body py-2">
                    <i class="fas fa-check-circle text-success me-2"></i>
                    ${rec}
                </div>
            </div>
        `;
    });
    recommendations.innerHTML = html;
}

async function downloadReport() {
    if (!currentResult) return;
    
    try {
        // Show loading
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.disabled = true;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating PDF...';
        
        // Send result to PDF generation endpoint
        const response = await fetch('/api/v1/download-report-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentResult)
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate PDF');
        }
        
        // Get PDF blob
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Download
        const link = document.createElement('a');
        link.href = url;
        link.download = `autism_screening_report_${Date.now()}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        // Reset button
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<i class="fas fa-download me-2"></i>Download Full Report';
        
    } catch (error) {
        console.error('Download error:', error);
        alert('Failed to download PDF: ' + error.message);
        
        // Reset button
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<i class="fas fa-download me-2"></i>Download Full Report';
    }
}

// Initialize tooltips
const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});