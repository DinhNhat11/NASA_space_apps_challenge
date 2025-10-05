
    // Generate animated stars
    function createStars() {
        const container = document.getElementById('starsContainer');
        const starCount = 100;
        
        for (let i = 0; i < starCount; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.animationDelay = Math.random() * 3 + 's';
            container.appendChild(star);
        }
    }

    // Navigation
    // Navigation
function showSection(sectionId) {
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
    
    document.querySelectorAll('.sidebar-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    if (event && event.target) {
        event.target.closest('.sidebar-btn')?.classList.add('active');
    }
    
    // Load data for specific sections
    if (sectionId === 'explore' && currentDatasetId) {
        loadExploreData();
    }
}

    // Page navigation
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            
            document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            if (page === 'dashboard') showSection('dashboard');
            else if (page === 'data') showSection('import');
            else if (page === 'models') showSection('predict');
        });
    });

    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());

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
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        alert(`File "${file.name}" uploaded successfully! (Demo mode)`);
    }

    // Model selection
    function selectModel(modelName) {
        alert(`${modelName.toUpperCase()} model selected! (Demo mode)`);
    }

    // Training simulation
    // function startTraining() {
    //     const progressCard = document.getElementById('trainingProgress');
    //     const progressFill = document.getElementById('progressFill');
    //     const progressText = document.getElementById('progressText');
    // }


    // ==================== ADDITIONAL CODE - ADD AFTER EXISTING CODE ====================

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentDatasetId = null;
let exploreData = null;
let chartInstances = {};

let trainingPollInterval = null;
let currentJobId = null;

// Override the existing handleFile function
function handleFile(file) {
    uploadFileToServer(file);
}

async function uploadFileToServer(file) {
    const uploadArea = document.getElementById('uploadArea');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading
        uploadArea.innerHTML = `
            <div style="font-size: 3rem; margin-bottom: 1rem;">⏳</div>
            <h3>Uploading...</h3>
            <p style="color: var(--text-secondary);">Processing ${file.name}</p>
        `;
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const result = await response.json();
        currentDatasetId = result.dataset_id;
        
        // Reset upload area with success message
        uploadArea.innerHTML = `
            <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
            <h3>File Uploaded Successfully!</h3>
            <p style="color: var(--text-secondary); margin: 1rem 0;">
                <strong>${result.info.filename}</strong><br>
                ${result.info.rows} rows × ${result.info.columns} columns
            </p>
            <div style="margin-top: 1rem;">
                <button class="btn btn-primary" onclick="showSection('explore'); loadExploreData();">
                    Explore Data
                </button>
                <button class="btn btn-secondary" onclick="resetUploadArea()">
                    Upload Another
                </button>
            </div>
        `;
        
        // Update dashboard stats
        updateDashboardStats();
        
    } catch (error) {
        uploadArea.innerHTML = `
            <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
            <h3>Upload Failed</h3>
            <p style="color: var(--error); margin: 1rem 0;">${error.message}</p>
            <button class="btn btn-primary" onclick="resetUploadArea()">Try Again</button>
        `;
    }
}

function resetUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
        <h3>Drag & Drop Your Dataset</h3>
        <p style="color: var(--text-secondary); margin: 1rem 0;">
            or click to browse files
        </p>
        <p style="color: var(--text-secondary); font-size: 0.9rem;">
            Supported formats: CSV, XLSX, JSON
        </p>
    `;
}

// ==================== DATA EXPLORATION ====================

async function loadExploreData() {
    if (!currentDatasetId) {
        alert('Please upload a dataset first');
        showSection('import');
        return;
    }
    
    const exploreSection = document.getElementById('explore');
    exploreSection.innerHTML = `
        <h1>Data Exploration</h1>
        <div class="card">
            <h2 class="card-title">Loading...</h2>
            <p style="color: var(--text-secondary);">Analyzing your dataset...</p>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE_URL}/explore/${currentDatasetId}`);
        if (!response.ok) throw new Error('Failed to load data');
        
        exploreData = await response.json();
        renderExploreData(exploreData);
        
    } catch (error) {
        exploreSection.innerHTML = `
            <h1>Data Exploration</h1>
            <div class="card">
                <h2 class="card-title">Error</h2>
                <p style="color: var(--error);">${error.message}</p>
                <button class="btn btn-primary" onclick="showSection('import')">Go Back</button>
            </div>
        `;
    }
}

function renderExploreData(data) {
    const exploreSection = document.getElementById('explore');
    const stats = data.statistics;
    
    let html = '<h1>Data Exploration</h1>';
    
    // Dataset Overview
    html += `
        <div class="card">
            <h2 class="card-title">Dataset Overview</h2>
            <div class="stats-grid" style="margin-top: 1rem;">
                <div class="stat-card">
                    <div class="stat-value">${stats.summary.total_rows}</div>
                    <div class="stat-label">Total Rows</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.summary.total_columns}</div>
                    <div class="stat-label">Total Columns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.summary.numeric_columns}</div>
                    <div class="stat-label">Numeric Columns</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.summary.missing_values}</div>
                    <div class="stat-label">Missing Values</div>
                </div>
            </div>
        </div>
    `;
    
    // Column Information
    html += `
        <div class="card">
            <h2 class="card-title">Column Information</h2>
            <div style="max-height: 300px; overflow-y: auto;">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Data Type</th>
                            <th>Missing Values</th>
                        </tr>
                    </thead>
                    <tbody>
    `;
    
    for (const [col, dtype] of Object.entries(stats.dtypes)) {
        const missing = stats.missing_by_column[col] || 0;
        html += `
            <tr>
                <td><strong>${col}</strong></td>
                <td>${dtype}</td>
                <td>${missing}</td>
            </tr>
        `;
    }
    
    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    // Statistical Summary
    html += `
        <div class="card">
            <h2 class="card-title">Statistical Summary</h2>
            <div style="max-height: 400px; overflow-y: auto;">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Std Dev</th>
                            <th>Min</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
    `;
    
    for (const [col, colStats] of Object.entries(stats.numeric_stats)) {
        html += `
            <tr>
                <td><strong>${col}</strong></td>
                <td>${colStats.mean !== null ? colStats.mean.toFixed(3) : 'N/A'}</td>
                <td>${colStats.median !== null ? colStats.median.toFixed(3) : 'N/A'}</td>
                <td>${colStats.std !== null ? colStats.std.toFixed(3) : 'N/A'}</td>
                <td>${colStats.min !== null ? colStats.min.toFixed(3) : 'N/A'}</td>
                <td>${colStats.max !== null ? colStats.max.toFixed(3) : 'N/A'}</td>
            </tr>
        `;
    }
    
    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    // Feature Distributions
    html += `
        <div class="card">
            <h2 class="card-title">Feature Distributions</h2>
            <div id="distributionCharts"></div>
        </div>
    `;
    
    // Sample Data
    html += `
        <div class="card">
            <h2 class="card-title">Sample Data (First 10 Rows)</h2>
            <div style="overflow-x: auto; max-height: 400px; overflow-y: auto;">
                ${generateSampleDataTable(data.sample_data)}
            </div>
        </div>
    `;
    
    // Action buttons
    html += `
        <div class="card">
            <h2 class="card-title">Next Steps</h2>
            <button class="btn btn-primary" onclick="showSection('train')">
                Train a Model
            </button>
            <button class="btn btn-secondary" onclick="showSection('import')">
                Upload New Dataset
            </button>
        </div>
    `;
    
    exploreSection.innerHTML = html;
    
    // Render distribution charts after DOM is ready
    setTimeout(() => renderDistributionCharts(data.distributions), 100);
}

function generateSampleDataTable(sampleData) {
    if (!sampleData || sampleData.length === 0) {
        return '<p style="color: var(--text-secondary);">No sample data available</p>';
    }
    
    const columns = Object.keys(sampleData[0]);
    
    let html = '<table class="results-table"><thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    sampleData.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
            html += `<td>${displayValue}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    return html;
}

function renderDistributionCharts(distributions) {
    const container = document.getElementById('distributionCharts');
    if (!container || !distributions) return;
    
    // Clear existing charts
    Object.values(chartInstances).forEach(chart => chart.destroy());
    chartInstances = {};
    
    const columns = Object.keys(distributions);
    
    if (columns.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No numeric columns available for distribution charts</p>';
        return;
    }
    
    let html = '';
    columns.forEach((col, index) => {
        html += `
            <div style="margin-bottom: 2rem;">
                <h3 style="color: var(--accent-secondary); margin-bottom: 1rem;">${col}</h3>
                <canvas id="chart-${index}" style="max-height: 300px;"></canvas>
            </div>
        `;
    });
    
    container.innerHTML = html;
    
    // Create charts
    columns.forEach((col, index) => {
        const dist = distributions[col];
        const ctx = document.getElementById(`chart-${index}`);
        
        if (ctx) {
            // Create bin labels (midpoints)
            const labels = dist.bins.slice(0, -1).map((bin, i) => {
                const midpoint = (bin + dist.bins[i + 1]) / 2;
                return midpoint.toFixed(2);
            });
            
            chartInstances[`chart-${index}`] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Frequency',
                        data: dist.counts,
                        backgroundColor: 'rgba(99, 102, 241, 0.6)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(21, 25, 55, 0.9)',
                            titleColor: '#e2e8f0',
                            bodyColor: '#e2e8f0',
                            borderColor: 'rgba(99, 102, 241, 0.5)',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(148, 163, 184, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(148, 163, 184, 0.1)'
                            },
                            ticks: {
                                color: '#94a3b8',
                                maxRotation: 45,
                                minRotation: 45
                            }
                        }
                    }
                }
            });
        }
    });
}

// ==================== HELPER FUNCTIONS ====================

async function updateDashboardStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/datasets`);
        if (response.ok) {
            const data = await response.json();
            const statCards = document.querySelectorAll('.stat-card .stat-value');
            if (statCards[1]) {
                statCards[1].textContent = data.datasets.length;
            }
        }
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    createStars();
});

// Load data for specific sections
if (sectionId === 'explore' && currentDatasetId) {
    loadExploreData();
}


// ==================== MODEL TRAINING ====================

// Global variable for polling
// let trainingPollInterval = null;
// let currentJobId = null;

// Override the existing startTraining function
function startTraining() {
    if (!currentDatasetId) {
        alert('Please upload a dataset first');
        showSection('import');
        return;
    }
    
    if (!exploreData) {
        alert('Please explore the dataset first to see available columns');
        showSection('explore');
        loadExploreData();
        return;
    }
    
    startModelTraining();
}

async function startModelTraining() {
    const targetColumn = document.getElementById('targetColumn').value;
    const algorithm = document.getElementById('algorithm').value;
    const testSplit = parseInt(document.getElementById('testSplit').value) / 100;
    const maxIterations = parseInt(document.getElementById('maxIterations').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    
    // Get selected feature columns
    const featureCheckboxes = document.querySelectorAll('#train input[type="checkbox"]:checked');
    const featureColumns = Array.from(featureCheckboxes).map(cb => cb.value);
    
    const config = {
        dataset_id: currentDatasetId,
        algorithm: algorithm,
        test_size: testSplit,
        max_iterations: maxIterations,
        learning_rate: learningRate,
        target_column: targetColumn,
        feature_columns: featureColumns.length > 0 ? featureColumns : null
    };
    
    try {
        // Show progress card
        const progressCard = document.getElementById('trainingProgress');
        progressCard.style.display = 'block';
        
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        progressFill.style.width = '0%';
        progressText.textContent = 'Starting training...';
        
        // Start training
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Training failed');
        }
        
        const result = await response.json();
        if (!result.job_id) {
            throw new Error("Server did not return a valid job ID");
        }

        currentJobId = result.job_id;

        // Start polling only after job_id is set
        pollTrainingStatus();

        
    } catch (error) {
        alert('Training failed: ' + error.message);
        document.getElementById('trainingProgress').style.display = 'none';
    }
}

async function pollTrainingStatus() {
    if (!currentJobId) return;
    
    // Clear existing interval
    if (trainingPollInterval) {
        clearInterval(trainingPollInterval);
    }
    
    trainingPollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/train/status/${currentJobId}`);
            if (!response.ok) throw new Error('Failed to get status');
            
            const status = await response.json();
            
            // Update progress bar
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            progressFill.style.width = status.progress + '%';
            progressText.textContent = status.message;
            
            // Check if complete
            if (status.status === 'completed') {
                clearInterval(trainingPollInterval);
                displayTrainingResults(status);
            } else if (status.status === 'failed') {
                clearInterval(trainingPollInterval);
                progressText.textContent = '❌ ' + status.message;
                alert('Training failed: ' + status.error);
            }
            
        } catch (error) {
            clearInterval(trainingPollInterval);
            console.error('Error polling status:', error);
        }
    }, 1000); // Poll every second
}

function displayTrainingResults(status) {
    const results = status.results;
    
    // Create results section
    const trainSection = document.getElementById('train');
    
    let resultsHtml = `
        <div class="card" style="margin-top: 2rem; border-color: var(--success);">
            <h2 class="card-title">✅ Training Results</h2>
            
            <div class="stats-grid" style="margin: 1.5rem 0;">
                <div class="stat-card">
                    <div class="stat-value" style="color: var(--success);">${(results.accuracy * 100).toFixed(2)}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(results.precision * 100).toFixed(2)}%</div>
                    <div class="stat-label">Precision</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(results.recall * 100).toFixed(2)}%</div>
                    <div class="stat-label">Recall</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(results.f1_score * 100).toFixed(2)}%</div>
                    <div class="stat-label">F1 Score</div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <h3 style="color: var(--accent-secondary); margin-bottom: 1rem;">Training Details</h3>
                <p style="color: var(--text-secondary); line-height: 1.8;">
                    <strong>Model ID:</strong> ${status.model_id}<br>
                    <strong>Algorithm:</strong> ${status.config.algorithm.replace(/_/g, ' ').toUpperCase()}<br>
                    <strong>Training Samples:</strong> ${results.train_size}<br>
                    <strong>Test Samples:</strong> ${results.test_size}<br>
                    <strong>Features Used:</strong> ${results.features_used.length}<br>
                    <strong>Target Column:</strong> ${status.config.target_column}
                </p>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <h3 style="color: var(--accent-secondary); margin-bottom: 1rem;">Confusion Matrix</h3>
                <div id="confusionMatrix" style="overflow-x: auto;"></div>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <button class="btn btn-primary" onclick="showSection('predict')">
                    Use This Model for Predictions
                </button>
                <button class="btn btn-secondary" onclick="location.reload()">
                    Train Another Model
                </button>
            </div>
        </div>
    `;
    
    trainSection.innerHTML += resultsHtml;
    
    // Render confusion matrix
    renderConfusionMatrix(results.confusion_matrix);
}

function renderConfusionMatrix(matrix) {
    const container = document.getElementById('confusionMatrix');
    if (!container || !matrix) return;
    
    const maxValue = Math.max(...matrix.flat());
    
    let html = '<table style="margin: 0 auto; border-collapse: collapse; min-width: 300px;">';
    
    matrix.forEach((row, i) => {
        html += '<tr>';
        row.forEach((value, j) => {
            const intensity = maxValue > 0 ? value / maxValue : 0;
            const bgColor = `rgba(99, 102, 241, ${intensity * 0.7 + 0.2})`;
            html += `
                <td style="padding: 1.5rem 2rem; border: 1px solid var(--border); 
                           background: ${bgColor}; text-align: center; font-weight: bold;
                           font-size: 1.2rem;">
                    ${value}
                </td>
            `;
        });
        html += '</tr>';
    });
    
    html += '</table>';
    html += '<p style="color: var(--text-secondary); text-align: center; margin-top: 1rem; font-size: 0.9rem;">Rows: Actual | Columns: Predicted</p>';
    container.innerHTML = html;
}

// Load training interface with dataset columns
// Load training interface with dataset columns
function loadTrainingInterface() {
    if (!currentDatasetId || !exploreData) {
        alert('Please upload and explore a dataset first');
        showSection('import');
        return;
    }
    
    const trainSection = document.getElementById('train');
    const columns = exploreData.column_info.all_columns;
    const numericColumns = exploreData.column_info.numeric_columns;
    
    // Build target column options
    let targetOptions = '';
    columns.forEach(col => {
        targetOptions += `<option value="${col}">${col}</option>`;
    });
    
    // Build feature column checkboxes
    let featureCheckboxes = '';
    if (numericColumns.length > 0) {
        numericColumns.forEach(col => {
            featureCheckboxes += `
                <label style="display: block; margin: 0.5rem 0; color: var(--text-secondary); cursor: pointer;">
                    <input type="checkbox" value="${col}" checked style="margin-right: 0.5rem;">
                    ${col}
                </label>
            `;
        });
    } else {
        featureCheckboxes = '<p style="color: var(--warning);">No numeric columns found. Please upload a dataset with numeric features.</p>';
    }
    
    // REPLACE THE ENTIRE TRAIN SECTION CONTENT (not just the first card)
    trainSection.innerHTML = `
        <h1>Train New Model</h1>
        
        <div class="card">
            <h2 class="card-title">Model Configuration</h2>
            
            <div class="form-group">
                <label class="form-label">Target Column (Label to Predict)</label>
                <select class="form-select" id="targetColumn">
                    ${targetOptions}
                </select>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">
                    Select the column you want to predict
                </p>
            </div>
            
            <div class="form-group">
                <label class="form-label">Algorithm</label>
                <select class="form-select" id="algorithm">
                    <option value="random_forest">Random Forest</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="neural_network">Neural Network</option>
                    <option value="svm">Support Vector Machine</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="form-label">Test Split (%) - <span id="testSplitValue">20</span>%</label>
                <input type="range" class="form-input" id="testSplit" value="20" min="10" max="40" 
                       oninput="document.getElementById('testSplitValue').textContent = this.value">
            </div>
            
            <div class="form-group">
                <label class="form-label">Max Iterations / Estimators</label>
                <input type="number" class="form-input" id="maxIterations" value="100" min="10" max="1000">
            </div>
            
            <div class="form-group">
                <label class="form-label">Learning Rate</label>
                <input type="number" class="form-input" id="learningRate" value="0.01" step="0.001" min="0.001" max="1">
            </div>
            
            <div class="form-group">
                <label class="form-label">Feature Columns</label>
                <p style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.5rem;">
                    Select features to use for training (numeric columns only)
                </p>
                <div style="max-height: 200px; overflow-y: auto; padding: 1rem; background: var(--bg-tertiary); border-radius: 8px; margin-top: 0.5rem;">
                    ${featureCheckboxes}
                </div>
            </div>
            
            <button class="btn btn-primary" onclick="startTraining()">
                🚀 Start Training
            </button>
            <button class="btn btn-secondary" onclick="showSection('explore')">
                Back to Exploration
            </button>
        </div>
        
        <div class="card" id="trainingProgress" style="display: none;">
            <h2 class="card-title">Training Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
            </div>
            <p style="color: var(--text-secondary); margin-top: 1rem;" id="progressText">
                Initializing...
            </p>
        </div>
    `;
}

// Update showSection to load training interface when navigating to train
// Update showSection to load training interface when navigating to train
const originalShowSection = showSection;
showSection = async function(sectionId) {
    originalShowSection(sectionId);
    
    // If navigating to train section
    if (sectionId === 'train') {
        if (!currentDatasetId) {
            alert('Please upload a dataset first');
            showSection('import');
            return;
        }
        
        // Load explore data if not already loaded
        if (!exploreData) {
            try {
                const response = await fetch(`${API_BASE_URL}/explore/${currentDatasetId}`);
                if (!response.ok) throw new Error('Failed to load data');
                exploreData = await response.json();
            } catch (error) {
                alert('Failed to load dataset information: ' + error.message);
                showSection('import');
                return;
            }
        }
        
        // Now load the training interface
        loadTrainingInterface();
    }
};