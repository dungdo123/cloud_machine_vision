/**
 * MVAS - Machine Vision Application Standard
 * Web UI Application
 */

// ============================================================================
// Global State
// ============================================================================

const state = {
    apps: [],
    cameras: [],
    results: [],
    currentPage: 'dashboard',
    inferenceMode: 'image',
    selectedFile: null,
    uploadFile: null,
    videoFile: null,
    ws: null,
    streaming: false,
    streamFrameCount: 0,
    streamStartTime: null,
    // Video processing state
    videoProcessing: false,
    videoStats: {
        frames: 0,
        pass: 0,
        fail: 0,
        review: 0,
        totalTime: 0,
        decisions: [],
    },
    // Live stream state
    liveStats: {
        frames: 0,
        pass: 0,
        fail: 0,
        totalTime: 0,
    },
};

// ============================================================================
// API Client
// ============================================================================

const API = {
    baseUrl: '',
    
    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    },
    
    async post(endpoint, data, isFormData = false) {
        const options = {
            method: 'POST',
            body: isFormData ? data : JSON.stringify(data),
        };
        if (!isFormData) {
            options.headers = { 'Content-Type': 'application/json' };
        }
        const response = await fetch(`${this.baseUrl}${endpoint}`, options);
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    },
    
    async delete(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, { method: 'DELETE' });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    },
    
    // Apps
    async getApps() {
        return this.get('/api/v1/apps');
    },
    
    async loadApp(appPath) {
        return this.post('/api/v1/apps/load', { app_path: appPath });
    },
    
    async uploadApp(file) {
        const formData = new FormData();
        formData.append('file', file);
        return this.post('/api/v1/apps/upload', formData, true);
    },
    
    async unloadApp(appId) {
        return this.delete(`/api/v1/apps/${appId}`);
    },
    
    async getAppStats(appId) {
        return this.get(`/api/v1/apps/${appId}/stats`);
    },
    
    // Cameras
    async getCameras() {
        return this.get('/api/v1/cameras');
    },
    
    async connectCamera(type, address, name, settings = {}) {
        return this.post('/api/v1/cameras/connect', {
            camera_type: type,
            address: address,
            name: name,
            settings: settings,
        });
    },
    
    async disconnectCamera(cameraId) {
        return this.delete(`/api/v1/cameras/${cameraId}`);
    },
    
    // Inspection
    async inspect(appId, imageFile, visualize = true) {
        const formData = new FormData();
        formData.append('file', imageFile);
        return this.post(`/api/v1/inspect/upload?app_id=${appId}&visualize=${visualize}`, formData, true);
    },
    
    // Status
    async getStatus() {
        return this.get('/api/v1/status');
    },
};

// ============================================================================
// Navigation
// ============================================================================

function navigateTo(page) {
    // Update state
    state.currentPage = page;
    
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
    
    // Show selected page
    document.getElementById(`page-${page}`).classList.remove('hidden');
    
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });
    
    // Page-specific actions
    switch (page) {
        case 'dashboard':
            refreshDashboard();
            break;
        case 'apps':
            refreshApps();
            break;
        case 'inference':
            populateAppSelects();
            populateCameraSelects();
            break;
        case 'cameras':
            refreshCameras();
            break;
        case 'stream':
            populateAppSelects();
            populateCameraSelects();
            break;
    }
}

// ============================================================================
// Dashboard
// ============================================================================

async function refreshDashboard() {
    try {
        const [apps, cameras, status] = await Promise.all([
            API.getApps(),
            API.getCameras(),
            API.getStatus().catch(() => ({})),
        ]);
        
        state.apps = apps;
        state.cameras = cameras;
        
        // Update stats
        document.getElementById('stat-apps').textContent = apps.length;
        document.getElementById('stat-cameras').textContent = cameras.length;
        document.getElementById('stat-inspections').textContent = status.total_inspections || 0;
        document.getElementById('stat-errors').textContent = status.errors_today || 0;
        document.getElementById('apps-count').textContent = apps.length;
        
        // Update apps list
        renderDashboardApps(apps);
    } catch (error) {
        showToast('error', 'Error', error.message);
    }
}

function renderDashboardApps(apps) {
    const container = document.getElementById('dashboard-apps-list');
    
    if (apps.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                </svg>
                <h3>No Applications Loaded</h3>
                <p>Upload a .mvapp file to get started</p>
                <button class="btn btn-primary" onclick="showUploadModal()">Upload App</button>
            </div>
        `;
        return;
    }
    
    container.innerHTML = apps.map(app => createAppCard(app)).join('');
}

function createAppCard(app) {
    const icons = {
        'anomaly_detection': '🔍',
        'classification': '🏷️',
        'segmentation': '🎯',
        'object_detection': '📦',
        'default': '📦'
    };
    const icon = icons[app.model_type] || icons.default;
    
    return `
        <div class="app-card" onclick="selectApp('${app.id}')">
            <div class="app-card-icon">${icon}</div>
            <div class="app-card-body">
                <div class="app-card-title">${app.name}</div>
                <div class="app-card-version">v${app.version} · ${app.model_type || 'Unknown'}</div>
                <div class="app-card-desc">${app.description || 'No description'}</div>
            </div>
            <div class="app-card-footer">
                <div class="app-status">
                    <div class="app-status-dot loaded"></div>
                    Loaded
                </div>
                <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); unloadApp('${app.id}')">
                    Unload
                </button>
            </div>
        </div>
    `;
}

// ============================================================================
// Applications
// ============================================================================

async function refreshApps() {
    try {
        const apps = await API.getApps();
        state.apps = apps;
        document.getElementById('apps-count').textContent = apps.length;
        
        const container = document.getElementById('apps-list');
        if (apps.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    </svg>
                    <h3>No Applications</h3>
                    <p>Upload a .mvapp file to get started</p>
                    <button class="btn btn-primary" onclick="showUploadModal()">Upload App</button>
                </div>
            `;
        } else {
            container.innerHTML = apps.map(app => createAppCard(app)).join('');
        }
    } catch (error) {
        showToast('error', 'Error', error.message);
    }
}

function selectApp(appId) {
    navigateTo('inference');
    document.getElementById('inference-app-select').value = appId;
}

async function unloadApp(appId) {
    if (!confirm(`Unload application ${appId}?`)) return;
    
    try {
        await API.unloadApp(appId);
        showToast('success', 'App Unloaded', `Application ${appId} has been unloaded`);
        refreshApps();
        refreshDashboard();
    } catch (error) {
        showToast('error', 'Error', error.message);
    }
}

// ============================================================================
// Upload
// ============================================================================

function showUploadModal() {
    document.getElementById('upload-modal').classList.add('active');
    state.uploadFile = null;
    document.getElementById('upload-btn').disabled = true;
    document.getElementById('upload-progress').classList.add('hidden');
}

function hideUploadModal() {
    document.getElementById('upload-modal').classList.remove('active');
}

async function uploadApp() {
    if (!state.uploadFile) return;
    
    const progressEl = document.getElementById('upload-progress');
    const progressBar = document.getElementById('upload-progress-bar');
    
    progressEl.classList.remove('hidden');
    progressBar.style.width = '30%';
    
    try {
        const result = await API.uploadApp(state.uploadFile);
        progressBar.style.width = '100%';
        
        showToast('success', 'App Uploaded', `Application "${result.app_info?.name || 'Unknown'}" loaded successfully`);
        hideUploadModal();
        refreshApps();
        refreshDashboard();
    } catch (error) {
        showToast('error', 'Upload Failed', error.message);
        progressEl.classList.add('hidden');
    }
}

// ============================================================================
// Cameras
// ============================================================================

async function refreshCameras() {
    try {
        const cameras = await API.getCameras();
        state.cameras = cameras;
        
        const container = document.getElementById('cameras-list');
        if (cameras.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                    <h3>No Cameras Connected</h3>
                    <p>Add a USB camera, GigE camera, or folder watcher</p>
                    <button class="btn btn-primary" onclick="showCameraModal()">Add Camera</button>
                </div>
            `;
        } else {
            container.innerHTML = cameras.map(cam => `
                <div class="app-card">
                    <div class="app-card-icon">📷</div>
                    <div class="app-card-body">
                        <div class="app-card-title">${cam.name || cam.id}</div>
                        <div class="app-card-version">${cam.type} · ${cam.address}</div>
                    </div>
                    <div class="app-card-footer">
                        <div class="app-status">
                            <div class="app-status-dot ${cam.connected ? 'loaded' : 'error'}"></div>
                            ${cam.connected ? 'Connected' : 'Disconnected'}
                        </div>
                        <button class="btn btn-sm btn-danger" onclick="disconnectCamera('${cam.id}')">
                            Remove
                        </button>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        showToast('error', 'Error', error.message);
    }
}

function showCameraModal() {
    document.getElementById('camera-modal').classList.add('active');
}

function hideCameraModal() {
    document.getElementById('camera-modal').classList.remove('active');
}

async function addCamera() {
    const type = document.getElementById('camera-type').value;
    const name = document.getElementById('camera-name').value || `Camera ${state.cameras.length + 1}`;
    const address = document.getElementById('camera-address').value;
    
    if (!address) {
        showToast('warning', 'Missing Address', 'Please enter a camera address or path');
        return;
    }
    
    try {
        await API.connectCamera(type, address, name);
        showToast('success', 'Camera Added', `Camera "${name}" connected successfully`);
        hideCameraModal();
        refreshCameras();
    } catch (error) {
        showToast('error', 'Connection Failed', error.message);
    }
}

async function disconnectCamera(cameraId) {
    if (!confirm('Disconnect this camera?')) return;
    
    try {
        await API.disconnectCamera(cameraId);
        showToast('success', 'Camera Removed', 'Camera disconnected');
        refreshCameras();
    } catch (error) {
        showToast('error', 'Error', error.message);
    }
}

// ============================================================================
// Inference Mode Switching
// ============================================================================

function switchInferenceMode(mode) {
    state.inferenceMode = mode;
    
    // Update tabs
    document.querySelectorAll('[data-inference-mode]').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.inferenceMode === mode);
    });
    
    // Hide all mode panels
    document.querySelectorAll('.inference-mode').forEach(panel => {
        panel.classList.add('hidden');
    });
    
    // Show selected mode panel
    document.getElementById(`inference-mode-${mode}`).classList.remove('hidden');
    
    // Populate selects for the mode
    populateAppSelects();
    populateCameraSelects();
}

// ============================================================================
// Inference - Selects Population
// ============================================================================

function populateAppSelects() {
    const selects = [
        document.getElementById('inference-app-select'),
        document.getElementById('video-app-select'),
        document.getElementById('live-app-select'),
        document.getElementById('stream-app-select'),
    ];
    
    selects.forEach(select => {
        if (!select) return;
        const currentValue = select.value;
        select.innerHTML = '<option value="">-- Select an app --</option>' +
            state.apps.map(app => `<option value="${app.id}">${app.name}</option>`).join('');
        select.value = currentValue;
    });
}

function populateCameraSelects() {
    const selects = [
        document.getElementById('live-camera-select'),
        document.getElementById('stream-camera-select'),
    ];
    
    selects.forEach(select => {
        if (!select) return;
        const currentValue = select.value;
        select.innerHTML = '<option value="">-- Select a camera --</option>' +
            state.cameras.map(cam => `<option value="${cam.id}">${cam.name || cam.id}</option>`).join('');
        select.value = currentValue;
    });
}

async function runInference() {
    const appId = document.getElementById('inference-app-select').value;
    const file = state.selectedFile;
    
    if (!appId || !file) {
        showToast('warning', 'Missing Input', 'Please select an app and upload an image');
        return;
    }
    
    const btn = document.getElementById('run-inference-btn');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Processing...';
    
    try {
        const result = await API.inspect(appId, file, true);
        displayResult(result);
        addResultToHistory(result, appId);
    } catch (error) {
        showToast('error', 'Inference Failed', error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            Run Inference
        `;
    }
}

function displayResult(result) {
    document.getElementById('results-placeholder').classList.add('hidden');
    document.getElementById('results-display').classList.remove('hidden');
    
    // Decision
    const decisionEl = document.getElementById('result-decision');
    decisionEl.textContent = result.decision.toUpperCase();
    decisionEl.className = `result-decision ${result.decision}`;
    
    // Request ID
    document.getElementById('result-request-id').textContent = `Request ID: ${result.request_id}`;
    
    // Metrics
    const score = result.anomaly_score || 0;
    document.getElementById('result-score').textContent = score.toFixed(3);
    document.getElementById('result-score-bar').style.width = `${score * 100}%`;
    
    const confidence = (result.confidence || 0) * 100;
    document.getElementById('result-confidence').textContent = `${confidence.toFixed(1)}%`;
    document.getElementById('result-confidence-bar').style.width = `${confidence}%`;
    
    document.getElementById('result-inference-time').textContent = `${result.inference_time_ms?.toFixed(1) || '--'} ms`;
    document.getElementById('result-total-time').textContent = `${result.total_time_ms?.toFixed(1) || '--'} ms`;
    
    // Visualization
    if (result.visualization_base64) {
        document.getElementById('result-visualization').src = 
            `data:image/jpeg;base64,${result.visualization_base64}`;
    }
}

function addResultToHistory(result, appId) {
    const app = state.apps.find(a => a.id === appId);
    state.results.unshift({
        ...result,
        app_name: app?.name || appId,
        timestamp: new Date().toISOString(),
    });
    
    // Keep only last 100 results
    if (state.results.length > 100) {
        state.results = state.results.slice(0, 100);
    }
    
    updateResultsTable();
}

function updateResultsTable() {
    const tbody = document.getElementById('results-table-body');
    
    if (state.results.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted" style="padding: 40px;">
                    No results recorded yet
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = state.results.map(r => `
        <tr>
            <td class="font-mono">${r.request_id?.slice(0, 12) || '--'}...</td>
            <td>${r.app_name}</td>
            <td>
                <span class="result-decision ${r.decision}" style="padding: 4px 8px; font-size: 11px;">
                    ${r.decision?.toUpperCase()}
                </span>
            </td>
            <td class="font-mono">${r.anomaly_score?.toFixed(3) || '--'}</td>
            <td>${r.inference_time_ms?.toFixed(1) || '--'} ms</td>
            <td>${new Date(r.timestamp).toLocaleString()}</td>
        </tr>
    `).join('');
}

// ============================================================================
// Video Processing
// ============================================================================

function setupVideoHandlers() {
    const videoDropzone = document.getElementById('video-dropzone');
    const videoInput = document.getElementById('video-file-input');
    
    if (!videoDropzone || !videoInput) return;
    
    videoDropzone.addEventListener('click', () => videoInput.click());
    videoDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoDropzone.classList.add('dragover');
    });
    videoDropzone.addEventListener('dragleave', () => {
        videoDropzone.classList.remove('dragover');
    });
    videoDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            handleVideoFile(file);
        }
    });
    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleVideoFile(file);
        }
    });
}

function handleVideoFile(file) {
    state.videoFile = file;
    
    // Update dropzone text
    document.getElementById('video-dropzone').querySelector('h4').textContent = file.name;
    
    // Show video preview
    const video = document.getElementById('video-preview');
    video.src = URL.createObjectURL(file);
    
    // Get video info when metadata loads
    video.onloadedmetadata = () => {
        document.getElementById('video-info').classList.remove('hidden');
        document.getElementById('video-info-name').textContent = file.name;
        document.getElementById('video-info-duration').textContent = formatDuration(video.duration);
        document.getElementById('video-info-resolution').textContent = `${video.videoWidth}x${video.videoHeight}`;
        
        // Estimate frames based on common frame rates
        const estFrames = Math.round(video.duration * 30);
        document.getElementById('video-info-frames').textContent = `~${estFrames}`;
    };
    
    // Enable process button if app is selected
    const appId = document.getElementById('video-app-select').value;
    document.getElementById('video-process-btn').disabled = !appId;
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function processVideo() {
    const appId = document.getElementById('video-app-select').value;
    if (!appId || !state.videoFile) {
        showToast('warning', 'Missing Input', 'Please select an app and upload a video');
        return;
    }
    
    // Get parameters
    const params = {
        frameSkip: parseInt(document.getElementById('video-param-skip').value) || 1,
        maxFrames: parseInt(document.getElementById('video-param-max').value) || 0,
        fps: parseInt(document.getElementById('video-param-fps').value) || 10,
        visualization: document.getElementById('video-param-visualization').value,
        threshold: parseFloat(document.getElementById('video-param-threshold').value) || 0.5,
    };
    
    // Reset stats
    state.videoStats = { frames: 0, pass: 0, fail: 0, review: 0, totalTime: 0, decisions: [] };
    state.videoProcessing = true;
    
    // Update UI
    document.getElementById('video-process-btn').classList.add('hidden');
    document.getElementById('video-stop-btn').classList.remove('hidden');
    document.getElementById('video-processing-badge').classList.remove('hidden');
    document.getElementById('video-progress-section').classList.remove('hidden');
    
    // Get video element and create canvas for frame extraction
    const video = document.getElementById('video-preview');
    const canvas = document.getElementById('video-canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.classList.remove('hidden');
    video.classList.add('hidden');
    
    // Wait for video to be ready
    await new Promise(resolve => {
        if (video.readyState >= 2) resolve();
        else video.oncanplay = resolve;
    });
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const totalFrames = params.maxFrames > 0 ? params.maxFrames : Math.floor(video.duration * 30 / params.frameSkip);
    const frameInterval = 1000 / params.fps;
    let frameIndex = 0;
    let currentTime = 0;
    const frameStep = params.frameSkip / 30; // Assuming 30fps source
    
    // Process frames
    const startTime = Date.now();
    
    while (state.videoProcessing && currentTime < video.duration) {
        if (params.maxFrames > 0 && frameIndex >= params.maxFrames) break;
        
        // Seek to frame
        video.currentTime = currentTime;
        await new Promise(resolve => video.onseeked = resolve);
        
        // Draw frame to canvas
        ctx.drawImage(video, 0, 0);
        
        // Convert canvas to blob and send for inference
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
        
        try {
            const result = await API.inspect(appId, blob, params.visualization !== 'none');
            
            // Update stats
            state.videoStats.frames++;
            state.videoStats.totalTime += result.inference_time_ms || 0;
            if (result.decision === 'pass') state.videoStats.pass++;
            else if (result.decision === 'fail') state.videoStats.fail++;
            else state.videoStats.review++;
            
            state.videoStats.decisions.push(result.decision);
            
            // Update visualization
            if (result.visualization_base64) {
                const img = new Image();
                img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                img.src = `data:image/jpeg;base64,${result.visualization_base64}`;
            }
            
            // Update progress
            const progress = (frameIndex / totalFrames) * 100;
            document.getElementById('video-progress-bar').style.width = `${progress}%`;
            document.getElementById('video-progress-text').textContent = `${frameIndex + 1} / ${totalFrames} frames`;
            
            // Update stats display
            updateVideoStats();
            updateVideoTimeline();
            
        } catch (error) {
            console.error('Frame processing error:', error);
        }
        
        frameIndex++;
        currentTime += frameStep;
        
        // Frame rate limiting
        await new Promise(resolve => setTimeout(resolve, frameInterval));
    }
    
    // Processing complete
    state.videoProcessing = false;
    document.getElementById('video-process-btn').classList.remove('hidden');
    document.getElementById('video-stop-btn').classList.add('hidden');
    document.getElementById('video-processing-badge').classList.add('hidden');
    
    const totalProcessTime = (Date.now() - startTime) / 1000;
    showToast('success', 'Video Processing Complete', 
        `Processed ${state.videoStats.frames} frames in ${totalProcessTime.toFixed(1)}s`);
}

function stopVideoProcessing() {
    state.videoProcessing = false;
    showToast('info', 'Processing Stopped', 'Video processing was stopped');
}

function updateVideoStats() {
    const stats = state.videoStats;
    document.getElementById('video-stat-frames').textContent = stats.frames;
    
    const passRate = stats.frames > 0 ? ((stats.pass / stats.frames) * 100).toFixed(1) : '--';
    document.getElementById('video-stat-pass').textContent = `${passRate}%`;
    
    const avgTime = stats.frames > 0 ? (stats.totalTime / stats.frames).toFixed(1) : '--';
    document.getElementById('video-stat-time').textContent = `${avgTime} ms`;
    
    const fps = stats.frames > 0 && stats.totalTime > 0 ? 
        (stats.frames / (stats.totalTime / 1000)).toFixed(1) : '--';
    document.getElementById('video-stat-fps').textContent = fps;
}

function updateVideoTimeline() {
    const timeline = document.getElementById('video-timeline');
    const bar = timeline.querySelector('.timeline-bar');
    
    bar.innerHTML = state.videoStats.decisions.map((decision, i) => 
        `<div class="timeline-segment ${decision}" title="Frame ${i + 1}: ${decision}"></div>`
    ).join('');
}

// ============================================================================
// Live Stream (Enhanced)
// ============================================================================

function startLiveStream() {
    const appId = document.getElementById('live-app-select').value;
    const cameraId = document.getElementById('live-camera-select').value;
    
    if (!appId || !cameraId) {
        showToast('warning', 'Missing Selection', 'Please select an app and camera');
        return;
    }
    
    // Get parameters
    const params = {
        fps: parseInt(document.getElementById('live-param-fps').value) || 10,
        frameSkip: parseInt(document.getElementById('live-param-skip').value) || 1,
        visualization: document.getElementById('live-param-visualization').value,
        threshold: parseFloat(document.getElementById('live-param-threshold').value) || 0.5,
        autoSave: document.getElementById('live-param-autosave').value === 'true',
        alert: document.getElementById('live-param-alert').value === 'true',
    };
    
    // Reset stats
    state.liveStats = { frames: 0, pass: 0, fail: 0, totalTime: 0 };
    
    // Create WebSocket connection
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const sessionId = 'live-' + Date.now();
    state.ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream/${sessionId}`);
    
    state.ws.onopen = () => {
        state.ws.send(JSON.stringify({
            type: 'config',
            app_id: appId,
            camera_id: cameraId,
            fps: params.fps,
            visualize: params.visualization !== 'none',
            threshold: params.threshold,
        }));
        
        state.streaming = true;
        state.streamStartTime = Date.now();
        
        document.getElementById('live-start-btn').classList.add('hidden');
        document.getElementById('live-stop-btn').classList.remove('hidden');
        document.getElementById('live-badge').classList.remove('hidden');
        
        showToast('success', 'Live Stream Started', 'Real-time inspection is active');
    };
    
    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'frame') {
            state.liveStats.frames++;
            
            // Update image
            if (data.image_base64) {
                document.getElementById('live-video').src = 
                    `data:image/jpeg;base64,${data.image_base64}`;
            }
            
            // Update metrics
            if (data.result) {
                const decision = data.result.decision?.toUpperCase() || '--';
                const score = data.result.anomaly_score || 0;
                const inferenceTime = data.result.inference_time_ms || 0;
                
                state.liveStats.totalTime += inferenceTime;
                if (data.result.decision === 'pass') state.liveStats.pass++;
                else state.liveStats.fail++;
                
                // Update current stats
                document.getElementById('live-stat-decision').textContent = decision;
                document.getElementById('live-stat-decision').className = `metric-value ${data.result.decision}`;
                document.getElementById('live-stat-score').textContent = score.toFixed(3);
                document.getElementById('live-score-bar').style.width = `${score * 100}%`;
                document.getElementById('live-stat-time').textContent = `${inferenceTime.toFixed(1)} ms`;
                document.getElementById('live-stat-frames').textContent = state.liveStats.frames;
                
                // Update decision badge
                const badge = document.getElementById('live-decision-badge');
                badge.textContent = decision;
                badge.className = `stream-badge ${data.result.decision}`;
                
                // Update session stats
                document.getElementById('live-total-pass').textContent = state.liveStats.pass;
                document.getElementById('live-total-fail').textContent = state.liveStats.fail;
                
                const passRate = state.liveStats.frames > 0 ? 
                    ((state.liveStats.pass / state.liveStats.frames) * 100).toFixed(1) : '--';
                document.getElementById('live-pass-rate').textContent = `${passRate}%`;
                
                const avgTime = state.liveStats.frames > 0 ?
                    (state.liveStats.totalTime / state.liveStats.frames).toFixed(1) : '--';
                document.getElementById('live-avg-time').textContent = `${avgTime} ms`;
                
                // Alert on fail
                if (params.alert && data.result.decision === 'fail') {
                    playAlertSound();
                    document.getElementById('inference-mode-live').classList.add('alert-flash');
                    setTimeout(() => {
                        document.getElementById('inference-mode-live').classList.remove('alert-flash');
                    }, 1500);
                }
            }
            
            // Update FPS
            const elapsed = (Date.now() - state.streamStartTime) / 1000;
            const fps = state.liveStats.frames / elapsed;
            document.getElementById('live-fps-badge').textContent = `${fps.toFixed(1)} FPS`;
        }
    };
    
    state.ws.onerror = (error) => {
        showToast('error', 'Stream Error', 'WebSocket connection error');
        stopLiveStream();
    };
    
    state.ws.onclose = () => {
        if (state.streaming) {
            showToast('info', 'Stream Ended', 'Connection closed');
            stopLiveStream();
        }
    };
}

function stopLiveStream() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
    
    state.streaming = false;
    
    document.getElementById('live-start-btn').classList.remove('hidden');
    document.getElementById('live-stop-btn').classList.add('hidden');
    document.getElementById('live-badge').classList.add('hidden');
}

function playAlertSound() {
    // Create a simple beep sound using Web Audio API
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.3;
        
        oscillator.start();
        setTimeout(() => oscillator.stop(), 200);
    } catch (e) {
        console.warn('Could not play alert sound:', e);
    }
}

// ============================================================================
// Legacy Live Stream (for backwards compatibility on stream page)
// ============================================================================

function startStream() {
    const appId = document.getElementById('stream-app-select').value;
    const cameraId = document.getElementById('stream-camera-select').value;
    const fps = parseInt(document.getElementById('stream-fps').value) || 10;
    
    if (!appId || !cameraId) {
        showToast('warning', 'Missing Selection', 'Please select an app and camera');
        return;
    }
    
    // Create WebSocket connection
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const sessionId = 'session-' + Date.now();
    state.ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream/${sessionId}`);
    
    state.ws.onopen = () => {
        state.ws.send(JSON.stringify({
            type: 'config',
            app_id: appId,
            camera_id: cameraId,
            fps: fps,
            visualize: true,
        }));
        
        state.streaming = true;
        state.streamFrameCount = 0;
        state.streamStartTime = Date.now();
        
        document.getElementById('stream-start-btn').classList.add('hidden');
        document.getElementById('stream-stop-btn').classList.remove('hidden');
        document.getElementById('stream-live-badge').classList.remove('hidden');
        
        showToast('success', 'Stream Started', 'Live inspection started');
    };
    
    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'frame') {
            state.streamFrameCount++;
            
            // Update image
            if (data.image_base64) {
                document.getElementById('stream-video').src = 
                    `data:image/jpeg;base64,${data.image_base64}`;
            }
            
            // Update metrics
            if (data.result) {
                document.getElementById('stream-decision').textContent = 
                    data.result.decision?.toUpperCase() || '--';
                document.getElementById('stream-score').textContent = 
                    data.result.anomaly_score?.toFixed(3) || '--';
                
                const badge = document.getElementById('stream-decision-badge');
                badge.textContent = data.result.decision?.toUpperCase() || '--';
                badge.className = `stream-badge ${data.result.decision}`;
            }
            
            // Update FPS
            const elapsed = (Date.now() - state.streamStartTime) / 1000;
            const fps = state.streamFrameCount / elapsed;
            document.getElementById('stream-fps-badge').textContent = `${fps.toFixed(1)} FPS`;
        }
    };
    
    state.ws.onerror = (error) => {
        showToast('error', 'Stream Error', 'WebSocket connection error');
        stopStream();
    };
    
    state.ws.onclose = () => {
        if (state.streaming) {
            showToast('info', 'Stream Ended', 'Connection closed');
            stopStream();
        }
    };
}

function stopStream() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
    
    state.streaming = false;
    
    document.getElementById('stream-start-btn').classList.remove('hidden');
    document.getElementById('stream-stop-btn').classList.add('hidden');
    document.getElementById('stream-live-badge').classList.add('hidden');
}

// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(type, title, message) {
    const container = document.getElementById('toast-container');
    
    const icons = {
        success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-icon">${icons[type]}</div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });
    
    // Auto remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================================================
// File Upload Handlers
// ============================================================================

function setupFileHandlers() {
    // Upload modal dropzone
    const uploadDropzone = document.getElementById('upload-dropzone');
    const uploadInput = document.getElementById('upload-file-input');
    
    uploadDropzone.addEventListener('click', () => uploadInput.click());
    uploadDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadDropzone.classList.add('dragover');
    });
    uploadDropzone.addEventListener('dragleave', () => {
        uploadDropzone.classList.remove('dragover');
    });
    uploadDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadDropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && (file.name.endsWith('.mvapp') || file.name.endsWith('.zip'))) {
            state.uploadFile = file;
            uploadDropzone.querySelector('h4').textContent = file.name;
            document.getElementById('upload-btn').disabled = false;
        }
    });
    uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            state.uploadFile = file;
            uploadDropzone.querySelector('h4').textContent = file.name;
            document.getElementById('upload-btn').disabled = false;
        }
    });
    
    // Inference dropzone
    const inferenceDropzone = document.getElementById('inference-dropzone');
    const inferenceInput = document.getElementById('inference-file-input');
    
    inferenceDropzone.addEventListener('click', () => inferenceInput.click());
    inferenceDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        inferenceDropzone.classList.add('dragover');
    });
    inferenceDropzone.addEventListener('dragleave', () => {
        inferenceDropzone.classList.remove('dragover');
    });
    inferenceDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        inferenceDropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleInferenceFile(file);
        }
    });
    inferenceInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleInferenceFile(file);
        }
    });
}

function handleInferenceFile(file) {
    state.selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-container').innerHTML = 
            `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
    
    // Update dropzone
    document.getElementById('inference-dropzone').querySelector('h4').textContent = file.name;
    
    // Enable button if app is selected
    const appId = document.getElementById('inference-app-select').value;
    document.getElementById('run-inference-btn').disabled = !appId;
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            if (page) navigateTo(page);
        });
    });
    
    // Image mode - App select change
    const inferenceAppSelect = document.getElementById('inference-app-select');
    if (inferenceAppSelect) {
        inferenceAppSelect.addEventListener('change', (e) => {
            const hasApp = !!e.target.value;
            const hasFile = !!state.selectedFile;
            document.getElementById('run-inference-btn').disabled = !hasApp || !hasFile;
        });
    }
    
    // Video mode - App select change
    const videoAppSelect = document.getElementById('video-app-select');
    if (videoAppSelect) {
        videoAppSelect.addEventListener('change', (e) => {
            const hasApp = !!e.target.value;
            const hasFile = !!state.videoFile;
            document.getElementById('video-process-btn').disabled = !hasApp || !hasFile;
        });
    }
    
    // Live mode - App and camera select change
    const liveAppSelect = document.getElementById('live-app-select');
    const liveCameraSelect = document.getElementById('live-camera-select');
    
    if (liveAppSelect) {
        liveAppSelect.addEventListener('change', updateLiveButtonState);
    }
    if (liveCameraSelect) {
        liveCameraSelect.addEventListener('change', updateLiveButtonState);
    }
    
    // Run inference button
    const runBtn = document.getElementById('run-inference-btn');
    if (runBtn) {
        runBtn.addEventListener('click', runInference);
    }
    
    // Setup file handlers
    setupFileHandlers();
    setupVideoHandlers();
}

function updateLiveButtonState() {
    const hasApp = !!document.getElementById('live-app-select')?.value;
    const hasCamera = !!document.getElementById('live-camera-select')?.value;
    const btn = document.getElementById('live-start-btn');
    if (btn) {
        btn.disabled = !hasApp || !hasCamera;
    }
}

// ============================================================================
// Initialization
// ============================================================================

async function init() {
    setupEventListeners();
    
    // Load initial data
    try {
        const [apps, cameras] = await Promise.all([
            API.getApps().catch(() => []),
            API.getCameras().catch(() => []),
        ]);
        
        state.apps = apps;
        state.cameras = cameras;
        
        // Update UI
        document.getElementById('apps-count').textContent = apps.length;
        populateAppSelects();
        populateCameraSelects();
        
        // Initialize inference mode
        switchInferenceMode('image');
        
        // Load dashboard
        refreshDashboard();
    } catch (error) {
        console.error('Initialization error:', error);
    }
}

// Expose functions to global scope for onclick handlers
window.switchInferenceMode = switchInferenceMode;
window.processVideo = processVideo;
window.stopVideoProcessing = stopVideoProcessing;
window.startLiveStream = startLiveStream;
window.stopLiveStream = stopLiveStream;
window.startStream = startStream;
window.stopStream = stopStream;
window.navigateTo = navigateTo;
window.showUploadModal = showUploadModal;
window.hideUploadModal = hideUploadModal;
window.uploadApp = uploadApp;
window.showCameraModal = showCameraModal;
window.hideCameraModal = hideCameraModal;
window.addCamera = addCamera;
window.disconnectCamera = disconnectCamera;
window.refreshApps = refreshApps;
window.refreshCameras = refreshCameras;
window.unloadApp = unloadApp;
window.selectApp = selectApp;

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);

