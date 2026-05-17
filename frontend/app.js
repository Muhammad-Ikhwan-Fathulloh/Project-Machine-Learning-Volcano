const API_BASE_URL = "http://localhost:8080";
let isApiOnline = false;

// On Load initialization
window.addEventListener('DOMContentLoaded', () => {
    checkApiHealth();
    fetchLogs();
    // Poll health every 5 seconds
    setInterval(checkApiHealth, 5000);
});

// Toast Helper
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toast-icon');
    const msgSpan = document.getElementById('toast-message');
    
    toast.className = 'toast';
    toast.classList.add(`toast-${type}`);
    
    // Set icons
    if (type === 'success') {
        icon.className = 'fa-solid fa-circle-check toast-icon';
    } else if (type === 'warning') {
        icon.className = 'fa-solid fa-triangle-exclamation toast-icon';
    } else {
        icon.className = 'fa-solid fa-circle-exclamation toast-icon';
    }
    
    msgSpan.innerText = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// Check Backend Health Status
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const dot = document.getElementById('api-status-dot');
        const text = document.getElementById('api-status-text');
        
        if (response.ok) {
            const data = await response.json();
            isApiOnline = true;
            dot.className = 'status-dot online';
            text.innerText = data.model_loaded ? 'API Online' : 'API Online (No Model)';
            
            // Enable/Disable buttons based on model state
            document.getElementById('predict-submit-btn').disabled = !data.model_loaded;
        } else {
            throw new Error('API Unhealthy');
        }
    } catch (err) {
        isApiOnline = false;
        const dot = document.getElementById('api-status-dot');
        const text = document.getElementById('api-status-text');
        dot.className = 'status-dot offline';
        text.innerText = 'API Offline';
        document.getElementById('predict-submit-btn').disabled = true;
    }
}

// 1. Predict Volcano Shape
async function predictVolcano() {
    if (!isApiOnline) {
        showToast("Cannot connect to Volcano API server.", "danger");
        return;
    }

    const tinggi = parseFloat(document.getElementById('p-tinggi').value);
    const lat = parseFloat(document.getElementById('p-lat').value);
    const lon = parseFloat(document.getElementById('p-lon').value);
    
    const submitBtn = document.getElementById('predict-submit-btn');
    const resultBox = document.getElementById('predict-result-box');
    
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tinggi_meter: tinggi, lat: lat, lon: lon })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Inference failed");
        }

        const res = await response.json();
        
        // Render predictions
        document.getElementById('res-shape').innerText = res.prediction;
        
        const confidencePct = Math.round(res.confidence * 100);
        document.getElementById('res-conf').innerText = `${confidencePct}%`;
        
        const fillBar = document.getElementById('res-conf-bar');
        fillBar.style.width = `${confidencePct}%`;
        
        // Color mapping based on confidence
        if (res.confidence < 0.5) {
            fillBar.style.background = 'var(--danger-color)';
            showToast("Low confidence shape classified! Dispatching warnings...", "warning");
        } else {
            fillBar.style.background = 'var(--accent-gradient)';
        }
        
        document.getElementById('res-log-id').innerText = `Log ID: ${res.input.id}`;
        resultBox.style.display = 'flex';
        
        showToast("Volcano shape classified successfully!");
        fetchLogs(); // refresh logs
        
    } catch (err) {
        showToast(err.message, "danger");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Shape';
    }
}

// 2. Add Labeled Training coordinate
async function addTrainingSample() {
    if (!isApiOnline) {
        showToast("Cannot connect to Volcano API server.", "danger");
        return;
    }

    const tinggi = parseFloat(document.getElementById('t-tinggi').value);
    const lat = parseFloat(document.getElementById('t-lat').value);
    const lon = parseFloat(document.getElementById('t-lon').value);
    const bentuk = document.getElementById('t-bentuk').value;
    
    if (!bentuk) {
        showToast("Please choose a target shape class", "warning");
        return;
    }

    const submitBtn = document.getElementById('add-data-submit-btn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

    try {
        const response = await fetch(`${API_BASE_URL}/add-training-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tinggi_meter: tinggi, lat: lat, lon: lon, bentuk: bentuk })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Saving failed");
        }

        const res = await response.json();
        showToast("Sample saved directly to DynamoDB dataset!");
        document.getElementById('add-data-form').reset();
        fetchLogs(); // refresh list
        
    } catch (err) {
        showToast(err.message, "danger");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fa-solid fa-database"></i> Add to DynamoDB';
    }
}

// 3. Verify Inference Log
async function verifyLog(id, selectId) {
    if (!isApiOnline) {
        showToast("Cannot connect to Volcano API server.", "danger");
        return;
    }

    const bentuk = document.getElementById(selectId).value;
    if (!bentuk) {
        showToast("Select a verified shape first", "warning");
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/verify-prediction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: id, bentuk: bentuk })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Verification failed");
        }

        showToast("Prediction promoted to verified training sample!");
        fetchLogs(); // Refresh
        
    } catch (err) {
        showToast(err.message, "danger");
    }
}

// 4. Retrieve Dataset and Logs
async function fetchLogs() {
    if (!isApiOnline) return;

    const container = document.getElementById('logs-container');
    
    try {
        const response = await fetch(`${API_BASE_URL}/logs`);
        if (!response.ok) throw new Error("Failed to load logs");
        
        const data = await response.json();
        
        if (data.logs.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 3rem 0; color: var(--text-secondary);">
                    <i class="fa-solid fa-folder-open" style="font-size: 1.5rem; margin-bottom: 0.75rem;"></i>
                    <p>No records in DynamoDB yet.</p>
                </div>`;
            return;
        }
        
        container.innerHTML = '';
        
        data.logs.forEach((log, index) => {
            const isTraining = log.is_training_sample === true;
            const date = log.timestamp ? new Date(log.timestamp).toLocaleString() : 'Unknown Time';
            
            const selectId = `shapes-select-${index}`;
            
            let actionHtml = '';
            if (!isTraining) {
                actionHtml = `
                    <div class="log-action-panel">
                        <span style="font-size: 0.75rem; font-weight: 600; color: var(--text-secondary);">Verify Ground Truth:</span>
                        <select id="${selectId}" class="verify-select">
                            <option value="${log.predicted_bentuk}" selected>${log.predicted_bentuk}</option>
                            <option value="stratovulkan">stratovulkan</option>
                            <option value="kompleks">kompleks</option>
                            <option value="kaldera">kaldera</option>
                            <option value="perisai">perisai</option>
                            <option value="kubah lava">kubah lava</option>
                            <option value="kerucut bara">kerucut bara</option>
                            <option value="bawah laut">bawah laut</option>
                            <option value="Fumarol">Fumarol</option>
                            <option value="supervulkan">supervulkan</option>
                        </select>
                        <button onclick="verifyLog('${log.id}', '${selectId}')" class="verify-btn">Verify</button>
                    </div>
                `;
            }
            
            const itemHtml = `
                <div class="log-item">
                    <div class="log-header">
                        <span class="log-badge ${isTraining ? 'badge-training' : 'badge-inference'}">
                            ${isTraining ? 'Training Labeled' : 'API Log'}
                        </span>
                        <span class="log-time">${date}</span>
                    </div>
                    <div class="log-body">
                        <strong>Shape:</strong> ${isTraining ? (log.bentuk || log.actual_bentuk) : log.predicted_bentuk}
                        ${!isTraining ? ` <span style="color: var(--text-secondary); font-size: 0.8rem;">(Conf: ${Math.round(log.confidence * 100)}%)</span>` : ''}
                    </div>
                    <div class="log-coords">
                        H: ${log.tinggi_meter}m | Lat: ${log.lat} | Lon: ${log.lon}
                    </div>
                    ${actionHtml}
                </div>
            `;
            
            container.innerHTML += itemHtml;
        });
        
    } catch (err) {
        container.innerHTML = `
            <div style="text-align: center; padding: 2rem 0; color: var(--danger-color);">
                <i class="fa-solid fa-triangle-exclamation" style="font-size: 1.5rem; margin-bottom: 0.75rem;"></i>
                <p>Error listing logs: ${err.message}</p>
            </div>`;
    }
}

// 5. Retrain Model
async function triggerRetraining() {
    if (!isApiOnline) {
        showToast("Cannot connect to Volcano API server.", "danger");
        return;
    }

    const spinner = document.getElementById('retrain-spinner');
    const statusText = document.getElementById('retrain-status-text');
    const codeOutput = document.getElementById('retrain-code-output');
    const metricsTable = document.getElementById('retrain-metrics-table');
    const metricsTbody = document.getElementById('retrain-metrics-tbody');
    const btn = document.getElementById('retrain-btn');
    
    btn.disabled = true;
    spinner.classList.add('active');
    statusText.innerHTML = '<span class="pulse">Retraining ML Model...</span>';
    codeOutput.style.display = 'none';
    metricsTable.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/retrain`, {
            method: 'POST'
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Retraining pipeline failed");
        }

        const res = await response.json();
        
        // Show completion status
        statusText.innerText = "Model Retrained successfully!";
        showToast("ML Model Retrained & Hot-Reloaded successfully!");
        
        // Display text metrics
        codeOutput.innerText = `>>> Retraining completed: ${res.message}\n>>> Training Score: ${res.metrics.training_score * 100}%\n>>> Total Training Samples: ${res.metrics.total_samples} (incl. ${res.metrics.custom_samples_used} custom verified)\n>>> Class Labels: ${res.metrics.classes.join(', ')}`;
        codeOutput.style.display = 'block';
        
        // Render classification report inside beautiful table
        metricsTbody.innerHTML = '';
        
        // Iterate over class reports
        for (const className in res.classification_report) {
            if (className === 'accuracy') continue;
            
            const scoreObj = res.classification_report[className];
            const tr = document.createElement('tr');
            
            tr.innerHTML = `
                <td style="font-weight: 600;">${className}</td>
                <td>${Math.round(scoreObj.precision * 100)}%</td>
                <td>${Math.round(scoreObj.recall * 100)}%</td>
                <td>${Math.round(scoreObj.f1_score * 100)}%</td>
                <td>${scoreObj.support}</td>
            `;
            metricsTbody.appendChild(tr);
        }
        
        metricsTable.style.display = 'table';
        checkApiHealth(); // refresh model loaded status
        
    } catch (err) {
        statusText.innerText = "Retraining Failed";
        showToast(err.message, "danger");
    } finally {
        spinner.classList.remove('active');
        btn.disabled = false;
    }
}
