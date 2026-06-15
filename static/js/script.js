// Dashboard JavaScript
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// File input change handler
fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload PNG or JPEG images only.');
        return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;
    hideError();

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        filePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
});

// Analyze button click handler
analyzeBtn.addEventListener('click', function() {
    if (!selectedFile) {
        showError('No file selected. Please choose an image first.');
        return;
    }

    uploadAndAnalyze(selectedFile);
});

// Upload and analyze function
function uploadAndAnalyze(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Hide preview and error, show loading
    filePreview.style.display = 'none';
    hideError();
    loadingSpinner.style.display = 'block';

    const startTime = Date.now();

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.error || 'Server error occurred');
            });
        }
        return response.json();
    })
    .then(data => {
        const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);
        
        // Store results in sessionStorage
        sessionStorage.setItem('analysisResults', JSON.stringify({
            ...data,
            processingTime: processingTime
        }));

        // Redirect to results page
        window.location.href = '/results';
    })
    .catch(error => {
        console.error('Error:', error);
        loadingSpinner.style.display = 'none';
        filePreview.style.display = 'block';
        showError(error.message || 'An error occurred during analysis. Please try again.');
    });
}

// Helper functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

// Drag and drop support
const uploadCard = document.querySelector('.upload-card');

uploadCard.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadCard.style.borderColor = 'rgba(102, 126, 234, 0.8)';
    uploadCard.style.transform = 'scale(1.02)';
});

uploadCard.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadCard.style.borderColor = '';
    uploadCard.style.transform = '';
});

uploadCard.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadCard.style.borderColor = '';
    uploadCard.style.transform = '';
    
    const file = e.dataTransfer.files[0];
    if (file) {
        // Create a new FileList-like object
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        
        // Trigger change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
});