// Results Page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Get results from sessionStorage
    const resultsData = sessionStorage.getItem('analysisResults');
    
    if (!resultsData) {
        // No results found, redirect to dashboard
        window.location.href = '/';
        return;
    }

    const results = JSON.parse(resultsData);
    displayResults(results);
});

function displayResults(results) {
    const { label, confidence, raw_score, processingTime } = results;
    
    // Determine if image is real or fake
    const isReal = label === 'Real';
    
    // Update verdict card
    const verdictCard = document.getElementById('verdictCard');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictLabel = document.getElementById('verdictLabel');
    const verdictDescription = document.getElementById('verdictDescription');
    
    verdictCard.classList.add(isReal ? 'real' : 'fake');
    verdictIcon.classList.add(isReal ? 'real' : 'fake');
    
    // Set icon
    if (isReal) {
        verdictIcon.innerHTML = `
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
        `;
    } else {
        verdictIcon.innerHTML = `
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
        `;
    }
    
    verdictLabel.textContent = isReal ? 'Authentic Image' : 'Manipulated Image';
    verdictDescription.textContent = isReal 
        ? 'No signs of digital manipulation detected' 
        : 'Digital manipulation or forgery detected';
    
    // Update confidence metric
    document.getElementById('confidenceValue').textContent = confidence.toFixed(2) + '%';
    document.getElementById('confidenceProgress').style.width = confidence + '%';
    
    // Update raw score metric
    const rawScorePercent = (raw_score * 100).toFixed(2);
    document.getElementById('rawScoreValue').textContent = rawScorePercent + '%';
    document.getElementById('rawScoreProgress').style.width = rawScorePercent + '%';
    
    // Update accuracy metric (using confidence as accuracy for display)
    const accuracyValue = confidence.toFixed(2);
    document.getElementById('accuracyValue').textContent = accuracyValue + '%';
    document.getElementById('accuracyProgress').style.width = accuracyValue + '%';
    
    // Update technical details
    document.getElementById('classification').textContent = label;
    document.getElementById('rawScoreDetail').textContent = raw_score.toFixed(6);
    document.getElementById('confidenceDetail').textContent = confidence.toFixed(2) + '%';
    document.getElementById('processingTime').textContent = processingTime + 's';
    
    // Animate progress bars
    setTimeout(() => {
        animateProgressBars();
    }, 100);
    
    // Animate metrics
    animateMetrics(confidence, rawScorePercent, accuracyValue);
}

function animateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
}

function animateMetrics(confidence, rawScore, accuracy) {
    animateValue('confidenceValue', 0, confidence, 1500, '%');
    animateValue('rawScoreValue', 0, rawScore, 1500, '%');
    animateValue('accuracyValue', 0, accuracy, 1500, '%');
}

function animateValue(elementId, start, end, duration, suffix = '') {
    const element = document.getElementById(elementId);
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(2) + suffix;
    }, 16);
}

// Add animation class when page loads
window.addEventListener('load', function() {
    document.querySelectorAll('.metric-card, .result-card, .analysis-details').forEach((el, index) => {
        setTimeout(() => {
            el.style.animation = 'fadeInUp 0.6s ease forwards';
        }, index * 100);
    });
});

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card, .result-card, .analysis-details {
        opacity: 0;
    }
`;
document.head.appendChild(style);