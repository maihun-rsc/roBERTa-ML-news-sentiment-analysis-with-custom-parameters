document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyze-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    const welcomeState = document.getElementById('welcome-state');
    const resultsPanel = document.getElementById('results-panel');
    const resTitle = document.getElementById('res-title');
    const resContext = document.getElementById('res-context');
    const barsContainer = document.getElementById('bars-container');
    
    const errorToast = document.getElementById('error-message');

    function showError(msg) {
        errorToast.textContent = msg;
        errorToast.classList.remove('hidden');
        setTimeout(() => {
            errorToast.classList.add('hidden');
        }, 5000);
    }

    // Tab Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    let activeMode = 'url-mode';

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active content
            activeMode = btn.dataset.tab;
            tabContents.forEach(content => {
                if(content.id === activeMode) {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        let payload = {};
        if (activeMode === 'url-mode') {
            payload.url = document.getElementById('url').value;
        } else {
            payload.raw_text = document.getElementById('raw_text').value;
            payload.outlet = document.getElementById('outlet').value;
        }
        payload.entity = document.getElementById('entity').value;
        
        // UI Loading State
        submitBtn.disabled = true;
        btnText.textContent = 'Executing...';
        loader.classList.remove('hidden');
        
        // Hide welcome state, show results panel (but maybe dim it)
        welcomeState.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to analyze article');
            }
            
            // Update Engine Info
            const engineDisplay = document.getElementById('engine-display');
            if (engineDisplay && data.engine) {
                engineDisplay.textContent = data.engine;
            }
            
            // Populate Results
            if (data.auto_entity) {
                resTitle.innerHTML = `${data.title} <br><span style="font-size: 0.85rem; color: #10b981; font-family: monospace;">[Auto-Detected Target: ${data.auto_entity}]</span>`;
            } else {
                resTitle.textContent = data.title;
            }
            resContext.textContent = `"...${data.context}..."`;
            
            // Clear old bars
            barsContainer.innerHTML = '';
            
            // Sort scores descending
            const sortedScores = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);
            
            // Render Bars
            sortedScores.forEach(([label, score]) => {
                const percentage = (score * 100).toFixed(1);
                
                const barRow = document.createElement('div');
                barRow.className = 'bar-row';
                
                barRow.innerHTML = `
                    <div class="bar-labels">
                        <span class="label-name">${label}</span>
                        <span class="label-score">${percentage}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill fill-${label.replace(/\s+/g, '-')}"></div>
                    </div>
                `;
                
                barsContainer.appendChild(barRow);
                
                // Animate fill after a tiny delay to trigger CSS transition
                setTimeout(() => {
                    const fillElement = barRow.querySelector('.bar-fill');
                    fillElement.style.width = `${percentage}%`;
                }, 50);
            });
            
            resultsPanel.classList.remove('hidden');
            
        } catch (err) {
            showError(err.message);
            // If it failed and we haven't shown results before, show welcome state again
            if (resTitle.textContent === '...') {
                welcomeState.classList.remove('hidden');
            }
        } finally {
            // Restore UI
            submitBtn.disabled = false;
            btnText.textContent = 'Execute Analysis';
            loader.classList.add('hidden');
        }
    });
});
