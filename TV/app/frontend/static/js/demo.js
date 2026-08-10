const chatWindow = document.getElementById('chat-window');
const turnCounter = document.getElementById('turn-counter');
const datasetSelect = document.getElementById('dataset-select');
const labelSelect = document.getElementById('label-select');
const datasetInfo = document.getElementById('dataset-info');
const startBtn = document.getElementById('start-btn');
const tvValue = document.getElementById('tv-value');
const tvFill = document.getElementById('tv-fill');
const thresholdLabel = document.getElementById('threshold-label');
const resultPanel = document.getElementById('result-panel');
const verdictBlock = document.getElementById('verdict-block');
const verdictText = document.getElementById('verdict-text');
const predictedLabel = document.getElementById('predicted-label');
const groundTruthLabel = document.getElementById('ground-truth-label');

let datasets = [];
let currentBubble = null;

async function loadDatasets() {
    const res = await fetch('/api/datasets');
    datasets = await res.json();
    datasetSelect.innerHTML = '';
    for (const ds of datasets) {
        const opt = document.createElement('option');
        opt.value = ds.key;
        opt.textContent = ds.name;
        datasetSelect.appendChild(opt);
    }
    updateDatasetInfo();
}

function updateDatasetInfo() {
    const ds = datasets.find(d => d.key === datasetSelect.value);
    if (!ds) return;
    datasetInfo.textContent = `Threshold: ${ds.threshold.toFixed(4)} | Test set: ${ds.test_total} (${ds.test_jailbreak} jailbreak / ${ds.test_safe} safe)`;
    thresholdLabel.textContent = `Threshold: ${ds.threshold.toFixed(4)}`;
}

datasetSelect.addEventListener('change', updateDatasetInfo);

function appendBubble(role) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role === 'human' ? 'attacker' : 'target'}`;
    const label = document.createElement('div');
    label.className = 'message-label';
    label.innerText = role === 'human' ? 'Human' : 'Assistant';
    const textNode = document.createElement('span');
    wrapper.appendChild(label);
    wrapper.appendChild(textNode);
    chatWindow.appendChild(wrapper);
    return textNode;
}

function appendDivider(text, color) {
    const divider = document.createElement('div');
    divider.style = `text-align: center; color: ${color}; margin: 20px 0; border-bottom: 1px solid ${color}; line-height: 0.1em;`;
    divider.innerHTML = `<span style="background: #0f1115; padding: 0 15px; font-size: 0.85em; font-weight: bold; letter-spacing: 1px;">${text}</span>`;
    chatWindow.appendChild(divider);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function startDemo() {
    startBtn.disabled = true;
    chatWindow.innerHTML = '';
    resultPanel.classList.add('hidden');
    tvValue.innerText = '0.00';
    tvFill.style.width = '0%';
    turnCounter.innerText = 'Turn: 0';

    const dataset = datasetSelect.value;
    const label = labelSelect.value;

    const response = await fetch(`/api/demo/stream?dataset=${encodeURIComponent(dataset)}&label=${encodeURIComponent(label)}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.replace('data: ', ''));

            if (data.error) {
                appendDivider(`ERROR: ${data.error}`, '#ef4444');
                continue;
            }

            switch (data.status) {
                case 'turn_start':
                    turnCounter.innerText = `Turn: ${data.turn_index + 1}`;
                    currentBubble = appendBubble(data.role);
                    break;
                case 'turn_text':
                    if (currentBubble) currentBubble.innerText = data.text;
                    chatWindow.scrollTop = chatWindow.scrollHeight;
                    break;
                case 'tv_update':
                    tvValue.innerText = data.tv.toFixed(2);
                    tvFill.style.width = `${Math.min(data.tv, 1) * 100}%`;
                    break;
                case 'threshold_crossed':
                    appendDivider(`⚠ THRESHOLD CROSSED (TV=${data.tv.toFixed(4)} ≥ ${data.threshold.toFixed(4)})`, '#ef4444');
                    break;
                case 'verdict':
                    resultPanel.classList.remove('hidden');
                    predictedLabel.innerText = data.predicted;
                    groundTruthLabel.innerText = data.ground_truth;
                    verdictText.innerText = data.correct ? '✓ CORRECT' : '✗ WRONG';
                    verdictBlock.className = `judge-block ${data.correct ? 'safe' : 'jailbreak'}`;
                    break;
            }
        }
    }

    startBtn.disabled = false;
}

startBtn.addEventListener('click', startDemo);
loadDatasets();
