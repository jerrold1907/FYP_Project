/**
 * AI Stock Advisor - Frontend Logic
 */

let stockData = []; // Store for modal access

// === Get Recommendations ===
async function getRecommendations() {
    const questions = ['loss_reaction', 'investment_goal', 'time_horizon',
                       'emotional_comfort', 'risk_return_preference'];
    const answers = {};

    for (const qid of questions) {
        const selected = document.querySelector(`input[name="${qid}"]:checked`);
        if (!selected) {
            alert('Please answer all 5 questions before getting recommendations.');
            return;
        }
        answers[qid] = parseInt(selected.value);
    }

    document.getElementById('submit-btn').disabled = true;
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results-section').classList.add('hidden');

    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ answers }),
        });

        const data = await response.json();
        if (data.error) { alert('Error: ' + data.error); return; }

        stockData = data.stocks;
        displayResults(data);

        // Cache for chatbot page
        localStorage.setItem('stockData', JSON.stringify(data.stocks));
        localStorage.setItem('riskCategory', data.risk_profile.category);
    } catch (err) {
        alert('Request failed: ' + err.message);
    } finally {
        document.getElementById('submit-btn').disabled = false;
        document.getElementById('loading').classList.add('hidden');
    }
}

// === Display Results ===
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');

    // Profile
    const profile = data.risk_profile;
    document.getElementById('result-category').textContent =
        getCategoryEmoji(profile.category) + ' ' + profile.category;
    document.getElementById('result-score').textContent =
        `Risk Score: ${profile.total_score}/20`;
    document.getElementById('result-desc').textContent = profile.description;

    // Subtitle
    const suitable = data.stocks.filter(s => s.suitability.rating === 'Suitable').length;
    const caution = data.stocks.filter(s => s.suitability.rating === 'Use Caution').length;
    document.getElementById('results-subtitle').textContent =
        `${suitable} suitable, ${caution} with caution — click any stock for details.`;

    // Compact cards
    const grid = document.getElementById('stock-cards');
    grid.innerHTML = '';

    data.stocks.forEach((stock, idx) => {
        grid.appendChild(createCompactCard(stock, idx));
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// === Compact Card ===
function createCompactCard(stock, idx) {
    const rating = stock.suitability.rating;
    const ratingClass = rating === 'Suitable' ? 'suitable'
        : rating === 'Use Caution' ? 'caution' : 'not-suitable';
    const signalClass = stock.signal.toLowerCase();
    const changeSign = stock.price_change_30d >= 0 ? '+' : '';
    const changeClass = stock.price_change_30d >= 0 ? 'positive' : 'negative';

    const card = document.createElement('div');
    card.className = `stock-compact-card ${ratingClass}`;
    card.onclick = () => openModal(idx);
    card.innerHTML = `
        <div class="compact-top">
            <div>
                <div class="compact-ticker">${stock.ticker}</div>
                <div class="compact-name">${stock.name}</div>
            </div>
            <div class="compact-badges">
                <span class="badge badge-${ratingClass}">${getRatingIcon(rating)} ${rating}</span>
                <span class="badge badge-${signalClass}">${getSignalIcon(stock.signal)} ${stock.signal}</span>
            </div>
        </div>
        <div class="compact-bottom">
            <span class="compact-price">$${stock.current_price.toFixed(2)}</span>
            <span class="compact-change ${changeClass}">${changeSign}${stock.price_change_30d.toFixed(2)}%</span>
            <span class="compact-hint">Click for details →</span>
        </div>
    `;
    return card;
}

// === Modal ===
function openModal(idx) {
    const stock = stockData[idx];
    if (!stock) return;

    const rating = stock.suitability.rating;
    const ratingClass = rating === 'Suitable' ? 'suitable'
        : rating === 'Use Caution' ? 'caution' : 'not-suitable';
    const signalClass = stock.signal.toLowerCase();
    const changeSign = stock.price_change_30d >= 0 ? '+' : '';
    const changeClass = stock.price_change_30d >= 0 ? 'positive' : 'negative';

    // Generate pros/cons based on indicators
    const pros = [];
    const cons = [];
    if (stock.signal === 'Buy') pros.push('Strong buy signal from ML model');
    else if (stock.signal === 'Hold') pros.push('Stable outlook — model suggests holding');
    else cons.push('Negative outlook — model signals avoid');

    if (stock.rsi < 30) pros.push('RSI indicates oversold (potential rebound)');
    else if (stock.rsi > 70) cons.push('RSI indicates overbought (potential pullback)');
    else pros.push(`RSI at ${stock.rsi} — neutral momentum zone`);

    if (stock.macd > 0) pros.push('MACD positive — upward trend momentum');
    else cons.push('MACD negative — downward trend pressure');

    if (stock.price_change_30d > 5) pros.push(`Strong 30-day gain (${changeSign}${stock.price_change_30d.toFixed(1)}%)`);
    else if (stock.price_change_30d < -5) cons.push(`Significant 30-day decline (${stock.price_change_30d.toFixed(1)}%)`);

    if (stock.volatility < 2) pros.push('Low volatility — relatively stable');
    else if (stock.volatility > 4) cons.push('High volatility — expect large price swings');

    if (rating === 'Suitable') pros.push('Well-aligned with your risk profile');
    else if (rating === 'Not Suitable') cons.push('Does not match your current risk tolerance');

    const content = document.getElementById('modal-content');
    content.innerHTML = `
        <div class="modal-header">
            <h2>${stock.ticker} — ${stock.name}</h2>
            <p class="modal-subtitle">Analysis based on latest market data and AI model</p>
            <div class="modal-badges">
                <span class="badge badge-${ratingClass}">${getRatingIcon(rating)} ${rating}</span>
                <span class="badge badge-${signalClass}">${getSignalIcon(stock.signal)} ${stock.signal}</span>
            </div>
        </div>

        <div class="modal-section">
            <h3>Key Metrics</h3>
            <div class="modal-metrics">
                <div class="modal-metric">
                    <div class="m-label">Price</div>
                    <div class="m-value">$${stock.current_price.toFixed(2)}</div>
                </div>
                <div class="modal-metric">
                    <div class="m-label">30d Change</div>
                    <div class="m-value ${changeClass}">${changeSign}${stock.price_change_30d.toFixed(2)}%</div>
                </div>
                <div class="modal-metric">
                    <div class="m-label">RSI (14d)</div>
                    <div class="m-value">${stock.rsi}</div>
                </div>
                <div class="modal-metric">
                    <div class="m-label">MACD</div>
                    <div class="m-value">${stock.macd}</div>
                </div>
                <div class="modal-metric">
                    <div class="m-label">Volatility</div>
                    <div class="m-value">${stock.volatility}%</div>
                </div>
                <div class="modal-metric">
                    <div class="m-label">Signal</div>
                    <div class="m-value">${stock.signal}</div>
                </div>
            </div>
        </div>

        <div class="modal-section">
            <h3>Price History (90 Days)</h3>
            <div class="modal-chart">
                <canvas id="modal-chart" height="180"></canvas>
            </div>
        </div>

        <div class="modal-section">
            <h3>Strengths & Risks</h3>
            <div class="modal-pros-cons">
                <div>
                    <strong style="color: var(--success);">Strengths</strong>
                    <ul class="pros-list">
                        ${pros.map(p => `<li>${p}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <strong style="color: var(--danger);">Risks</strong>
                    <ul class="cons-list">
                        ${cons.length > 0 ? cons.map(c => `<li>${c}</li>`).join('') : '<li>No major risks identified</li>'}
                    </ul>
                </div>
            </div>
        </div>

        <div class="modal-section">
            <h3>AI Recommendation Explanation</h3>
            <div class="modal-explanation">
                ${stock.suitability.explanation}
            </div>
        </div>
    `;

    document.getElementById('modal-overlay').classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    // Render chart
    setTimeout(() => renderModalChart(stock), 50);
}

function closeModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('modal-overlay').classList.add('hidden');
    document.body.style.overflow = '';
}

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// === Chart Rendering ===
function renderModalChart(stock) {
    const ctx = document.getElementById('modal-chart');
    if (!ctx) return;

    const labels = stock.dates.map((d, i) => i % 10 === 0 ? d.slice(5) : '');
    const priceColor = stock.prices[stock.prices.length - 1] >= stock.prices[0]
        ? '#10b981' : '#ef4444';

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${stock.ticker} Price`,
                data: stock.prices,
                borderColor: priceColor,
                backgroundColor: priceColor + '15',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 }, color: '#9691b8' } },
                y: {
                    grid: { color: '#ece9f8' },
                    ticks: { font: { size: 10 }, color: '#9691b8', callback: v => '$' + v.toFixed(0) }
                }
            }
        }
    });
}

// === Helpers ===
function getCategoryEmoji(c) {
    return { 'Conservative':'🟢', 'Moderately Conservative':'🟡', 'Balanced':'🔵', 'Growth':'🟠', 'Aggressive':'🔴' }[c] || '⚪';
}
function getRatingIcon(r) {
    return { 'Suitable':'✅', 'Use Caution':'⚠️', 'Not Suitable':'❌' }[r] || '';
}
function getSignalIcon(s) {
    return { 'Buy':'🟢', 'Hold':'🟡', 'Avoid':'🔴' }[s] || '';
}

// === Backtest ===
async function runBacktest() {
    const btn = document.getElementById('bt-run-btn');
    const loading = document.getElementById('bt-loading');
    const resultsSection = document.getElementById('bt-results');

    const tickersStr = document.getElementById('bt-tickers').value;
    const tickers = tickersStr.split(',').map(t => t.trim()).filter(t => t);
    if (tickers.length === 0) { alert('Please enter at least one ticker.'); return; }

    const payload = {
        tickers,
        start_date: document.getElementById('bt-start').value,
        end_date: document.getElementById('bt-end').value,
        model_type: document.getElementById('bt-model').value,
        train_window: parseInt(document.getElementById('bt-train').value),
        test_window: parseInt(document.getElementById('bt-test').value),
    };

    btn.disabled = true;
    btn.textContent = 'Running...';
    loading.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    try {
        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (data.error) { alert('Backtest error: ' + data.error); return; }
        displayBacktestResults(data);
    } catch (err) {
        alert('Backtest failed: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Run Backtest';
        loading.classList.add('hidden');
    }
}

function displayBacktestResults(data) {
    const resultsSection = document.getElementById('bt-results');
    resultsSection.classList.remove('hidden');

    const agg = data.aggregate;

    // --- Aggregate Results Panel (right side) ---
    const aggTable = document.getElementById('bt-agg-table');
    const excessReturn = (agg.excess_return_pct || ((agg.total_return_pct||0) - (agg.benchmark_return_pct||0)));
    const sharpeStr = `${(agg.sharpe_ratio||0).toFixed(2)}`;
    const sharpeCIStr = agg.sharpe_ci_lower !== undefined
        ? `<br><span style="font-size:0.8rem;color:#8783a8;">(95% CI: [${agg.sharpe_ci_lower.toFixed(2)}, ${agg.sharpe_ci_upper.toFixed(2)}])</span>`
        : '';

    aggTable.innerHTML = `
        <tr style="border-bottom:1px solid #e7e5f2;"><td style="padding:0.5rem 0;">Total Return (Strategy)</td><td style="text-align:right;font-weight:600;color:${(agg.total_return_pct||0)>=0?'#10b981':'#ef4444'}">${(agg.total_return_pct||0)>=0?'+':''}${(agg.total_return_pct||0).toFixed(1)}%</td></tr>
        <tr style="border-bottom:1px solid #e7e5f2;"><td style="padding:0.5rem 0;">Total Return (Buy & Hold)</td><td style="text-align:right;font-weight:600;color:${(agg.benchmark_return_pct||0)>=0?'#10b981':'#ef4444'}">${(agg.benchmark_return_pct||0)>=0?'+':''}${(agg.benchmark_return_pct||0).toFixed(1)}%</td></tr>
        <tr style="border-bottom:1px solid #e7e5f2;"><td style="padding:0.5rem 0;">Excess Return</td><td style="text-align:right;font-weight:600;color:${excessReturn>=0?'#10b981':'#ef4444'}">${excessReturn>=0?'+':''}${excessReturn.toFixed(1)}%</td></tr>
        <tr style="border-bottom:1px solid #e7e5f2;"><td style="padding:0.5rem 0;">Win Rate (pooled)</td><td style="text-align:right;font-weight:600;">${(agg.pooled_win_rate_pct || agg.win_rate_pct||0).toFixed(1)}%</td></tr>
        <tr style="border-bottom:1px solid #e7e5f2;"><td style="padding:0.5rem 0;">Sharpe Ratio</td><td style="text-align:right;font-weight:600;">${sharpeStr}${sharpeCIStr}</td></tr>
        <tr style="border-bottom:1px solid #e7e5f2;"><td style="padding:0.5rem 0;">Max Drawdown</td><td style="text-align:right;font-weight:600;color:#ef4444;">${(agg.max_drawdown_pct||0).toFixed(1)}%</td></tr>
        <tr><td style="padding:0.5rem 0;">Total Trades</td><td style="text-align:right;font-weight:600;">${(agg.total_trades||0).toLocaleString()}</td></tr>
    `;

    // --- Combined Equity Curve Chart (left side) ---
    const chartCanvas = document.getElementById('bt-equity-chart');
    if (window._btChart) window._btChart.destroy();

    const strategyCurve = data.combined_strategy_curve || [];
    const benchmarkCurve = data.combined_benchmark_curve || [];
    const maxLen = Math.max(strategyCurve.length, benchmarkCurve.length);
    const labels = Array.from({length: maxLen}, (_, i) => i);

    window._btChart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Strategy (Net)',
                    data: strategyCurve,
                    borderColor: '#4f46e5',
                    backgroundColor: 'rgba(37,99,235,0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                },
                {
                    label: 'Buy & Hold',
                    data: benchmarkCurve,
                    borderColor: '#8783a8',
                    borderDash: [5, 3],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Strategy vs Buy & Hold (All Tickers)',
                    font: { size: 14 }
                },
                legend: { position: 'top' }
            },
            scales: {
                x: {
                    display: true,
                    title: { display: false },
                    ticks: { maxTicksLimit: 6, display: false }
                },
                y: {
                    title: { display: true, text: 'Cumulative Return' },
                    ticks: {
                        callback: (v) => v.toFixed(0) + '%'
                    }
                }
            },
            interaction: { intersect: false, mode: 'index' },
        }
    });

    // --- Per-Stock Summary Table ---
    const table = document.getElementById('bt-summary-table');
    if (data.summary && data.summary.length > 0) {
        const headers = Object.keys(data.summary[0]);
        table.querySelector('thead').innerHTML = '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';
        table.querySelector('tbody').innerHTML = data.summary.map(row =>
            '<tr>' + headers.map(h => `<td>${row[h]}</td>`).join('') + '</tr>'
        ).join('');
    }

    // --- Per-Stock Charts ---
    const chartsDiv = document.getElementById('bt-charts');
    chartsDiv.innerHTML = '';
    for (const [ticker, stockData] of Object.entries(data.per_stock)) {
        if (stockData.equity_curve && stockData.equity_curve.length > 0) {
            const canvasId = `chart-${ticker}`;
            chartsDiv.innerHTML += `<div style="margin-bottom:1rem;"><canvas id="${canvasId}" height="150"></canvas></div>`;
        }
    }
    // Render per-stock mini charts after DOM update
    setTimeout(() => {
        for (const [ticker, stockData] of Object.entries(data.per_stock)) {
            const canvasId = `chart-${ticker}`;
            const el = document.getElementById(canvasId);
            if (!el || !stockData.equity_curve) continue;
            new Chart(el, {
                type: 'line',
                data: {
                    labels: Array.from({length: stockData.equity_curve.length}, (_, i) => i),
                    datasets: [{
                        label: `${ticker} Strategy`,
                        data: stockData.equity_curve,
                        borderColor: '#4f46e5',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                    }, {
                        label: `${ticker} Benchmark`,
                        data: stockData.benchmark_curve || [],
                        borderColor: '#8783a8',
                        borderDash: [4,2],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { title: { display: true, text: ticker }, legend: { display: true } },
                    scales: { x: { display: false }, y: { ticks: { callback: v => '$' + v.toFixed(0) } } }
                }
            });
        }
    }, 100);

    // --- Detailed Reports ---
    const reportsDiv = document.getElementById('bt-reports');
    reportsDiv.innerHTML = '';
    for (const [ticker, stockData] of Object.entries(data.per_stock)) {
        if (stockData.report) {
            reportsDiv.innerHTML += `<h4>${ticker}</h4><div class="report-block">${stockData.report}</div>`;
        }
    }

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}
