/**
 * Floating AI Advisor chat widget.
 * Available on every page. Reads cached recommendation context from
 * localStorage so answers are personalised to the user's risk profile.
 */

const CW_STORAGE_KEY = 'cw_history';
let cwOpen = false;
let cwBusy = false;

/**
 * Read cached recommendation context.
 * risk_category stays null until the questionnaire is actually completed —
 * never substitute a default, or the advisor would assume a risk tolerance
 * the user never gave.
 */
function cwContext() {
    return {
        stock_data: JSON.parse(localStorage.getItem('stockData') || '[]'),
        risk_category: localStorage.getItem('riskCategory') || null,
    };
}

function toggleChatWidget() {
    cwOpen = !cwOpen;
    const panel = document.getElementById('chat-panel');
    const launcher = document.getElementById('chat-launcher');
    panel.classList.toggle('hidden', !cwOpen);
    launcher.classList.toggle('active', cwOpen);
    if (cwOpen) {
        document.getElementById('cw-input').focus();
        cwUpdateProfileBadge();
        cwScrollBottom();
    }
}

function cwScrollBottom() {
    const box = document.getElementById('cw-messages');
    box.scrollTop = box.scrollHeight;
}

function cwRender(text, type, save = true) {
    const box = document.getElementById('cw-messages');
    const wrap = document.createElement('div');
    wrap.className = `cw-msg ${type}`;
    wrap.innerHTML = `<div class="cw-bubble">${cwFormat(text)}</div>`;
    box.appendChild(wrap);
    cwScrollBottom();

    if (save) {
        const history = JSON.parse(sessionStorage.getItem(CW_STORAGE_KEY) || '[]');
        history.push({ text, type });
        sessionStorage.setItem(CW_STORAGE_KEY, JSON.stringify(history.slice(-40)));
    }
}

/**
 * Render a bot answer: optional AI summary paragraph, then the structured
 * data-grounded detail. Stored as one history entry so reloads look identical.
 */
function cwRenderAnswer(detail, summary) {
    const box = document.getElementById('cw-messages');
    const wrap = document.createElement('div');
    wrap.className = 'cw-msg bot';

    let inner = '';
    if (summary) {
        inner += `<div class="cw-summary"><span class="cw-summary-tag">AI summary</span>${cwFormat(summary)}</div>`;
    }
    inner += `<div class="cw-detail">${cwFormat(detail)}</div>`;
    wrap.innerHTML = `<div class="cw-bubble">${inner}</div>`;
    box.appendChild(wrap);
    cwScrollBottom();

    const history = JSON.parse(sessionStorage.getItem(CW_STORAGE_KEY) || '[]');
    history.push({ text: detail, summary: summary || null, type: 'bot' });
    sessionStorage.setItem(CW_STORAGE_KEY, JSON.stringify(history.slice(-40)));
}

/** Replay a stored answer without re-saving it to history. */
function cwRenderAnswerNoSave(detail, summary) {
    const box = document.getElementById('cw-messages');
    const wrap = document.createElement('div');
    wrap.className = 'cw-msg bot';
    let inner = '';
    if (summary) {
        inner += `<div class="cw-summary"><span class="cw-summary-tag">AI summary</span>${cwFormat(summary)}</div>`;
    }
    inner += `<div class="cw-detail">${cwFormat(detail)}</div>`;
    wrap.innerHTML = `<div class="cw-bubble">${inner}</div>`;
    box.appendChild(wrap);
    cwScrollBottom();
}

/** Minimal markdown: bold, bullets, tables, line breaks. */
function cwFormat(text) {
    const escaped = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    const lines = escaped.split('\n');
    let html = '';
    let inTable = false;

    for (const line of lines) {
        const trimmed = line.trim();

        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
            const cells = trimmed.slice(1, -1).split('|').map(c => c.trim());
            if (cells.every(c => /^-+$/.test(c))) continue; // separator row
            if (!inTable) { html += '<table class="cw-table">'; inTable = true; }
            const tag = /\*\*/.test(trimmed) ? 'th' : 'td';
            html += '<tr>' + cells.map(c => `<${tag}>${cwInline(c)}</${tag}>`).join('') + '</tr>';
            continue;
        }
        if (inTable) { html += '</table>'; inTable = false; }

        if (trimmed.startsWith('•') || trimmed.startsWith('-')) {
            html += `<div class="cw-bullet">${cwInline(trimmed.replace(/^[•-]\s*/, ''))}</div>`;
        } else if (trimmed === '') {
            html += '<div class="cw-space"></div>';
        } else {
            html += `<div>${cwInline(trimmed)}</div>`;
        }
    }
    if (inTable) html += '</table>';
    return html;
}

function cwInline(s) {
    return s
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>');
}

function cwTyping(on) {
    const existing = document.getElementById('cw-typing');
    if (on && !existing) {
        const box = document.getElementById('cw-messages');
        const el = document.createElement('div');
        el.id = 'cw-typing';
        el.className = 'cw-msg bot';
        el.innerHTML = '<div class="cw-bubble cw-typing"><span></span><span></span><span></span></div>';
        box.appendChild(el);
        cwScrollBottom();
    } else if (!on && existing) {
        existing.remove();
    }
}

async function sendChatWidget(preset) {
    if (cwBusy) return;
    const input = document.getElementById('cw-input');
    const message = (preset || input.value).trim();
    if (!message) return;

    if (!cwOpen) toggleChatWidget();
    input.value = '';
    cwRender(message, 'user');
    document.getElementById('cw-suggestions').classList.add('hidden');

    cwBusy = true;
    document.getElementById('cw-send').disabled = true;
    cwTyping(true);
    cwStatus('Thinking...', 'busy');

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, ...cwContext() }),
        });
        const data = await res.json();
        cwTyping(false);

        if (data.error) {
            cwRender(`Sorry — ${data.error}`, 'bot');
        } else {
            cwRenderAnswer(data.response, data.summary);
        }

        if (data.profile_assessed === false) {
            cwStatus('No risk profile set', 'warn');
        } else {
            cwStatus(data.llm_used ? 'Summarised by Llama 3.2' : 'Ready', 'ok');
        }
    } catch (err) {
        cwTyping(false);
        cwRender('I could not reach the server. Please try again.', 'bot');
        cwStatus('Offline', 'error');
    } finally {
        cwBusy = false;
        document.getElementById('cw-send').disabled = false;
        input.focus();
    }
}

function cwStatus(text, state) {
    document.getElementById('cw-status-text').textContent = text;
    const dot = document.getElementById('cw-status-dot');
    dot.className = 'cw-dot ' + (state || 'ok');
}

function clearChatWidget() {
    sessionStorage.removeItem(CW_STORAGE_KEY);
    document.getElementById('cw-messages').innerHTML = '';
    document.getElementById('cw-suggestions').classList.remove('hidden');
    cwGreet();
}

function cwGreet() {
    const ctx = cwContext();
    const greeting = ctx.risk_category
        ? `Hi. I have your **${ctx.risk_category}** profile and the latest analysis for ${ctx.stock_data.length} stocks loaded. Ask me anything.`
        : 'Hi. I am your AI stock advisor.\n\n**You have not completed the risk assessment yet**, so I cannot recommend stocks for you personally. Take the 5-question assessment on the Recommendations page and I will know your risk profile.\n\nUntil then I can still tell you about any stock, or compare two of them.';
    cwRender(greeting, 'bot', false);
    cwUpdateProfileBadge();
}

/** Reflect assessment status in the panel header. */
function cwUpdateProfileBadge() {
    const el = document.getElementById('cw-status-text');
    if (!el) return;
    const ctx = cwContext();
    if (ctx.risk_category) {
        el.textContent = `Profile: ${ctx.risk_category}`;
        cwStatus(`Profile: ${ctx.risk_category}`, 'ok');
    } else {
        cwStatus('No risk profile set', 'warn');
    }
}

// Restore conversation on page navigation
document.addEventListener('DOMContentLoaded', () => {
    const history = JSON.parse(sessionStorage.getItem(CW_STORAGE_KEY) || '[]');
    if (history.length) {
        history.forEach(m => {
            if (m.type === 'bot' && m.summary !== undefined) {
                cwRenderAnswerNoSave(m.text, m.summary);
            } else {
                cwRender(m.text, m.type, false);
            }
        });
        document.getElementById('cw-suggestions').classList.add('hidden');
        cwUpdateProfileBadge();
    } else {
        cwGreet();
    }

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && cwOpen) toggleChatWidget();
    });
});
