"""Generate a Gantt chart for the project workplan — midterm submission point."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Tasks: (name, start_week, duration_weeks, category, done)
# done = True means completed before midterm, False = still to do
tasks = [
    # Phase 1: Foundation (completed)
    ("Project setup & structure", 1, 2, "core", True),
    ("Input validation module", 1, 2, "core", True),
    ("Risk questionnaire design", 2, 2, "core", True),
    ("Risk classifier (rule-based)", 2, 2, "core", True),
    ("Suitability engine (5×3 matrix)", 3, 2, "core", True),
    # Phase 2: ML Pipeline (completed)
    ("Feature engineering (9 indicators)", 4, 2, "ml", True),
    ("Data acquisition (yfinance)", 5, 1, "ml", True),
    ("Model training (LR, DT, RF)", 5, 2, "ml", True),
    ("Model evaluation & selection", 6, 1, "ml", True),
    ("Model export & stock recommender", 6, 1, "ml", True),
    # Phase 3: Web App & Backtesting (completed)
    ("Flask backend + API routes", 7, 2, "web", True),
    ("HTML/CSS frontend (cards, modals)", 7, 2, "web", True),
    ("Chart.js price graphs", 8, 1, "web", True),
    ("Walk-forward backtesting module", 7, 2, "ml", True),
    ("LLM chatbot + hallucination filter", 8, 1, "web", True),
    ("Property & unit tests", 3, 6, "testing", True),
    # Post-feedback corrective work (completed)
    ("Leakage diagnosis & correction", 9, 1, "rigour", True),
    ("Purged chronological evaluation", 9, 1, "rigour", True),
    ("Statistical inference layer", 10, 1, "rigour", True),
    ("Regime analysis (ANOVA, Tukey)", 10, 1, "rigour", True),
    ("Stationarity diagnostics (ADF/KPSS)", 11, 1, "rigour", True),
    ("LSTM sequence baseline", 11, 1, "rigour", True),
    ("Questionnaire reliability analysis", 12, 1, "rigour", True),
    ("Suitability matrix validation", 12, 1, "rigour", True),
    ("LLM filter evaluation", 12, 1, "rigour", True),
    # === CURRENT POSITION (end of week 12) ===
    # Remaining work
    ("Stationary feature redesign", 13, 2, "enhance", False),
    ("Regime detection & abstention", 13, 2, "enhance", False),
    ("Confidence scores in UI", 14, 1, "enhance", False),
    ("User evaluation (SUS, 20-30 users)", 13, 2, "eval", False),
    ("Expert review of suitability matrix", 14, 1, "eval", False),
    ("Expanded LLM filter corpus", 14, 1, "eval", False),
    ("Final report: all chapters", 15, 2, "report", False),
    ("Video demonstration", 16, 1, "report", False),
    ("Final submission preparation", 16, 1, "report", False),
]

# Colours
colours = {
    "core": "#3b82f6",
    "ml": "#8b5cf6",
    "testing": "#06b6d4",
    "web": "#f59e0b",
    "rigour": "#0ea5e9",
    "enhance": "#ec4899",
    "eval": "#10b981",
    "report": "#6b7280",
}

fig, ax = plt.subplots(figsize=(14, 10))

n = len(tasks)
for i, (name, start_week, duration, cat, done) in enumerate(reversed(tasks)):
    y = i
    left = (start_week - 1) * 7
    width = duration * 7
    alpha = 0.9 if done else 0.45
    hatch = '' if done else '///'

    bar = ax.barh(y, width, left=left, height=0.6,
                  color=colours[cat], alpha=alpha,
                  edgecolor='white', linewidth=0.5)
    if hatch:
        bar[0].set_hatch(hatch)
        bar[0].set_edgecolor(colours[cat])

    # Label
    text_color = 'white' if done else '#1e293b'
    ax.text(left + 2, y, name, va='center', ha='left',
            fontsize=7.5, fontweight='500', color=text_color)

# Midterm line
midterm_x = 12 * 7  # End of week 12
ax.axvline(x=midterm_x, color='#ef4444', linestyle='-', linewidth=2.5, zorder=5)
ax.text(midterm_x + 2, n + 0.3, "← YOU ARE HERE\n   (Midterm Submission)",
        ha='left', va='bottom', fontsize=9, color='#ef4444', fontweight='bold')

# Milestones
milestones = [
    (2 * 7, "Core modules\ncomplete"),
    (6 * 7, "ML pipeline\ncomplete"),
    (8 * 7, "Prototype\nworking"),
    (13 * 7, "Evaluation\ncomplete"),
    (16 * 7, "Final\nsubmission"),
]
for x, label in milestones:
    ax.plot(x, -1.2, marker='^', color='#1e293b', markersize=7, zorder=5)
    ax.text(x, -2, label, ha='center', va='top', fontsize=6.5, color='#475569')

# X-axis
week_ticks = [(i * 7) for i in range(17)]
week_labels = [f"W{i+1}" for i in range(17)]
ax.set_xticks(week_ticks)
ax.set_xticklabels(week_labels, fontsize=7.5)

# Add date labels below
date_labels = [
    (0, "16 Jun"), (4*7, "14 Jul"), (8*7, "11 Aug"),
    (12*7, "8 Sep"), (16*7, "6 Oct")
]
for x, label in date_labels:
    ax.text(x, -3.5, label, ha='center', fontsize=7, color='#64748b')

# Y-axis
ax.set_yticks(range(n))
ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=7.5)
ax.set_ylim(-4, n + 2)
ax.set_xlim(-3, 17 * 7 + 5)

# Legend
legend_patches = [
    mpatches.Patch(color=colours["core"], label="Core Modules (done)"),
    mpatches.Patch(color=colours["ml"], label="ML & Backtesting (done)"),
    mpatches.Patch(color=colours["testing"], label="Testing (done)"),
    mpatches.Patch(color=colours["rigour"], label="Evaluation rigour (done)"),
    mpatches.Patch(color=colours["web"], label="Web App & UI (done)"),
    mpatches.Patch(color=colours["enhance"], alpha=0.45, label="Enhancements (planned)"),
    mpatches.Patch(color=colours["eval"], alpha=0.45, label="Evaluation (planned)"),
    mpatches.Patch(color=colours["report"], alpha=0.45, label="Report & Submission (planned)"),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7.5, framealpha=0.95)

# Title
ax.set_title(
    "CM3070 Final Year Project — Gantt Chart\n"
    "AI-Powered Stock Advisor  |  16-Week Timeline",
    fontsize=11, fontweight='bold', pad=15
)
ax.set_xlabel("Project Timeline", fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.15)

plt.tight_layout()
plt.savefig("docs/gantt_chart.png", dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Gantt chart saved to: docs/gantt_chart.png")
