"""
Generate the Preliminary Project Report (.docx).

This revision responds to marker feedback by replacing asserted claims with
measured evidence. Every figure below traces to a script in experiments/.

Run: python docs/generate_report.py
"""
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)


def heading_c(text, level=0):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return h


def para(text, bold=False, italic=False, align=None, after=Pt(6)):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = after
    return p


def table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = h
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
    for ri, row in enumerate(rows):
        for ci, v in enumerate(row):
            c = t.rows[ri + 1].cells[ci]
            c.text = str(v)
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(10)
    doc.add_paragraph()
    return t


# ============================== COVER ==============================
doc.add_paragraph()
doc.add_paragraph()
heading_c("UNIVERSITY OF LONDON", 1)
heading_c("INTERNATIONAL PROGRAMMES", 2)
heading_c("BSc Computer Science and Related Subjects", 2)
doc.add_paragraph()
heading_c("CM3070 PROJECT", 1)
heading_c("PRELIMINARY PROJECT REPORT", 2)
doc.add_paragraph()
para("Project Idea 4.2: Financial Advisor Bot",
     bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
para("(AI-Powered Stock Recommendation System)",
     bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
doc.add_paragraph()
para("Author: Jerrold Ng Yang Zhong")
para("Student Number: 10262360")
para("Date of Submission: [DD/MM/YYYY]")
para("Supervisor: [Supervisor Name]")

# ============================== TOC ==============================
doc.add_page_break()
doc.add_heading("Contents", level=1)
for item in [
    "CHAPTER 1: INTRODUCTION",
    "CHAPTER 2: LITERATURE REVIEW",
    "CHAPTER 3: PROJECT DESIGN",
    "CHAPTER 4: FEATURE PROTOTYPE AND EVALUATION",
    "CHAPTER 5: APPENDICES",
    "CHAPTER 6: REFERENCES",
]:
    para(item)

# ============================== CH 1 ==============================
doc.add_page_break()
doc.add_heading("CHAPTER 1: INTRODUCTION", level=1)

doc.add_heading("1.1 Project Concept", level=2)
para(
    "This project develops an AI-powered financial advisor bot for the stock "
    "market, following Project Idea 4.2. The system combines machine "
    "learning-based stock signal prediction with a scenario-based investor risk "
    "questionnaire to produce personalised suitability recommendations. It "
    "addresses a gap in retail investment tools: most stock screeners issue "
    "generic signals without regard to an individual's risk tolerance, "
    "emotional comfort with volatility, or investment horizon."
)
para("The system comprises five components:")
para(
    "1. Risk Questionnaire \u2014 five scenario-based items, informed by Grable "
    "and Lytton (1999), classifying investors as Conservative, Moderately "
    "Conservative, Balanced, Growth or Aggressive."
)
para(
    "2. Stock Recommender \u2014 a classifier trained on five years of daily "
    "data for twenty US equities across nine sectors, predicting Buy, Hold or "
    "Avoid from "
    "nine technical indicators."
)
para(
    "3. Suitability Engine \u2014 a deterministic 5\u00d73 mapping from risk "
    "category and stock signal to a Suitable, Use Caution or Not Suitable "
    "rating, each with a written justification."
)
para(
    "4. Walk-Forward Backtester \u2014 temporal validation measuring how the "
    "advisor's recommendations would have performed historically."
)
para(
    "5. Conversational Interface \u2014 a chatbot answering questions about the "
    "recommendations, with a filter that rejects language-model output "
    "contradicting the underlying data."
)

doc.add_heading("1.2 Motivation", level=2)
para(
    "Retail investors increasingly trade without professional advice through "
    "platforms such as Robinhood, Trading 212 and eToro. Barber and Odean "
    "(2013) document that individual investors systematically underperform, "
    "partly through behavioural bias and partly through mismatches between "
    "their risk tolerance and their holdings. Professional advice mitigates "
    "this but is economically inaccessible to small portfolios. This project "
    "examines whether an automated, transparent system can deliver a basic "
    "suitability assessment, and \u2014 importantly \u2014 whether such a "
    "system genuinely adds value when evaluated rigorously."
)

doc.add_heading("1.3 Aim and Objectives", level=2)
para(
    "Aim: to develop and rigorously evaluate an AI financial advisor bot that "
    "produces personalised stock suitability recommendations, integrating "
    "scenario-based risk profiling with machine learning prediction, and to "
    "establish through leakage-free evaluation and statistical inference "
    "whether its advice has measurable value.",
    bold=True,
)
table(
    ["#", "Objective", "Deliverable"],
    [
        ["O1", "Scenario-based risk questionnaire with five categories",
         "src/risk_questionnaire.py"],
        ["O2", "Technical indicator feature engineering",
         "src/features.py"],
        ["O3", "Train and compare classifiers under leakage-free evaluation",
         "src/evaluation.py, retrain_model.py"],
        ["O4", "Suitability engine mapping profiles to signals",
         "src/suitability.py"],
        ["O5", "Web application for non-technical users",
         "Flask app, HTML/CSS/JS frontend"],
        ["O6", "Walk-forward backtesting of recommendation quality",
         "src/backtester.py"],
        ["O7", "Statistical inference on all reported results",
         "src/statistics_tests.py"],
        ["O8", "Validate the questionnaire, matrix and LLM filter",
         "src/instrument_validation.py, src/llm_evaluation.py"],
    ],
)

doc.add_heading("1.4 Justification", level=2)
para(
    "O1 establishes personalisation; O2 and O3 provide the predictive "
    "component; O4 converts both into actionable advice; O5 makes the system "
    "usable; O6 tests whether the advice would have worked; O7 establishes "
    "whether the measured differences are distinguishable from chance; and O8 "
    "validates the components that interact with users. O7 and O8 were added "
    "in response to marker feedback that the work needed a stronger evidential "
    "basis."
)

# ============================== CH 2 ==============================
doc.add_page_break()
doc.add_heading("CHAPTER 2: LITERATURE REVIEW", level=1)

doc.add_heading("2.1 Market Efficiency and the Limits of Prediction", level=2)
para(
    "Any claim to predict returns must engage with the efficient market "
    "hypothesis. Fama (1970) argues that prices already reflect available "
    "information, so in the semi-strong form no strategy built from public data "
    "\u2014 which is precisely what technical indicators are \u2014 should earn "
    "abnormal risk-adjusted returns. Malkiel (2003) reviews the evidence three "
    "decades on and concedes that documented anomalies exist, but maintains that "
    "most vanish once transaction costs and the incentives of arbitrageurs are "
    "accounted for."
)
para(
    "This framing matters for how the present project's results should be read. "
    "If markets are close to efficient, a weak out-of-sample edge is the expected "
    "outcome rather than a failure of implementation, and a large reported edge is "
    "grounds for suspecting a methodological error. That expectation shaped the "
    "decision to test for leakage (Section 4.3) rather than accept an unusually "
    "high accuracy at face value, and it explains why this review treats reported "
    "accuracies in the applied literature sceptically. It also justifies including "
    "transaction costs in the backtester, since Malkiel's central objection is "
    "that gross returns overstate what is achievable."
)

doc.add_heading("2.2 Classical Machine Learning for Stock Prediction", level=2)
para(
    "Random Forest, the model originally selected for this project, was "
    "introduced by Breiman (2001). It aggregates decorrelated decision trees "
    "grown on bootstrap samples with random feature subsets, so variance falls "
    "without a corresponding rise in bias. Its appeal for indicator-based "
    "prediction is that it captures interactions between features without their "
    "being specified in advance, tolerates differently scaled inputs and is "
    "resistant to overfitting relative to a single deep tree. Gradient-boosted "
    "variants such as XGBoost (Chen and Guestrin, 2016) typically improve on it "
    "for tabular data and dominate applied competitions."
)
para(
    "Patel et al. (2015) compared ANN, SVM, Random Forest and Naive Bayes on "
    "Indian indices, reporting Random Forest highest at 86.69% accuracy from "
    "technical indicators. The result is widely cited but must be read "
    "cautiously: the paper does not describe purging label overlap, and this "
    "project's own experiments show such accuracies collapse under leakage-free "
    "evaluation (Section 4.3). The methodological lesson proved more valuable "
    "here than the reported figure."
)
para(
    "Basak et al. (2019) found tree ensembles outperform single classifiers and "
    "noted that prediction is rarely connected to investor suitability, the gap "
    "this project's Suitability Engine addresses. Nti et al. (2020) reviewed 122 "
    "studies and established that technical indicators dominate feature "
    "selection and that shorter horizons yield better accuracy, which motivated "
    "the 30-day labelling window used here. Gu et al. (2020) provide the most "
    "careful large-scale assessment, comparing machine learning methods for "
    "asset pricing on strictly out-of-sample data and finding real but modest "
    "gains, with the improvement concentrated in flexible models' ability to "
    "capture interactions. Their scale of improvement, not the headline figures "
    "of smaller studies, is the realistic benchmark for this work."
)

doc.add_heading("2.3 Sequence Models: LSTM and Its Successors", level=2)
para(
    "The decisive limitation of tree ensembles and logistic regression for this "
    "task is architectural: both treat each row as an independent observation. A "
    "row containing today's RSI and moving averages summarises history through "
    "hand-chosen aggregates, but the model cannot see the order in which prices "
    "arrived. Two stocks with identical indicator values may have reached them by "
    "opposite paths, and a row-wise model cannot distinguish them."
)
para(
    "Recurrent networks address this by maintaining a hidden state updated at each "
    "time step, but simple recurrent units cannot learn long-range dependencies "
    "because gradients propagated through many steps vanish or explode. "
    "Hochreiter and Schmidhuber (1997) solved this with the long short-term "
    "memory unit, which adds a cell state carried forward largely unchanged and "
    "three learned gates: an input gate controlling what enters the cell, a forget "
    "gate controlling what is discarded, and an output gate controlling what is "
    "exposed to the next layer. Because the cell state passes through additive "
    "rather than repeated multiplicative updates, gradients survive over far "
    "longer sequences. Cho et al. (2014) later showed a simplified gated "
    "recurrent unit achieves comparable results with fewer parameters, which "
    "matters when training data is limited."
)
para(
    "In finance specifically, Fischer and Krauss (2018) applied LSTMs to all S&P "
    "500 constituents from 1992 to 2015 and reported directional accuracy "
    "exceeding random forests, deep feedforward networks and logistic regression, "
    "with the advantage strongest in the more volatile earlier part of their "
    "sample. Sezer et al. (2020) survey the field and confirm LSTM variants are "
    "the most frequently applied deep architecture to financial time series, "
    "while also noting that experimental protocols vary widely enough to make "
    "cross-study comparison unreliable."
)
para(
    "More recent work is markedly more cautious. Siami-Namini et al. (2019) found "
    "LSTM and bidirectional LSTM advantages depend heavily on the characteristics "
    "of the series. Vaswani et al. (2017) introduced the Transformer, whose "
    "self-attention removes the sequential bottleneck, and Lim et al. (2021) "
    "adapted it as the Temporal Fusion Transformer with interpretable attention "
    "for multi-horizon forecasting. Yet Zeng et al. (2023) demonstrated that "
    "strikingly simple linear models match or beat these architectures on many "
    "standard benchmarks, arguing that permutation-invariant attention is a poor "
    "fit for ordered data and that reported gains often reflect experimental "
    "design rather than modelling power."
)
para(
    "This project therefore implements an LSTM baseline rather than assuming "
    "either that sequence modelling is necessary or that it is superfluous. "
    "Section 4.4 reports that it achieves the best macro F1 but a significantly "
    "worse error rate than logistic regression, a split outcome consistent with "
    "Zeng et al. (2023) that locates the binding constraint in the feature set "
    "rather than the model class.",
    bold=True,
)

doc.add_heading("2.4 Why Time Series Demands Different Treatment", level=2)
para(
    "The second marker's observation that time series is a specific type of data "
    "identifies the deeper problem with applying either Random Forest or logistic "
    "regression here. Both are fitted under an assumption of independent, "
    "identically distributed observations drawn from a stable joint distribution. "
    "Financial series violate both parts of this."
)
para(
    "Observations are not independent: returns exhibit autocorrelation and "
    "volatility clusters, so adjacent rows carry overlapping information. Nor is "
    "the distribution stable. A series is stationary when its mean, variance and "
    "autocovariance do not depend on time, and price levels plainly are not: they "
    "trend. Dickey and Fuller (1979) developed the standard test for a unit root, "
    "with non-stationarity as the null hypothesis, while Kwiatkowski et al. (1992) "
    "constructed a complementary test whose null is stationarity. Because the two "
    "are opposed, agreement between them is stronger evidence than either alone, "
    "which is why both are applied in Section 4.5."
)
para(
    "Non-stationarity is particularly damaging to tree-based models. A decision "
    "tree partitions the input space using thresholds learned from training data "
    "and predicts a constant within each leaf, so it cannot extrapolate beyond "
    "the range it has seen. When a feature is an absolute price that has risen "
    "outside the training range, every future observation falls into the same "
    "boundary leaf and the model's effective discrimination collapses. Linear "
    "models extrapolate, though to an unreliable degree. The standard remedy is to "
    "transform features into stationary form \u2014 differencing, or expressing "
    "levels as ratios \u2014 and Section 4.5 measures which of this project's "
    "features require it. This is the concrete justification for the model choice "
    "that earlier versions of this report asserted without evidence.",
    bold=True,
)

doc.add_heading("2.5 Data Leakage and Evaluation Methodology", level=2)
para(
    "De Prado (2018) argues that most published financial machine learning is "
    "compromised by leakage, and prescribes purging and embargoing: removing "
    "training observations whose label horizon overlaps the test period, and "
    "additionally discarding a buffer immediately after it to account for "
    "serial correlation. Because labels here are 30-day forward returns, adjacent "
    "rows share overlapping futures, so a random split lets training labels "
    "encode test outcomes. He further argues that cross-validation as ordinarily "
    "applied is invalid for financial data for exactly this reason, and that a "
    "walk-forward protocol respecting the arrow of time is the minimum acceptable "
    "standard \u2014 the protocol adopted for the backtester in Section 4.6."
)
para(
    "Kaufman et al. (2012) formalise leakage as the introduction of "
    "information unavailable at prediction time, and Kapoor and Narayanan "
    "(2023) surveyed 294 papers across seventeen fields, finding leakage "
    "pervasive and reproducibly inflating reported performance. Their "
    "diagnosis matches this project's own experience exactly: a random split "
    "produced weighted F1 of 0.819 where a purged chronological split gives "
    "0.368 (Section 4.3)."
)

doc.add_heading("2.6 Statistical Comparison of Models and Strategies", level=2)
para(
    "Reporting two accuracies and declaring the larger better is not an inference. "
    "Dietterich (1998) established that comparing classifiers on a shared test set "
    "requires a paired test, because when both models see identical data their "
    "errors are correlated and an unpaired comparison overstates significance. He "
    "recommends McNemar's test (McNemar, 1947), which examines only the discordant "
    "cases \u2014 instances one model classifies correctly and the other does not "
    "\u2014 and asks whether the split between them departs from an even division. "
    "Demsar (2006) extended the problem to many models over many datasets, showing "
    "that repeated pairwise testing inflates the family-wise error rate and "
    "recommending rank-based procedures with explicit correction."
)
para(
    "The second marker's point that ANOVA is standard in investment curve analysis "
    "identifies a genuine omission from the earlier version of this review. "
    "Analysis of variance, originating with Fisher (1925), tests whether the means "
    "of three or more groups are equal by decomposing total variability into "
    "between-group and within-group components; the F statistic is their ratio. "
    "It is the correct tool for asking whether performance differs across market "
    "regimes, since running three separate t-tests would inflate the error rate. "
    "ANOVA assumes approximately normal residuals and equal variances, and where "
    "the response is a binary correct/incorrect indicator, as it is here, those "
    "assumptions are not met. The rank-based Kruskal-Wallis test (Kruskal and "
    "Wallis, 1952) makes no distributional assumption and is therefore the more "
    "defensible of the two, though reporting both and observing agreement is "
    "stronger than reporting either. A significant omnibus result establishes only "
    "that some difference exists, so Tukey's honestly significant difference "
    "procedure (Tukey, 1949) is used to identify which pairs differ while "
    "controlling the family-wise error rate. All three appear in Section 4.7.",
    bold=True,
)
para(
    "Strategy evaluation raises separate problems. Lo (2002) derived the standard "
    "error of the Sharpe ratio and showed that estimates from short samples are "
    "highly uncertain and additionally biased when returns are serially "
    "correlated, which is directly relevant here, where most per-ticker Sharpe "
    "intervals include zero. Bailey and Lopez de Prado (2014) introduced the "
    "deflated Sharpe ratio to correct for the selection bias that arises when many "
    "strategy variants are tried and the best reported. Harvey and Liu (2015) "
    "argue conventional thresholds are far too lenient for financial strategies "
    "given the number of hypotheses typically tested, and propose substantially "
    "higher bars for claiming a genuine effect."
)
para(
    "For proportions such as accuracy and win rate, Wilson (1927) intervals are "
    "used in preference to the normal approximation, which loses nominal coverage "
    "and can produce bounds outside [0, 1] at the small sample sizes this "
    "project's backtest produces. Interval width is treated as a result in its own "
    "right rather than a technicality, since it is what distinguishes an "
    "informative measurement from an anecdote."
)

doc.add_heading("2.7 Risk Profiling and Suitability", level=2)
para(
    "Grable and Lytton (1999) developed the standard 13-item risk tolerance "
    "instrument, isolating loss reaction, investment horizon and financial "
    "knowledge as the dimensions that discriminate between investors. Their "
    "validation rests on Cronbach's alpha (Cronbach, 1951), which estimates "
    "internal consistency as the extent to which items correlate with one another "
    "and so plausibly measure a single underlying trait; Nunnally and Bernstein "
    "(1994) established 0.70 as the conventional acceptability threshold. The "
    "important methodological point is that a questionnaire is an instrument whose "
    "reliability is measurable, not a design detail to be asserted, which is why "
    "Section 4.8 computes alpha rather than simply presenting the items. "
    "Kannadhasan (2015) confirmed demographic predictors in emerging markets, "
    "with age negatively and experience positively associated with tolerance."
)
para(
    "More recent work examines automated advice specifically. D'Acunto and "
    "Rossi (2021) review robo-advisory, noting that algorithmic advice improves "
    "diversification but that suitability logic is rarely validated "
    "empirically. FINRA Rule 2111 requires a reasonable basis for believing a "
    "recommendation suits the customer; this project operationalises that "
    "obligation as an auditable deterministic mapping rather than opaque model "
    "output."
)

doc.add_heading("2.8 Language Models and Hallucination Control", level=2)
para(
    "Ji et al. (2023) survey hallucination in natural language generation and draw "
    "the distinction that governs this project's design: intrinsic hallucination "
    "contradicts the provided source, whereas extrinsic hallucination adds content "
    "that cannot be verified against it. Intrinsic hallucination is the dangerous "
    "case in a financial advisory setting, because a summary that misattributes a "
    "suitability rating is not merely unhelpful but actively misleading, and is "
    "detectable precisely because the source is available."
)
para(
    "Huang et al. (2025) survey mitigation strategies and group them into training "
    "interventions, decoding controls and post-generation verification, concluding "
    "that verification against a trusted source is among the more dependable "
    "defences and is disproportionately valuable for small models, which "
    "hallucinate more readily. Retrieval-augmented generation (Lewis et al., 2020) "
    "attacks the same problem earlier in the pipeline by grounding generation in "
    "retrieved evidence, but grounding reduces rather than eliminates "
    "contradiction, so an output check remains necessary."
)
para(
    "This shapes the architecture directly. The chatbot composes its factual answer "
    "deterministically from model output and uses the language model only to "
    "rephrase, then verifies the rephrasing against the structured data and "
    "discards it on contradiction. The design responds to an observed failure "
    "rather than a hypothetical one: llama3.2:1b attributed one stock's suitability "
    "rating to another during testing. Crucially, Section 4.8 measures the filter's "
    "own precision and recall instead of assuming a safeguard works because it "
    "exists \u2014 a filter is itself a classifier and carries its own error rates.",
    bold=True,
)

doc.add_heading("2.9 Critical Synthesis", level=2)
para(
    "Five gaps emerge, and each maps onto a specific piece of work in Chapter 4."
)
para(
    "First, the prediction literature optimises accuracy without connecting it to "
    "investor suitability, so a signal is produced but never matched to a person. "
    "Second, and more seriously, much of that literature is methodologically "
    "compromised: Kapoor and Narayanan (2023) find leakage pervasive across "
    "seventeen fields, and this project reproduced exactly that inflation on its "
    "own data, turning a reported 0.84 into 0.37. Third, few systems test whether "
    "following the advice would have generated value, as distinct from whether the "
    "classifier is accurate; this project shows the two diverge sharply, with a "
    "better-than-chance win rate coexisting with significant underperformance "
    "against buy-and-hold. Fourth, model choice is usually justified by convention "
    "or by cited precedent rather than by the properties of the data, and the "
    "stationarity literature supplies a test that is rarely applied. Fifth, the "
    "human-facing components of robo-advisory systems are rarely validated at all; "
    "D'Acunto and Rossi (2021) note this explicitly, and it motivated the "
    "reliability, monotonicity and filter analyses in Section 4.8."
)
para(
    "Taken together these define the contribution attempted here. The novelty is "
    "not a new architecture but the combination of investor profiling with signal "
    "prediction under an evaluation regime strict enough that a negative result "
    "can be trusted \u2014 which, given the efficiency arguments in Section 2.1, "
    "is the result that should be expected.",
    bold=True,
)

# ============================== CH 3 ==============================
doc.add_page_break()
doc.add_heading("CHAPTER 3: PROJECT DESIGN", level=1)

doc.add_heading("3.1 Domain and Users", level=2)
para("Domain: personal finance and retail investment decision support.", bold=True)
para("Target users:", bold=True)
para(
    "\u2022 Self-directed retail investors using consumer trading platforms\n"
    "\u2022 Basic equity knowledge, limited technical analysis expertise\n"
    "\u2022 Seeking guidance calibrated to their own risk tolerance\n"
    "\u2022 Valuing transparency: why a recommendation was made, not only what\n"
    "\u2022 Accessing the system by web browser before making trading decisions"
)
para("User stories:", bold=True)
para(
    "\u2022 As a cautious investor I want to answer a few questions about my "
    "comfort with risk and then see which stocks suit me, so I can invest "
    "without taking on exposure I would not tolerate.\n"
    "\u2022 As a growth-oriented investor I want to open a recommended stock and "
    "see its chart, key indicators and the reasoning behind its rating, so I "
    "can judge the advice rather than merely accept it.\n"
    "\u2022 As a sceptical user I want evidence of how the advisor's "
    "recommendations performed historically, including where they failed, so I "
    "can calibrate my trust in the system."
)

doc.add_heading("3.2 System Architecture", level=2)
para("Offline training pipeline:", bold=True)
para(
    "yfinance \u2192 OHLCV data \u2192 nine technical indicators \u2192 purged "
    "chronological split \u2192 train candidate models \u2192 evaluate against "
    "naive baselines \u2192 export the selected model"
)
para("Online inference pipeline:", bold=True)
para(
    "questionnaire responses \u2192 risk category \u2192 download recent data "
    "for the stock universe \u2192 compute features \u2192 predict signal per "
    "stock \u2192 suitability engine \u2192 ranked recommendations with charts "
    "and explanations"
)
para("Evaluation pipeline:", bold=True)
para(
    "historical data \u2192 rolling train/test windows \u2192 retrain per window "
    "\u2192 simulate trading with transaction costs \u2192 compare against "
    "buy-and-hold \u2192 interval estimates and hypothesis tests"
)
para("Conversational layer:", bold=True)
para(
    "user question \u2192 intent classification \u2192 deterministic answer from "
    "model output \u2192 optional LLM rephrasing \u2192 contradiction filter "
    "\u2192 response displayed above the structured data"
)

doc.add_heading("3.3 Technology Choices", level=2)
table(
    ["Technology", "Purpose", "Justification"],
    [
        ["Python 3.13", "Implementation language",
         "Standard for data science; mature ecosystem"],
        ["scikit-learn", "Classical models",
         "Consistent API across LR, DT, RF and baselines"],
        ["PyTorch", "LSTM baseline",
         "Required to test whether sequence modelling helps"],
        ["statsmodels", "Stationarity tests",
         "Reference implementations of ADF and KPSS"],
        ["SciPy", "Statistical inference",
         "McNemar, ANOVA, Kruskal-Wallis, Tukey HSD"],
        ["Flask", "Web framework",
         "Lightweight; full control of the frontend"],
        ["Chart.js", "Price charts",
         "Client-side rendering, no server dependency"],
        ["Ollama (llama3.2:1b)", "Response phrasing",
         "Runs locally; no data leaves the machine"],
        ["Hypothesis", "Property-based testing",
         "Verifies invariants across generated inputs"],
        ["yfinance", "Market data",
         "Free, no API key, adequate for academic use"],
    ],
)

doc.add_heading("3.4 Key Design Decisions", level=2)
para(
    "1. Purged chronological evaluation rather than a random split. Adjacent "
    "rows share rolling features and overlapping label horizons, so a random "
    "split leaks future information. Section 4.3 quantifies the effect."
)
para(
    "2. Every metric reported against naive baselines. Weighted F1 on an "
    "imbalanced three-class target is uninterpretable alone; majority-class and "
    "random baselines establish the floor."
)
para(
    "3. Interval estimates and paired hypothesis tests, not point estimates. "
    "With thirteen trades a win rate carries almost no information, and only "
    "an interval makes that visible."
)
para(
    "4. Rule-based suitability rather than a learned mapping. Transparency is a "
    "requirement for advice, and a deterministic table is auditable and "
    "exhaustively testable. Section 4.6 validates its ordering properties."
)
para(
    "5. Scenario-based questionnaire rather than a single slider. Five "
    "behavioural items support a reliability analysis; one slider does not."
)
para(
    "6. The language model rephrases but never originates content, and its "
    "output is verified against the structured data before display."
)

doc.add_heading("3.5 Workplan", level=2)
table(
    ["Week", "Tasks", "Milestone"],
    [
        ["1-2", "Project setup, validation, risk questionnaire", "Core modules"],
        ["3-4", "Suitability engine, feature engineering, property tests",
         "Deterministic components tested"],
        ["5-6", "Data acquisition, model training", "Models trained"],
        ["7-8", "Backtester, Flask application, chatbot", "Prototype working"],
        ["9", "Leakage diagnosis and correction; retraining",
         "Evaluation methodology corrected"],
        ["10", "Statistical inference layer; regime analysis",
         "Results carry intervals and tests"],
        ["11", "LSTM baseline, stationarity diagnostics",
         "Model choice evidenced"],
        ["12", "Instrument and LLM filter validation",
         "Human-facing components validated"],
        ["13-14", "User evaluation with real participants; SUS",
         "Usability data collected"],
        ["15-16", "Final report, video, submission", "Submission ready"],
    ],
)

doc.add_heading("3.6 Evaluation Strategy", level=2)
para(
    "Each metric is chosen for a stated reason rather than by convention, because "
    "on this problem different metrics support different conclusions."
)
para(
    "Accuracy is the proportion of correct predictions. It is reported because it "
    "is the most widely understood figure, but on a three-class target with "
    "roughly 39% Avoid, 39% Hold and 22% Buy it is misleading in isolation: a "
    "model can score well by never predicting the minority class. Weighted F1 "
    "averages the per-class harmonic mean of precision and recall in proportion "
    "to class frequency, so it still under-weights rare classes. Macro F1 weights "
    "every class equally and therefore penalises a model that ignores Buy, and "
    "balanced accuracy does the same for recall. Because the system exists to "
    "surface Buy candidates, macro F1 is treated as the metric of record and "
    "accuracy as a secondary reference. Every one of these is quoted against "
    "majority-class and random baselines computed on the same test set, since an "
    "absolute value carries no information about difficulty."
)
para(
    "The backtest metrics answer a different question. Win rate is the proportion "
    "of simulated trades closing at a profit and measures signal quality only "
    "among positions actually taken. Total return measures what a user following "
    "the advice would have earned, and is meaningful only against the "
    "buy-and-hold return on the same ticker over the same window. The Sharpe "
    "ratio divides excess return by its standard deviation and is reported with an "
    "interval because point estimates from short samples are unreliable (Lo, "
    "2002). Maximum drawdown records the worst peak-to-trough fall, which matters "
    "to a risk-averse user in a way that average return does not."
)
para("Every threshold below is justified rather than asserted.", bold=True)
table(
    ["Technique", "Target", "Criterion and justification"],
    [
        ["Purged chronological split", "All classifiers",
         "No training label may overlap the test period (De Prado, 2018)"],
        ["Naive baselines", "All classifiers",
         "Must exceed majority-class and random baselines measured on the "
         "same test set, not a nominal 0.5"],
        ["McNemar paired test", "Model comparison",
         "p < 0.05; paired because models share a test set (Dietterich, 1998)"],
        ["Wilson intervals", "Accuracy, win rate",
         "95% coverage; Wilson preferred to Wald at small n (Wilson, 1927)"],
        ["ANOVA and Kruskal-Wallis", "Regime comparison",
         "p < 0.05; rank test given precedence as returns are non-normal"],
        ["Sharpe ratio interval", "Strategy risk-adjusted return",
         "Interval must exclude zero to claim skill (Lo, 2002)"],
        ["Power analysis", "All proportion claims",
         "Claims are reported as indicative where n is below the required "
         "sample size"],
        ["Cronbach's alpha", "Questionnaire",
         "alpha >= 0.70, the conventional threshold (Nunnally and Bernstein, 1994)"],
        ["Monotonicity checks", "Suitability matrix",
         "Permissiveness must not decrease as risk tolerance or signal "
         "strength rises"],
        ["Filter recall", "LLM contradiction filter",
         "Recall on attributable contradictions must be 1.0; a false accept "
         "sends wrong advice to the user"],
        ["SUS questionnaire", "Interface usability",
         "Target above 68, the established average (Sauro and Lewis, 2016)"],
    ],
)

# ============================== CH 4 ==============================
doc.add_page_break()
doc.add_heading("CHAPTER 4: FEATURE PROTOTYPE AND EVALUATION", level=1)

doc.add_heading("4.1 Prototype Description", level=2)
para(
    "The prototype is a working Flask application implementing the architecture of "
    "Section 3.2. A user answers five scenario-based questions; the system "
    "classifies their risk profile, predicts a signal for each of twenty equities "
    "and returns suitability ratings sorted by relevance, each expanding to show a "
    "chart, indicators, strengths, risks and a justification. Separate pages "
    "provide comparison and backtesting."
)

doc.add_heading("4.2 Data and Experimental Setup", level=2)
table(
    ["Parameter", "Value"],
    [
        ["Data source", "Yahoo Finance via yfinance (Aroussi, 2024)"],
        ["Universe", "20 US equities across 9 GICS sectors: AAPL, MSFT, GOOGL, "
                     "META, NVDA, AMD, MU, AMZN, TSLA, HD, KO, PEP, WMT, JPM, "
                     "JNJ, UNH, XOM, CAT, DIS, NEE"],
        ["Period", "1 January 2020 to 31 December 2024"],
        ["Observations", "23,560 rows after feature engineering"],
        ["Features", "close, daily return, 5/20/50-day SMA, 20-day volatility, "
                     "volume, 14-day RSI, MACD"],
        ["Target", "30-day forward return: Buy >10%, Hold 0-10%, Avoid <0%"],
        ["Class balance", "Avoid 9,222 / Hold 9,171 / Buy 5,167"],
        ["Split", "Purged chronological, 20% test, 30-day purge"],
        ["Test block", "From 2023-12-08; 600 rows purged; "
                       "train n = 18,240, test n = 4,720"],
        ["Transaction cost", "0.1% per trade, applied on entry and exit"],
    ],
)
para(
    "The universe was widened from ten technology names to twenty across nine "
    "sectors, because a single-sector set makes generalisation untestable. Adding "
    "defensives and cyclicals means Sections 4.6 and 4.7 now span assets that "
    "behave differently across the cycle, and doubling the sample narrowed every "
    "interval below."
)

doc.add_heading("4.3 A Methodological Correction: Detecting Data Leakage", level=2)
para(
    "An earlier version of this project reported weighted F1 of 0.84 from a "
    "random stratified split and selected Random Forest on that basis. "
    "Investigating the gap between that figure and a walk-forward signal accuracy "
    "of 48.8% revealed the split itself was invalid. Both mechanisms described in "
    "Section 2.3 were present, and isolating them requires holding data and "
    "features fixed while varying only the split.",
    bold=True,
)
table(
    ["Splitting strategy", "Weighted F1", "Random baseline", "n test"],
    [
        ["Random stratified (originally reported)", "0.819", "0.346", "4,712"],
        ["Chronological", "0.387", "0.342", "4,712"],
        ["Chronological with purged label overlap", "0.368", "0.345", "4,712"],
    ],
)
para(
    "The reported 0.84 was therefore not an estimate of out-of-sample "
    "performance. Under correct evaluation the same model scores 0.368 against "
    "a random baseline of 0.345 \u2014 effectively no skill. Abandoning the "
    "random split costs 0.432 F1 and purging a further 0.019, so the first "
    "mechanism dominates. The effect reproduced when the universe was doubled "
    "(0.833 to 0.819 random, 0.356 to 0.368 purged), so it is not an artefact of "
    "one sample of stocks. This invalidated the earlier headline result and forced "
    "correction of the whole pipeline. Output: experiments/exp01_results.txt.",
    bold=True,
)

doc.add_heading("4.4 Model Comparison Under Correct Evaluation", level=2)
para(
    "All models were retrained on the purged split and scored against naive "
    "baselines on exactly the rows the LSTM can predict (n = 4,340), so the "
    "comparison is like-for-like. Accuracy carries 95% Wilson intervals. A "
    "decision tree scored lowest on every metric (F1 0.364)."
)
table(
    ["Model", "Accuracy [95% CI]", "Weighted F1", "Macro F1", "Balanced acc."],
    [
        ["Logistic Regression", "0.401 [0.386, 0.416]", "0.371", "0.336",
         "0.357"],
        ["Random Forest", "0.388 [0.373, 0.402]", "0.367", "0.343", "0.358"],
        ["LSTM (20-day sequences)", "0.384 [0.369, 0.398]", "0.381", "0.365",
         "0.362"],
        ["Baseline: majority class", "0.362 [0.347, 0.376]", "0.192", "0.177",
         "0.333"],
        ["Baseline: random", "0.350 [0.336, 0.365]", "0.350", "0.330", "0.331"],
    ],
)
para(
    "Paired McNemar tests on the shared test set, the appropriate comparison "
    "when models see identical data (Dietterich, 1998):"
)
table(
    ["Comparison", "p-value", "Conclusion"],
    [
        ["LSTM vs Logistic Regression", "0.017",
         "Logistic Regression has significantly lower error rate"],
        ["LSTM vs Random Forest", "0.702", "No significant difference"],
        ["LSTM vs random baseline", "0.001", "LSTM significantly better"],
        ["Logistic Regression vs Random Forest", "0.171",
         "No significant difference"],
        ["Logistic Regression vs Decision Tree", "0.003",
         "Logistic Regression significantly better"],
        ["Logistic Regression vs random baseline", "<0.001",
         "Logistic Regression significantly better"],
    ],
)
para(
    "The ranking depends on which metric is chosen, and that disagreement is "
    "the most informative result here. Logistic regression has the highest "
    "accuracy and McNemar confirms its error rate is significantly lower than "
    "the LSTM's (p = 0.017), yet the LSTM leads on weighted F1, macro F1 and "
    "balanced accuracy. The per-class breakdown explains why: logistic "
    "regression almost never predicts the minority Buy class, whose recall is "
    "0.07 against a precision of 0.54. Its accuracy comes from predicting the "
    "two frequent classes.",
    bold=True,
)
para(
    "For this application that matters more than the headline figure. A tool meant "
    "to surface Buy candidates that finds 7% of them fails at its actual task "
    "while scoring well on the most-quoted metric. This is why macro F1 and "
    "balanced accuracy are reported alongside accuracy: on an imbalanced target, "
    "accuracy rewards ignoring the class the user cares about."
)
para(
    "The original Random Forest selection was likewise an artefact of leakage: "
    "under valid evaluation it is indistinguishable from logistic regression "
    "(p = 0.171). The LSTM, despite modelling temporal order, beats only the "
    "random baseline, and the best model of any kind exceeds the strongest "
    "baseline by 0.031 F1. The ceiling is therefore the feature set and the "
    "difficulty of 30-day prediction rather than model capacity, consistent with "
    "Zeng et al. (2023). Logistic regression is retained for accuracy, "
    "interpretability and inference cost, with its Buy recall a known defect.",
    bold=True,
)

doc.add_heading("4.5 Why Tree Ensembles Struggle: Stationarity", level=2)
para(
    "Both production models assume a stable joint distribution. ADF and KPSS "
    "were applied to each feature; their nulls are opposed, so agreement is "
    "stronger evidence than either alone."
)
table(
    ["Feature", "ADF p", "KPSS p", "Verdict"],
    [
        ["close_price", "0.549", "0.010", "Non-stationary"],
        ["ma_5", "0.496", "0.010", "Non-stationary"],
        ["ma_20", "0.404", "0.010", "Non-stationary"],
        ["ma_50", "0.391", "0.010", "Non-stationary"],
        ["volatility", "0.000", "0.010", "Conflicting"],
        ["volume", "0.000", "0.010", "Conflicting"],
        ["daily_return", "0.000", "0.100", "Stationary"],
        ["RSI", "0.000", "0.100", "Stationary"],
        ["MACD", "0.000", "0.100", "Stationary"],
    ],
)
para(
    "Six of nine features are not clearly stationary, matching theory: price "
    "levels and moving averages trend, whereas differenced and bounded features "
    "are well behaved. A model fitted to 2021 price levels is extrapolating at "
    "2024 levels. This is a measured answer to why model choice matters for time "
    "series and identifies a specific fix: replacing absolute price features with "
    "relative ones such as price-to-moving-average ratios."
)

doc.add_heading("4.6 Backtesting: Does the Advice Add Value?", level=2)
para(
    "Classification accuracy and investment value are distinct. The backtester "
    "retrains on each rolling window (500-day train, 60-day test), simulates the "
    "signals with transaction costs and compares against buy-and-hold across all "
    "twenty tickers."
)
table(
    ["Metric", "Result", "Interpretation"],
    [
        ["Pooled win rate", "169/272 = 62.1% [56.2%, 67.7%]",
         "Excludes 50%, p < 0.001: distinguishable from chance"],
        ["Mean strategy return", "+6.46%",
         "Positive but highly dispersed across tickers"],
        ["Mean benchmark return", "+63.72%", "Buy-and-hold over the same period"],
        ["Mean excess return", "-57.25% (p = 0.008)",
         "Reliable underperformance, now significant at n = 20 tickers"],
        ["Sharpe ratios", "4 of 5 tested include zero",
         "Risk-adjusted return mostly indistinguishable from no skill"],
        ["Tickers with no trades", "1 of 20 (JNJ)",
         "Model never issued a Buy signal; strategy held cash and lost to a "
         "+4.3% benchmark"],
    ],
)
para(
    "Doubling the universe changed the status of the central claim. At ten tickers "
    "the mean excess of -74.5% was not significant (p = 0.078); at twenty the "
    "deficit is smaller but significant (p = 0.008), the added degrees of freedom "
    "outweighing the reduced effect size. Underperformance can now be asserted "
    "with evidence rather than merely observed.",
    bold=True,
)
para(
    "The zero-trade ticker is reported rather than dropped. An earlier version of "
    "this analysis silently excluded it, removing the one case where the strategy "
    "did nothing while the benchmark rose."
)
para(
    "Per-ticker rates show why pooling is necessary: the best was PEP at 100% "
    "from five trades, an interval 43 points wide, the worst HD at 16.7% from "
    "six. Power analysis settles it. The pooled 62.1% needs 126 trades and has "
    "272, whereas detecting a smaller but still valuable 55% edge would need 778. "
    "No single ticker comes close, so the pooled figure carries the claim."
)
para(
    "Three statements are simultaneously true, and only intervals make that "
    "visible: the bot picks profitable trades more often than chance among those "
    "it takes; its risk-adjusted return is generally indistinguishable from zero; "
    "and it significantly trails passive holding. The first concerns signal "
    "quality and the third investment value, and conflating them is the error "
    "this evaluation was restructured to avoid.",
    bold=True,
)

doc.add_heading("4.7 Regime Dependence", level=2)
para(
    "For each regime a fresh model was trained on all preceding data, keeping "
    "every evaluation out-of-sample. The earliest regime cannot be tested as no "
    "prior data exists."
)
table(
    ["Regime", "Train n", "Test n", "Accuracy [95% CI]"],
    [
        ["2021 bull market", "3,240", "5,040", "0.437 [0.424, 0.451]"],
        ["2022 bear market", "8,280", "5,020", "0.318 [0.305, 0.331]"],
        ["2023-24 recovery", "13,320", "9,420", "0.405 [0.395, 0.415]"],
    ],
)
para(
    "One-way ANOVA gives F = 83.35, p < 0.001 and Kruskal-Wallis p < 0.001; all "
    "three Tukey HSD pairs survive family-wise correction, the largest gap 0.119 "
    "between 2021 and 2022. Since per-row accuracy is binary rather than normal, "
    "the rank test takes precedence, with ANOVA reported alongside as the "
    "conventional expectation; the two agree."
)
para(
    "The substantive finding is that 2022 bear-market accuracy falls to 0.318, "
    "with an upper bound of 0.331 sitting below the 0.333 random baseline. The "
    "model performs worst precisely when avoiding losses matters most, and the "
    "wider universe confirms this is not specific to technology stocks. This is "
    "a material limitation for an advisory system and is consistent with the "
    "non-stationarity in Section 4.5.",
    bold=True,
)

doc.add_heading("4.8 Validating the Human-Facing Components", level=2)
para(
    "The questionnaire, suitability matrix and hallucination filter were "
    "validated rather than assumed correct; full output is in Appendices D and E."
)
para(
    "Questionnaire reliability, computed on simulated responses because real "
    "respondent data is not yet collected, gives Cronbach's alpha of 0.910 with "
    "all five item-total correlations between 0.750 and 0.796 against a 0.30 "
    "threshold. Shifting every band cut-off one point reclassifies 25% of "
    "attainable scores, so the bands are consequential and need empirical "
    "calibration."
)
para(
    "The matrix was validated structurally: all 15 combinations present with "
    "explanations, and permissiveness never decreasing as risk tolerance rises or "
    "the signal strengthens. All checks pass, establishing coherence but not "
    "agreement with professional judgement."
)
para(
    "The filter was added after llama3.2:1b reported META as \u201cnot "
    "suitable\u201d when that rating belonged to AAPL, an intrinsic hallucination "
    "in the sense of Ji et al. (2023). On 12 labelled cases recall and precision "
    "are both 1.000, but the recall interval of [0.566, 1.000] makes that "
    "indicative only. It adjudicates only sentences naming exactly one ticker, "
    "since a claim mentioning two or none cannot be attributed; those are "
    "accepted by design, mitigated by always showing the structured data.",
    bold=True,
)

doc.add_heading("4.9 Testing", level=2)
para(
    "The codebase carries 313 automated tests: Hypothesis property tests for the "
    "deterministic components, unit tests for the statistical machinery and "
    "integration tests for the pipeline. Statistical functions are checked "
    "against independently derived values, including Wilson intervals against "
    "the published formula."
)

doc.add_heading("4.10 Critical Evaluation", level=2)
para("What the evidence supports:", bold=True)
para(
    "\u2022 A working end-to-end system from questionnaire to explained "
    "recommendation\n"
    "\u2022 Leakage-free evaluation, every metric referenced to naive baselines\n"
    "\u2022 Model selection settled by paired significance tests\n"
    "\u2022 Signal quality and investment value measured separately, and shown to "
    "diverge\n"
    "\u2022 Quantified error rates for the hallucination filter"
)
para("Limitations, stated plainly:", bold=True)
para(
    "1. Predictive skill is weak: 0.401 accuracy against a 0.350 baseline.\n"
    "2. Buy recall is 0.07, so the model rarely finds the class the tool exists "
    "to surface.\n"
    "3. The strategy significantly trails buy-and-hold (-57.2%, p = 0.008) and 4 "
    "of 5 Sharpe intervals include zero.\n"
    "4. Accuracy falls below baseline in the 2022 bear market, when sound advice "
    "matters most.\n"
    "5. Six of nine features are non-stationary, violating a model assumption.\n"
    "6. Instrument reliability and the filter's error rates rest on simulated or "
    "small samples, and the matrix lacks expert review.\n"
    "7. Twenty US equities is better than ten but remains one market, one "
    "currency and one capitalisation band."
)

doc.add_heading("4.11 Planned Improvements", level=2)
para(
    "1. Replace absolute price features with stationary relative ones, targeting "
    "Section 4.5.\n"
    "2. Address Buy recall through class weighting or threshold tuning, then "
    "re-evaluate on macro F1.\n"
    "3. Add regime detection so the system widens its caution in conditions "
    "resembling 2022.\n"
    "4. Administer the questionnaire to 20-30 respondents and calibrate the bands; "
    "obtain expert review of the matrix.\n"
    "5. Expand the filter set to 100+ labelled summaries including adversarial "
    "phrasings.\n"
    "6. Show confidence scores so users see the model is often uncertain.\n"
    "7. Apply the deflated Sharpe ratio (Bailey and Lopez de Prado, 2014) for "
    "backtest selection bias."
)

# ============================== CH 5 ==============================
doc.add_page_break()
doc.add_heading("CHAPTER 5: APPENDICES", level=1)

para("Appendix A: Web Application Interface", bold=True)
para("Figure A1: Risk tolerance questionnaire", italic=True)
para("Figure A2: Recommendations with risk profile summary", italic=True)
para("Figure A3: Stock detail with chart and explanation", italic=True)
para("Figure A4: Stock flagged Not Suitable", italic=True)
para("Figure A5: Conversational assistant with AI summary", italic=True)
doc.add_paragraph()

para("Appendix B: Backtesting Evidence", bold=True)
para("Figure B1: Backtest configuration and aggregate metrics", italic=True)
para("Figure B2: Per-ticker results with confidence intervals", italic=True)
doc.add_paragraph()

para("Appendix C: Test Evidence", bold=True)
para("Figure C1: 313 automated tests passing", italic=True)
doc.add_paragraph()

para("Appendix D: Questionnaire Reliability Analysis", bold=True)
para(
    "Cronbach's alpha computed on 500 simulated response sets per coherence "
    "level. Latent coherence is the correlation between a respondent's "
    "underlying risk tolerance and their answer to each item; low coherence "
    "represents inconsistent responding. Source: "
    "experiments/exp05_instrument_and_llm.py."
)
table(
    ["Latent coherence", "Cronbach's alpha", "Interpretation"],
    [
        ["0.5", "0.759", "Acceptable"],
        ["0.7", "0.910", "Excellent"],
        ["0.9", "0.984", "Excellent"],
    ],
)
para(
    "Band sensitivity: a one-point shift in every cut-off reclassifies 25% of "
    "attainable total scores, a two-point shift 50%. Matrix checks: all 15 "
    "risk-category by signal combinations present with explanations; "
    "monotonicity in risk tolerance and in signal strength both hold."
)
doc.add_paragraph()

para("Appendix E: LLM Contradiction Filter Error Rates", bold=True)
para(
    "Measured on 12 labelled cases generated from real ratings and signals, so "
    "ground truth follows by construction. Source: "
    "experiments/exp05_instrument_and_llm.py."
)
table(
    ["Metric", "Value", "Meaning"],
    [
        ["Recall", "1.000 [0.566, 1.000]",
         "All attributable contradictions caught"],
        ["Precision", "1.000", "No sound summary discarded"],
        ["False accept rate", "0.000", "No contradiction reached the user"],
        ["False reject rate", "0.000", "No valid summary needlessly rejected"],
        ["Evaluation set", "12 cases (5 contradictory, 7 sound)",
         "Generated from real ratings and signals"],
    ],
)
doc.add_paragraph()

para("Appendix F: Reproducible Experiment Scripts", bold=True)
para(
    "All quantitative claims in Chapter 4 are reproducible from the following "
    "scripts, each of which writes its full output to the experiments directory:"
)
table(
    ["Script", "Produces", "Reported in"],
    [
        ["exp01_leakage_diagnostic.py", "Three-way split comparison", "4.3"],
        ["exp02 (retrain_model.py)", "Model comparison with baselines", "4.4"],
        ["exp03_statistical_analysis.py",
         "Intervals, McNemar, ANOVA, power", "4.4, 4.6, 4.7"],
        ["exp04_model_justification.py",
         "Stationarity tests and LSTM baseline", "4.4, 4.5"],
        ["exp05_instrument_and_llm.py",
         "Reliability, band sensitivity, matrix, filter", "4.8"],
    ],
)

# ============================== CH 6 ==============================
doc.add_heading("CHAPTER 6: REFERENCES", level=1)
references = [
    "Achelis, S.B. (2001) Technical Analysis from A to Z. 2nd edn. New York: "
    "McGraw-Hill.",
    "Aroussi, R. (2024) yfinance: download market data from Yahoo! Finance API. "
    "Available at: https://github.com/ranaroussi/yfinance (Accessed: 10 June 2025).",
    "Bailey, D.H. and Lopez de Prado, M. (2014) 'The deflated Sharpe ratio: "
    "correcting for selection bias, backtest overfitting and non-normality', "
    "Journal of Portfolio Management, 40(5), pp. 94-107.",
    "Barber, B.M. and Odean, T. (2013) 'The behavior of individual investors', "
    "in Handbook of the Economics of Finance, Vol. 2. Amsterdam: Elsevier, "
    "pp. 1533-1570.",
    "Basak, S., Kar, S., Saha, S., Khaidem, L. and Dey, S.R. (2019) 'Predicting "
    "the direction of stock market prices using tree-based ensemble learning', "
    "North American Journal of Economics and Finance, 47, pp. 552-567.",
    "Breiman, L. (2001) 'Random forests', Machine Learning, 45(1), pp. 5-32.",
    "Chen, T. and Guestrin, C. (2016) 'XGBoost: a scalable tree boosting system', "
    "in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
    "Discovery and Data Mining. New York: ACM, pp. 785-794.",
    "Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., "
    "Schwenk, H. and Bengio, Y. (2014) 'Learning phrase representations using RNN "
    "encoder-decoder for statistical machine translation', in Proceedings of the "
    "2014 Conference on Empirical Methods in Natural Language Processing. Doha: "
    "ACL, pp. 1724-1734.",
    "Cronbach, L.J. (1951) 'Coefficient alpha and the internal structure of "
    "tests', Psychometrika, 16(3), pp. 297-334.",
    "D'Acunto, F. and Rossi, A.G. (2021) 'Robo-advising', in The Palgrave "
    "Handbook of Technological Finance. Cham: Palgrave Macmillan, pp. 725-749.",
    "Demsar, J. (2006) 'Statistical comparisons of classifiers over multiple "
    "data sets', Journal of Machine Learning Research, 7, pp. 1-30.",
    "De Prado, M.L. (2018) Advances in Financial Machine Learning. Hoboken: "
    "Wiley.",
    "Dickey, D.A. and Fuller, W.A. (1979) 'Distribution of the estimators for "
    "autoregressive time series with a unit root', Journal of the American "
    "Statistical Association, 74(366), pp. 427-431.",
    "Dietterich, T.G. (1998) 'Approximate statistical tests for comparing "
    "supervised classification learning algorithms', Neural Computation, 10(7), "
    "pp. 1895-1923.",
    "Fama, E.F. (1970) 'Efficient capital markets: a review of theory and "
    "empirical work', Journal of Finance, 25(2), pp. 383-417.",
    "FINRA (2014) Rule 2111: Suitability. Financial Industry Regulatory "
    "Authority. Available at: https://www.finra.org/rules-guidance/rulebooks/"
    "finra-rules/2111 (Accessed: 10 June 2025).",
    "Fischer, T. and Krauss, C. (2018) 'Deep learning with long short-term "
    "memory networks for financial market predictions', European Journal of "
    "Operational Research, 270(2), pp. 654-669.",
    "Fisher, R.A. (1925) Statistical Methods for Research Workers. Edinburgh: "
    "Oliver and Boyd.",
    "Grable, J.E. and Lytton, R.H. (1999) 'Financial risk tolerance revisited: "
    "the development of a risk assessment instrument', Financial Services "
    "Review, 8(3), pp. 163-181.",
    "Gu, S., Kelly, B. and Xiu, D. (2020) 'Empirical asset pricing via machine "
    "learning', Review of Financial Studies, 33(5), pp. 2223-2273.",
    "Harvey, C.R. and Liu, Y. (2015) 'Backtesting', Journal of Portfolio "
    "Management, 42(1), pp. 13-28.",
    "Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory', "
    "Neural Computation, 9(8), pp. 1735-1780.",
    "Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, "
    "W., Feng, X., Qin, B. and Liu, T. (2025) 'A survey on hallucination in "
    "large language models', ACM Transactions on Information Systems, 43(2), "
    "pp. 1-55.",
    "Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., "
    "Madotto, A. and Fung, P. (2023) 'Survey of hallucination in natural "
    "language generation', ACM Computing Surveys, 55(12), pp. 1-38.",
    "Kannadhasan, M. (2015) 'Retail investors' financial risk tolerance and "
    "their risk-taking behaviour', IIMB Management Review, 27(3), pp. 175-184.",
    "Kapoor, S. and Narayanan, A. (2023) 'Leakage and the reproducibility "
    "crisis in machine-learning-based science', Patterns, 4(9), 100804.",
    "Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012) 'Leakage in "
    "data mining: formulation, detection, and avoidance', ACM Transactions on "
    "Knowledge Discovery from Data, 6(4), pp. 1-21.",
    "Kruskal, W.H. and Wallis, W.A. (1952) 'Use of ranks in one-criterion "
    "variance analysis', Journal of the American Statistical Association, "
    "47(260), pp. 583-621.",
    "Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992) "
    "'Testing the null hypothesis of stationarity against the alternative of a "
    "unit root', Journal of Econometrics, 54(1-3), pp. 159-178.",
    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., "
    "Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S. and Kiela, D. "
    "(2020) 'Retrieval-augmented generation for knowledge-intensive NLP tasks', "
    "in Advances in Neural Information Processing Systems 33, pp. 9459-9474.",
    "Lim, B., Arik, S.O., Loeff, N. and Pfister, T. (2021) 'Temporal fusion "
    "transformers for interpretable multi-horizon time series forecasting', "
    "International Journal of Forecasting, 37(4), pp. 1748-1764.",
    "Lo, A.W. (2002) 'The statistics of Sharpe ratios', Financial Analysts "
    "Journal, 58(4), pp. 36-52.",
    "Malkiel, B.G. (2003) 'The efficient market hypothesis and its critics', "
    "Journal of Economic Perspectives, 17(1), pp. 59-82.",
    "McNemar, Q. (1947) 'Note on the sampling error of the difference between "
    "correlated proportions or percentages', Psychometrika, 12(2), pp. 153-157.",
    "Nti, I.K., Adekoya, A.F. and Weyori, B.A. (2020) 'A systematic review of "
    "fundamental and technical analysis of stock market predictions', "
    "Artificial Intelligence Review, 53, pp. 3007-3057.",
    "Nunnally, J.C. and Bernstein, I.H. (1994) Psychometric Theory. 3rd edn. "
    "New York: McGraw-Hill.",
    "Patel, J., Shah, S., Thakkar, P. and Kotecha, K. (2015) 'Predicting stock "
    "and stock price index movement using trend deterministic data preparation "
    "and machine learning techniques', Expert Systems with Applications, 42(1), "
    "pp. 259-268.",
    "Sauro, J. and Lewis, J.R. (2016) Quantifying the User Experience: "
    "Practical Statistics for User Research. 2nd edn. Cambridge, MA: Morgan "
    "Kaufmann.",
    "Sezer, O.B., Gudelek, M.U. and Ozbayoglu, A.M. (2020) 'Financial time "
    "series forecasting with deep learning: a systematic literature review, "
    "2005-2019', Applied Soft Computing, 90, 106181.",
    "Siami-Namini, S., Tavakoli, N. and Siami Namin, A. (2019) 'The performance "
    "of LSTM and BiLSTM in forecasting time series', in IEEE International "
    "Conference on Big Data. Los Angeles: IEEE, pp. 3285-3292.",
    "Tukey, J.W. (1949) 'Comparing individual means in the analysis of variance', "
    "Biometrics, 5(2), pp. 99-114.",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, "
    "A.N., Kaiser, L. and Polosukhin, I. (2017) 'Attention is all you need', in "
    "Advances in Neural Information Processing Systems 30, pp. 5998-6008.",
    "Wilson, E.B. (1927) 'Probable inference, the law of succession, and "
    "statistical inference', Journal of the American Statistical Association, "
    "22(158), pp. 209-212.",
    "Zeng, A., Chen, M., Zhang, L. and Xu, Q. (2023) 'Are transformers "
    "effective for time series forecasting?', in Proceedings of the AAAI "
    "Conference on Artificial Intelligence, 37(9), pp. 11121-11128.",
]
for reference in references:
    para(reference, after=Pt(3))

# ============================== SAVE ==============================
output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "Preliminary_Project_Report_v4.docx")
doc.save(output)
print(f"Report saved to: {output}")
print(f"References: {len(references)}")
