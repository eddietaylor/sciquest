from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import base64
import mimetypes

from .io import read_yaml


def _logo_html(quest_path: Path) -> str:
    candidates = [
        quest_path / "artifacts" / "logo.png",
        quest_path / "artifacts" / "logo.jpg",
        quest_path / "artifacts" / "logo.jpeg",
        Path("/home/edtaylor/Downloads/ChatGPT Image Apr 26, 2026, 09_35_07 PM.png"),
    ]
    logo = next((p for p in candidates if p.exists() and p.is_file()), None)
    if not logo:
        return '<div class="logo-fallback">SciQuest</div>'
    mime = mimetypes.guess_type(str(logo))[0] or "image/png"
    data = base64.b64encode(logo.read_bytes()).decode("ascii")
    return f'<img class="sciquest-logo" src="data:{mime};base64,{data}" alt="SciQuest logo">'


def _list_items(items: list[Any]) -> str:
    if not items:
        return "<li>Not documented</li>"
    return "".join(f"<li>{escape(str(item))}</li>" for item in items)


def _metric_table(validation: dict[str, Any]) -> str:
    metrics = validation.get("metrics", {}) or {}
    scores = validation.get("scores", {}) or {}
    rows = []
    for name, raw in metrics.items():
        score = scores.get(name, {}) if isinstance(scores, dict) else {}
        rows.append(
            "<tr>"
            f"<td>{escape(str(name))}</td>"
            f"<td>{escape(str(raw))}</td>"
            f"<td>{escape(str(score.get('normalized', '')))}</td>"
            f"<td>{escape(str(score.get('weight', '')))}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="4">No validation metrics recorded.</td></tr>'


def _extract_interpretation(report_text: str) -> str:
    if not report_text:
        return "No interpretation report available."
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    evidence = []
    speculation = []
    target = None
    for line in lines:
        lower = line.lower()
        if lower.startswith("## evidence"):
            target = evidence
            continue
        if lower.startswith("## speculation"):
            target = speculation
            continue
        if lower.startswith("## "):
            target = None
            continue
        if target is not None and not line.startswith("#"):
            target.append(line)
    text = "\n".join(evidence[-2:] + speculation[:2]) or "\n".join(lines[:8])
    return text[:1600]


def _inline_svg(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return f'<div class="empty">Could not read {escape(path.name)}</div>'
    if "<svg" not in text.lower():
        return f'<pre>{escape(text[:4000])}</pre>'
    return text


def _svg_cards(files: list[Path], exp_dir: Path, title: str) -> str:
    if not files:
        return f'<div class="empty">No {escape(title.lower())} recorded.</div>'
    cards = []
    for file in files:
        label = escape(file.stem.replace("_", " ").title())
        cards.append(f'<figure><figcaption>{label}</figcaption><div class="svg-wrap">{_inline_svg(file)}</div></figure>')
    return "".join(cards)


def _model_abstraction_section(meta: dict[str, Any]) -> str:
    architecture = escape(str(meta.get("model_architecture", "Not documented")))
    task_type = escape(str(meta.get("task_type", "Not documented")))
    validation = escape(str(meta.get("validation_technique", "Not documented")))
    svg = """
<svg id="ResearchModelAbstraction" xmlns="http://www.w3.org/2000/svg" width="1180" height="430" viewBox="0 0 1180 430">
  <defs><marker id="rma-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#38bdf8"/></marker></defs>
  <rect width="1180" height="430" fill="#07111f"/>
  <text x="38" y="42" fill="#e5f0ff" font-size="24" font-family="Arial" font-weight="700">SciQuest model abstraction</text>
  <g font-family="Arial" font-size="14">
    <rect x="55" y="105" width="220" height="120" rx="16" fill="#0f2d3a" stroke="#22d3ee" stroke-width="2"/>
    <text x="78" y="140" fill="#e5f0ff" font-size="18" font-weight="700">State/context x</text>
    <text x="78" y="170" fill="#b6c2d2">observed environment</text>
    <text x="78" y="194" fill="#b6c2d2">history, constraints, covariates</text>
    <rect x="55" y="270" width="220" height="88" rx="16" fill="#3d2a0b" stroke="#fbbf24" stroke-width="2"/>
    <text x="78" y="305" fill="#e5f0ff" font-size="18" font-weight="700">Action a</text>
    <text x="78" y="333" fill="#b6c2d2">candidate intervention</text>
    <path d="M275 165 C340 165 340 230 410 230" stroke="#38bdf8" stroke-width="3" fill="none" marker-end="url(#rma-arrow)"/>
    <path d="M275 314 C340 314 340 260 410 260" stroke="#38bdf8" stroke-width="3" fill="none" marker-end="url(#rma-arrow)"/>
    <rect x="410" y="150" width="300" height="150" rx="16" fill="#123927" stroke="#34d399" stroke-width="2"/>
    <text x="435" y="185" fill="#e5f0ff" font-size="18" font-weight="700">Learned world model fθ(x,a)</text>
    <text x="435" y="218" fill="#b6c2d2">predicts counterfactual outcomes</text>
    <text x="435" y="242" fill="#b6c2d2">compared against scientific baselines</text>
    <text x="435" y="266" fill="#fbbf24">ŷ = fθ(x, a)</text>
    <path d="M710 225 C770 225 770 225 835 225" stroke="#38bdf8" stroke-width="3" fill="none" marker-end="url(#rma-arrow)"/>
    <rect x="835" y="105" width="280" height="120" rx="16" fill="#271b45" stroke="#a78bfa" stroke-width="2"/>
    <text x="860" y="140" fill="#e5f0ff" font-size="18" font-weight="700">Predicted outcome ŷ</text>
    <text x="860" y="170" fill="#b6c2d2">target variables and artifacts</text>
    <text x="860" y="194" fill="#b6c2d2">plots, reports, validation results</text>
    <rect x="835" y="270" width="280" height="90" rx="16" fill="#451a2b" stroke="#fb7185" stroke-width="2"/>
    <text x="860" y="304" fill="#e5f0ff" font-size="18" font-weight="700">Validation score</text>
    <text x="860" y="333" fill="#fbbf24">S = Σ wᵢ sᵢ / Σ wᵢ</text>
  </g>
</svg>
"""
    return f'''
<section class="panel" id="research-model-abstraction">
  <h3>Research Model Abstraction</h3>
  <div class="grid two">
    <article>
      <h4>Latest model architecture</h4>
      <p>{architecture}</p>
      <h4>Task type</h4>
      <p>{task_type}</p>
      <h4>Validation technique</h4>
      <p>{validation}</p>
      <p>The common SciQuest pattern is: represent the state/context, choose or simulate an action/intervention, train a model to predict the target/counterfactual outcome, then score it with a validation suite.</p>
      <p class="equation">\\(\\hat{{y}} = f_\\theta(x, a)\\)</p>
      <p class="equation">\\(S = \\sum_i w_i s_i / \\sum_i w_i\\)</p>
    </article>
    <article><div class="svg-wrap">{svg}</div></article>
  </div>
</section>
'''


def build_dashboard(quest_path: Path, output_dir: Path | None = None) -> Path:
    """Build a static, dynamic-by-metadata dashboard for a SciQuest quest."""
    output_dir = output_dir or quest_path / "reports" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    quest = read_yaml(quest_path / "quest.yaml", {})
    state = read_yaml(quest_path / "state.yaml", {})
    experiments = sorted((quest_path / "experiments").glob("exp_*"))
    nav = []
    sections = []
    active_exp_id = next((p.name for p in reversed(experiments) if p.is_dir()), None)
    latest_meta = read_yaml(quest_path / "experiments" / active_exp_id / "experiment.yaml", {}) if active_exp_id else {}
    for exp in experiments:
        if not exp.is_dir():
            continue
        exp_id = exp.name
        meta = read_yaml(exp / "experiment.yaml", {})
        validation = read_yaml(exp / "validation_results.yaml", {})
        report_text = (exp / "experiment_report.md").read_text(encoding="utf-8", errors="ignore") if (exp / "experiment_report.md").exists() else ""
        diagrams = sorted((exp / "artifacts" / "diagrams").glob("*.svg")) if (exp / "artifacts" / "diagrams").exists() else []
        plots = sorted((exp / "artifacts" / "plots").glob("*.svg")) if (exp / "artifacts" / "plots").exists() else []
        active_class = " active" if exp_id == active_exp_id else ""
        nav.append(f'<button type="button" class="exp-tab{active_class}" data-exp-target="{exp_id}" onclick="showExperiment(\'{exp_id}\')">{exp_id}<span>{escape(str(validation.get("aggregate_score", "pending")))}</span></button>')
        sections.append(f'''
<section class="experiment{active_class}" id="{exp_id}">
  <header class="exp-header">
    <div><p class="eyebrow">Experiment</p><h2>{exp_id}</h2></div>
    <div class="score"><span>Aggregate</span><strong>{escape(str(validation.get('aggregate_score', 'pending')))}</strong></div>
  </header>
  <div class="grid two">
    <article class="panel"><h3>Task + Model</h3><dl>
      <dt>Task type</dt><dd>{escape(str(meta.get('task_type', 'Not documented')))}</dd>
      <dt>Model architecture</dt><dd>{escape(str(meta.get('model_architecture', 'Not documented')))}</dd>
      <dt>Validation technique</dt><dd>{escape(str(meta.get('validation_technique', 'Not documented')))}</dd>
    </dl></article>
    <article class="panel"><h3>Features</h3><div class="feature-cols"><div><h4>Inputs</h4><ul>{_list_items(meta.get('input_features') or [])}</ul></div><div><h4>Targets</h4><ul>{_list_items(meta.get('target_features') or [])}</ul></div></div></article>
  </div>
  <article class="panel"><h3>Technical Diagrams</h3><div class="media-grid">{_svg_cards(diagrams, exp, 'Technical Diagrams')}</div></article>
  <article class="panel"><h3>Validation Metrics</h3><table><thead><tr><th>Metric</th><th>Raw</th><th>Normalized</th><th>Weight</th></tr></thead><tbody>{_metric_table(validation)}</tbody></table></article>
  <article class="panel"><h3>Result Graphs</h3><div class="media-grid">{_svg_cards(plots, exp, 'Result Graphs')}</div></article>
  <article class="panel interpretation"><h3>Graph Interpretation</h3><pre>{escape(_extract_interpretation(report_text))}</pre></article>
</section>
''')
    html = DASHBOARD_TEMPLATE.format(
        logo=_logo_html(quest_path),
        title=escape(str(quest.get("slug", quest_path.name))),
        hero=escape(str(quest.get("hero_statement", ""))),
        problem=escape(str(quest.get("problem_statement", ""))),
        status=escape(str(state.get("quest_status", "unknown"))),
        best=escape(str(state.get("best_score", "pending"))),
        nav="".join(nav) or '<span class="empty">No experiments yet.</span>',
        sections="".join(sections) or '<section class="panel"><h2>No experiments yet</h2></section>',
        metric_definitions=METRIC_DEFINITIONS_HTML,
        model_abstraction=_model_abstraction_section(latest_meta),
    )
    out = output_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SciQuest Dashboard - {title}</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)'], ['$', '$']], displayMath: [['\\\\[', '\\\\]']] }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {{ color-scheme: dark; --bg:#050B16; --panel:#0A1626; --panel2:#101F33; --text:#EAF7FF; --muted:#8CA4BD; --cyan:#22E6FF; --aqua:#3FFFE0; --emerald:#45FF7A; --lime:#A6FF3D; --violet:#9B6CFF; --magenta:#C06BFF; --amber:#D6FF4D; --line:rgba(34,230,255,.22); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter, Space Grotesk, ui-sans-serif, system-ui, sans-serif; background:radial-gradient(circle at 20% 0%, rgba(34,230,255,.16), transparent 32%), radial-gradient(circle at 90% 15%, rgba(155,108,255,.12), transparent 30%), #050B16; color:var(--text); }}
.shell {{ display:grid; grid-template-columns:320px 1fr; min-height:100vh; }}
.sidebar {{ position:sticky; top:0; height:100vh; padding:22px; border-right:1px solid rgba(34,230,255,.18); background:linear-gradient(180deg,rgba(5,11,22,.96),rgba(8,18,31,.9)); box-shadow:10px 0 40px rgba(34,230,255,.05); overflow:auto; }}
.brand {{ margin-bottom:20px; }} .sciquest-logo {{ width:100%; max-width:235px; display:block; margin:0 auto 12px; border-radius:18px; box-shadow:0 0 28px rgba(34,230,255,.22),0 0 50px rgba(166,255,61,.08); }} .logo-fallback {{ font-size:28px; font-weight:900; background:linear-gradient(90deg,var(--cyan),var(--emerald),var(--lime)); -webkit-background-clip:text; color:transparent; }} .brand-title {{ font-weight:800; letter-spacing:.08em; color:var(--cyan); text-align:center; text-transform:uppercase; font-size:12px; }}
.exp-tab {{ width:100%; display:flex; justify-content:space-between; gap:12px; padding:10px 12px; margin:8px 0; color:var(--text); text-align:left; border:1px solid rgba(34,230,255,.18); border-radius:14px; background:rgba(10,22,38,.78); cursor:pointer; box-shadow:inset 0 0 16px rgba(155,108,255,.03); }} .exp-tab:hover,.exp-tab.active {{ border-color:var(--cyan); background:linear-gradient(90deg,rgba(34,230,255,.16),rgba(69,255,122,.08)); box-shadow:0 0 18px rgba(34,230,255,.16); }}
main {{ padding:34px; }} .hero {{ position:relative; border:1px solid rgba(34,230,255,.23); border-radius:26px; padding:30px; background:linear-gradient(135deg,rgba(34,230,255,.12),rgba(155,108,255,.09) 54%,rgba(166,255,61,.07)); margin-bottom:24px; box-shadow:0 0 40px rgba(34,230,255,.08), inset 0 0 24px rgba(155,108,255,.04); }}
h1 {{ margin:.2rem 0; font-size:34px; letter-spacing:-.03em; line-height:1.05; }} h2 {{ font-weight:650; color:#dff7ff; }} .muted, .eyebrow {{ color:var(--muted); }} .eyebrow {{ text-transform:uppercase; letter-spacing:.16em; font-size:12px; color:var(--aqua); }} .cycle-strip {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }} .cycle-strip span {{ border:1px solid rgba(166,255,61,.32); color:#dfffc6; border-radius:999px; padding:6px 10px; font-size:12px; letter-spacing:.08em; text-transform:uppercase; background:rgba(166,255,61,.06); }}
.experiment {{ display:none; margin:28px 0; padding-top:8px; }} .experiment.active {{ display:block; }} .exp-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }} .score {{ border:1px solid var(--emerald); padding:12px 16px; border-radius:16px; background:rgba(69,255,122,.08); box-shadow:0 0 20px rgba(69,255,122,.08); }} .score span {{ display:block; color:var(--muted); font-size:12px; }} .score strong {{ color:var(--emerald); }}
.grid.two {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }} .panel {{ border:1px solid var(--line); border-radius:20px; background:rgba(10,22,38,.82); padding:20px; margin-bottom:18px; box-shadow:0 20px 60px rgba(0,0,0,.28), 0 0 24px rgba(34,230,255,.05), inset 0 0 18px rgba(155,108,255,.035); backdrop-filter:blur(10px); }} .panel h3 {{ margin-top:0; color:#F2FBFF; letter-spacing:-.01em; }}
dl {{ display:grid; grid-template-columns:160px 1fr; gap:8px 14px; }} dt {{ color:var(--muted); }} dd {{ margin:0; }} .feature-cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.media-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }} figure {{ margin:0; border:1px solid var(--line); background:white; border-radius:14px; padding:10px; overflow:hidden; }} figcaption {{ color:#0f172a; font-weight:700; margin-bottom:8px; }} .svg-wrap svg {{ width:100%; height:auto; display:block; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:10px; border-bottom:1px solid var(--line); }} th {{ color:var(--cyan); }} pre {{ white-space:pre-wrap; font-family:inherit; color:var(--text); line-height:1.5; }} .empty {{ color:var(--muted); padding:12px; }} .metric-defs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; }} .metric-card {{ border:1px solid var(--line); border-radius:14px; padding:14px; background:rgba(2,6,23,.45); }} .metric-card h4 {{ margin:0 0 8px; color:var(--cyan); }} .equation {{ color:var(--amber); font-family:'Times New Roman',serif; font-size:18px; }}
@media (max-width: 900px) {{ .shell {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; height:auto; }} .grid.two,.feature-cols {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="shell"><aside class="sidebar"><div class="brand">{logo}<div class="brand-title">SciQuest Dashboard</div></div><p class="muted">Status: {status}<br>Best score: {best}</p><nav>{nav}</nav></aside><main><section class="hero"><div class="cycle-strip"><span>Hypothesize</span><span>Model</span><span>Validate</span><span>Iterate</span></div><p class="eyebrow">Quest</p><h1>{title}</h1><h2>{hero}</h2><p>{problem}</p></section>{sections}{metric_definitions}{model_abstraction}</main></div>
<script>
function showExperiment(id) {{
  document.querySelectorAll('.experiment').forEach(el => el.classList.toggle('active', el.id === id));
  document.querySelectorAll('.exp-tab').forEach(el => el.classList.toggle('active', el.dataset.expTarget === id));
  history.replaceState(null, '', '#' + id);
}}
window.addEventListener('DOMContentLoaded', () => {{
  const requested = location.hash ? location.hash.slice(1) : null;
  if (requested && document.getElementById(requested)) showExperiment(requested);
}});
</script>
</body>
</html>"""

METRIC_DEFINITIONS_HTML = r"""
<section class="panel" id="metric-definitions">
  <h3>Validation Metric Definitions</h3>
  <div class="metric-defs">
    <article class="metric-card">
      <h4>Weighted aggregate score</h4>
      <p>Normalized metric scores are combined with user/agent-defined weights.</p>
      <p class="equation">\(S = \sum_i w_i s_i / \sum_i w_i\)</p>
    </article>
    <article class="metric-card">
      <h4>WAPE</h4>
      <p>Weighted absolute percentage error. Lower is better for forecast/counterfactual error.</p>
      <p class="equation">\(WAPE = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}\)</p>
    </article>
    <article class="metric-card">
      <h4>Relative WAPE lift</h4>
      <p>Improvement versus a baseline simulator or model. Higher is better.</p>
      <p class="equation">\(Lift = (WAPE_{baseline} - WAPE_{model}) / WAPE_{baseline}\)</p>
    </article>
    <article class="metric-card">
      <h4>RMSE</h4>
      <p>Root mean squared error, used here for demand/unit prediction error. Lower is better.</p>
      <p class="equation">\(RMSE = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}\)</p>
    </article>
    <article class="metric-card">
      <h4>Rank correlation</h4>
      <p>Spearman correlation of true and predicted price-action rankings. Higher is better.</p>
      <p class="equation">\(\rho = corr(rank(y), rank(\hat{y}))\)</p>
    </article>
    <article class="metric-card">
      <h4>Law-of-demand pass rate</h4>
      <p>Share of comparable states where predicted demand is non-increasing as price rises.</p>
      <p class="equation">\(PassRate = \frac{1}{|G|}\sum_g 1[\hat{d}_{g,p_1} \geq \hat{d}_{g,p_2}\;\forall p_1&lt;p_2]\)</p>
    </article>
  </div>
</section>
"""
