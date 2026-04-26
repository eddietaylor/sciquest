from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .io import read_yaml


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


def build_dashboard(quest_path: Path, output_dir: Path | None = None) -> Path:
    """Build a static, dynamic-by-metadata dashboard for a SciQuest quest."""
    output_dir = output_dir or quest_path / "reports" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    quest = read_yaml(quest_path / "quest.yaml", {})
    state = read_yaml(quest_path / "state.yaml", {})
    experiments = sorted((quest_path / "experiments").glob("exp_*"))
    nav = []
    sections = []
    for exp in experiments:
        if not exp.is_dir():
            continue
        exp_id = exp.name
        meta = read_yaml(exp / "experiment.yaml", {})
        validation = read_yaml(exp / "validation_results.yaml", {})
        report_text = (exp / "experiment_report.md").read_text(encoding="utf-8", errors="ignore") if (exp / "experiment_report.md").exists() else ""
        diagrams = sorted((exp / "artifacts" / "diagrams").glob("*.svg")) if (exp / "artifacts" / "diagrams").exists() else []
        plots = sorted((exp / "artifacts" / "plots").glob("*.svg")) if (exp / "artifacts" / "plots").exists() else []
        nav.append(f'<a href="#{exp_id}">{exp_id}<span>{escape(str(validation.get("aggregate_score", "pending")))}</span></a>')
        sections.append(f'''
<section class="experiment" id="{exp_id}">
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
        title=escape(str(quest.get("slug", quest_path.name))),
        hero=escape(str(quest.get("hero_statement", ""))),
        problem=escape(str(quest.get("problem_statement", ""))),
        status=escape(str(state.get("quest_status", "unknown"))),
        best=escape(str(state.get("best_score", "pending"))),
        nav="".join(nav) or '<span class="empty">No experiments yet.</span>',
        sections="".join(sections) or '<section class="panel"><h2>No experiments yet</h2></section>',
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
<style>
:root {{ color-scheme: dark; --bg:#07111f; --panel:#101b2d; --panel2:#0c1728; --text:#e5f0ff; --muted:#93a4bb; --cyan:#22d3ee; --emerald:#34d399; --violet:#a78bfa; --amber:#fbbf24; --line:#22334d; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, sans-serif; background:radial-gradient(circle at top left,#123456 0,#07111f 35%,#020617 100%); color:var(--text); }}
.shell {{ display:grid; grid-template-columns:280px 1fr; min-height:100vh; }}
.sidebar {{ position:sticky; top:0; height:100vh; padding:24px; border-right:1px solid var(--line); background:rgba(2,6,23,.82); }}
.brand {{ font-weight:800; letter-spacing:.08em; color:var(--cyan); }} .sidebar a {{ display:flex; justify-content:space-between; gap:12px; padding:10px 12px; margin:8px 0; color:var(--text); text-decoration:none; border:1px solid var(--line); border-radius:12px; background:#0f172a; }}
main {{ padding:32px; }} .hero {{ border:1px solid var(--line); border-radius:24px; padding:28px; background:linear-gradient(135deg,rgba(34,211,238,.12),rgba(167,139,250,.1)); margin-bottom:24px; }}
h1 {{ margin:.2rem 0; font-size:32px; }} .muted, .eyebrow {{ color:var(--muted); }} .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-size:12px; }}
.experiment {{ margin:28px 0; padding-top:8px; }} .exp-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }} .score {{ border:1px solid var(--emerald); padding:12px 16px; border-radius:16px; background:rgba(52,211,153,.08); }} .score span {{ display:block; color:var(--muted); font-size:12px; }} .score strong {{ color:var(--emerald); }}
.grid.two {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }} .panel {{ border:1px solid var(--line); border-radius:18px; background:rgba(15,23,42,.88); padding:18px; margin-bottom:18px; box-shadow:0 20px 50px rgba(0,0,0,.22); }}
dl {{ display:grid; grid-template-columns:160px 1fr; gap:8px 14px; }} dt {{ color:var(--muted); }} dd {{ margin:0; }} .feature-cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.media-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }} figure {{ margin:0; border:1px solid var(--line); background:white; border-radius:14px; padding:10px; overflow:hidden; }} figcaption {{ color:#0f172a; font-weight:700; margin-bottom:8px; }} .svg-wrap svg {{ width:100%; height:auto; display:block; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:10px; border-bottom:1px solid var(--line); }} th {{ color:var(--cyan); }} pre {{ white-space:pre-wrap; font-family:inherit; color:var(--text); line-height:1.5; }} .empty {{ color:var(--muted); padding:12px; }}
@media (max-width: 900px) {{ .shell {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; height:auto; }} .grid.two,.feature-cols {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="shell"><aside class="sidebar"><div class="brand">SciQuest Dashboard</div><p class="muted">Status: {status}<br>Best score: {best}</p><nav>{nav}</nav></aside><main><section class="hero"><p class="eyebrow">Quest</p><h1>{title}</h1><h2>{hero}</h2><p>{problem}</p></section>{sections}</main></div>
</body>
</html>"""
