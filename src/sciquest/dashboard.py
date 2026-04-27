from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import base64
import csv
import mimetypes
import re

from .io import read_yaml

SEMANTIC_CSS_CLASSES = "semantic-state semantic-action semantic-pass semantic-metric semantic-risk semantic-best"

OPERATOR_EXPLANATIONS: tuple[dict[str, Any], ...] = (
    {
        "term": "Inverse-propensity-weighted ridge regression",
        "aliases": ("inverse-propensity-weighted ridge regression", "inverse propensity weighted ridge regression"),
        "explanation": (
            "Some observations are more likely to appear under certain pricing policies. "
            "Inverse propensity weighting tries to rebalance the training data so the model does not simply learn "
            "the historical pricing policy. Ridge regression is the actual prediction model, with regularization "
            "to avoid overfitting."
        ),
    },
    {
        "term": "Log-demand space",
        "aliases": ("log-demand space", "log demand space"),
        "explanation": (
            "The model predicts the logarithm of demand instead of raw demand. This usually stabilizes variance "
            "and helps keep predictions positive after exponentiation."
        ),
    },
    {
        "term": "Rank-aware objective",
        "aliases": ("rank-aware objective", "rank aware objective"),
        "explanation": (
            "The model is not only judged by absolute error. It is also judged by whether it correctly ranks "
            "scenarios from better to worse."
        ),
    },
    {
        "term": "Law-of-demand pass rate",
        "aliases": ("law-of-demand pass rate", "law of demand pass rate"),
        "explanation": "This checks whether predicted demand usually decreases when price increases.",
    },
)


_OPERATOR_ALIAS_MAP = {
    alias.lower(): item
    for item in OPERATOR_EXPLANATIONS
    for alias in item["aliases"]
}


def _operator_explain_text(text: str) -> str:
    if not text:
        return escape("Not documented")
    aliases = sorted(_OPERATOR_ALIAS_MAP, key=len, reverse=True)
    if not aliases:
        return escape(text)
    pattern = re.compile(r"(?<![\w-])(" + "|".join(re.escape(alias) for alias in aliases) + r")(?![\w-])", re.IGNORECASE)
    chunks: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        chunks.append(escape(text[last:match.start()]))
        item = _OPERATOR_ALIAS_MAP[match.group(0).lower()]
        chunks.append(
            '<details class="operator-explain">'
            f'<summary>{escape(str(item["term"]))}</summary>'
            f'<p>{escape(str(item["explanation"]))}</p>'
            '</details>'
        )
        last = match.end()
    chunks.append(escape(text[last:]))
    return "".join(chunks)


def _operator_glossary_section() -> str:
    cards = []
    for item in OPERATOR_EXPLANATIONS:
        cards.append(
            '<details class="operator-explain operator-card">'
            f'<summary>{escape(str(item["term"]))}</summary>'
            f'<p>{escape(str(item["explanation"]))}</p>'
            '</details>'
        )
    return f'''
<section class="panel operator-mode" id="operator-explanations">
  <p class="eyebrow">Explainability Mode</p>
  <h3>Explain this like I’m the operator</h3>
  <p class="muted">Technical phrases in the dashboard appear as expandable chips. Open them when you want the operational meaning without losing the scientific detail.</p>
  <div class="operator-grid">{"".join(cards)}</div>
</section>
'''


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


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    n = _num(value)
    if n is None:
        return escape(str(value if value is not None else "—"))
    return f"{n:.{digits}f}"


def _list_items(items: list[Any]) -> str:
    if not items:
        return "<li>Not documented</li>"
    return "".join(f"<li>{_operator_explain_text(str(item))}</li>" for item in items)


def _split_description(text: str) -> str:
    parts = [p.strip(" .") for p in re.split(r";|\. (?=[A-Z])", text or "") if p.strip()]
    if len(parts) <= 1:
        return f"<p>{_operator_explain_text(text or 'Not documented')}</p>"
    labels = ["Architecture", "Training signal", "Constraints", "Baselines", "Calibration", "Exclusions", "Notes"]
    rows = []
    for i, part in enumerate(parts):
        rows.append(f"<li><span>{labels[min(i, len(labels)-1)]}</span>{_operator_explain_text(part)}</li>")
    return f'<ul class="model-field-list">{"".join(rows)}</ul>'


def _metric_table(validation: dict[str, Any]) -> str:
    metrics = validation.get("metrics", {}) or {}
    scores = validation.get("scores", {}) or {}
    rows = []
    for name, raw in metrics.items():
        score = scores.get(name, {}) if isinstance(scores, dict) else {}
        rows.append(
            "<tr>"
            f"<td>{_operator_explain_text(str(name).replace('_', ' '))}</td>"
            f"<td class='mono'>{_fmt(raw, 6)}</td>"
            f"<td class='mono'>{_fmt(score.get('normalized'), 6) if score else '—'}</td>"
            f"<td class='mono'>{_fmt(score.get('weight'), 3) if score else '—'}</td>"
            "</tr>"
        )
    return "".join(rows) or '<tr><td colspan="4">No validation metrics recorded.</td></tr>'


def _metric_scorecards(validation: dict[str, Any]) -> str:
    metrics = validation.get("metrics", {}) or {}
    scores = validation.get("scores", {}) or {}
    cards = []
    for name, raw in list(metrics.items())[:8]:
        normalized = _num((scores.get(name) or {}).get("normalized")) if isinstance(scores, dict) else None
        cls = "semantic-pass" if normalized is not None and normalized >= 0.75 else "semantic-risk" if normalized is not None and normalized < 0.45 else "semantic-metric"
        cards.append(f"""
        <article class="metric-scorecard {cls}">
          <span>{_operator_explain_text(str(name).replace('_', ' '))}</span>
          <strong class="mono">{_fmt(raw, 5)}</strong>
          <small>normalized {_fmt(normalized, 3) if normalized is not None else '—'}</small>
        </article>""")
    return f'<div class="metric-scorecards">{"".join(cards)}</div>' if cards else ""


def _extract_interpretation(report_text: str) -> str:
    if not report_text:
        return "No interpretation report available."
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    evidence, speculation = [], []
    target = None
    for line in lines:
        lower = line.lower()
        if lower.startswith("## evidence"):
            target = evidence; continue
        if lower.startswith("## speculation"):
            target = speculation; continue
        if lower.startswith("## "):
            target = None; continue
        if target is not None and not line.startswith("#"):
            target.append(line.lstrip("- "))
    text = "\n".join(evidence[-2:] + speculation[:2]) or "\n".join(lines[:8])
    return text[:1600]


def _one_line_caption(file: Path, interpretation: str) -> str:
    name = file.stem.replace("_", " ").replace("exp002", "").title()
    if "wape" in file.stem.lower():
        return "Compares counterfactual revenue error against the baseline."
    if "predicted" in file.stem.lower():
        return "Shows calibration of predicted versus true counterfactual revenue."
    if "metric" in file.stem.lower():
        return "Summarizes validation strengths and weak points."
    return (interpretation.split(".")[0] + ".") if interpretation else f"{name} artifact."


def _theme_svg(text: str) -> str:
    # Keep scientific content, but remove white-canvas feel when agents emit simple white SVGs.
    themed = text
    replacements = {
        'fill="white"': 'fill="#07111f"',
        "fill='white'": "fill='#07111f'",
        'fill="#FFFFFF"': 'fill="#07111f"',
        'fill="#fff"': 'fill="#07111f"',
        'stroke="#555"': 'stroke="#8CA4BD"',
        'stroke="black"': 'stroke="#EAF7FF"',
        'fill="black"': 'fill="#EAF7FF"',
    }
    for old, new in replacements.items():
        themed = themed.replace(old, new)
    if "<svg" in themed and "filter=" not in themed:
        themed = themed.replace("<svg", "<svg class=\"themed-svg\"", 1)
    return themed


def _inline_svg(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return f'<div class="empty">Could not read {escape(path.name)}</div>'
    if "<svg" not in text.lower():
        return f'<pre>{escape(text[:4000])}</pre>'
    return _theme_svg(text)


def _svg_cards(files: list[Path], exp_dir: Path, title: str, interpretation: str = "") -> str:
    if not files:
        return f'<div class="empty">No {escape(title.lower())} recorded.</div>'
    cards = []
    for file in files:
        label = escape(file.stem.replace("_", " ").title())
        caption = escape(_one_line_caption(file, interpretation))
        cards.append(f'<figure><figcaption>{label}</figcaption><div class="svg-wrap">{_inline_svg(file)}</div><p class="figure-caption">{caption}</p></figure>')
    return "".join(cards)


def _verdict(score: float | None, delta: float | None) -> str:
    if score is None:
        return "Pending validation."
    if score >= 0.8 and (delta is None or delta >= 0):
        return "Breakthrough-quality iteration: preserve this line of inquiry and stress-test it."
    if delta is not None and delta < -0.05:
        return "Scientifically useful negative result: failure mode exposed; analyze risk before iterating."
    if score >= 0.65:
        return "Promising but not decisive: continue with stronger validation and baselines."
    return "Weak validation result: treat as a falsifying or diagnostic iteration."


def _main_improvement_failure(validation: dict[str, Any]) -> tuple[str, str]:
    scores = validation.get("scores", {}) or {}
    if not scores:
        return "No scored metrics yet.", "No metric failures identified."
    ordered = sorted(scores.items(), key=lambda kv: _num(kv[1].get("normalized")) or 0)
    failure_name, failure = ordered[0]
    improve_name, improve = ordered[-1]
    return (
        f"{improve_name.replace('_', ' ')} normalized at {_fmt(improve.get('normalized'), 3)}.",
        f"{failure_name.replace('_', ' ')} is the weakest metric at {_fmt(failure.get('normalized'), 3)}.",
    )


def _recommended_next_experiment(validation: dict[str, Any], interpretation: str) -> str:
    _, failure = _main_improvement_failure(validation)
    return f"Target the current weakest validation signal: {escape(failure)} Use the interpretation above to design one hypothesis that isolates this failure mode while preserving the strongest passing behavior."


def _human_title(quest: dict[str, Any], quest_path: Path) -> str:
    explicit = quest.get("title") or quest.get("name")
    if explicit:
        return str(explicit)
    slug = str(quest.get("slug") or quest_path.name)
    return slug.replace("-", " ").replace("_", " ").title()


def _data_source_label(manifest: dict[str, Any]) -> str:
    text = " ".join(
        str(value)
        for value in [manifest.get("status"), manifest.get("source"), manifest.get("provenance"), manifest.get("description")]
        if value
    )
    for dataset in manifest.get("datasets") or []:
        if isinstance(dataset, dict):
            text += " " + " ".join(str(dataset.get(key, "")) for key in ("name", "path", "provenance", "description", "source"))
    lower = text.lower()
    if any(word in lower for word in ("synthetic", "generated", "simulated", "agent generated")):
        return "Synthetic"
    if any(word in lower for word in ("user", "provided", "uploaded", "declared")):
        return "User provided"
    if manifest.get("status") == "missing_user_data":
        return "Missing user data; agent required"
    return "Not documented"


def _dataset_candidates(quest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for dataset in manifest.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        raw_path = dataset.get("path")
        if raw_path:
            p = Path(str(raw_path))
            candidates.append(p if p.is_absolute() else quest_path / p)
    candidates.extend(sorted((quest_path / "data" / "raw").glob("*.csv")))
    candidates.extend(sorted((quest_path / "data" / "processed").glob("*.csv")))
    seen: set[Path] = set()
    unique = []
    for p in candidates:
        resolved = p.resolve() if p.exists() else p
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def _dataset_profile(quest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    for path in _dataset_candidates(quest_path, manifest):
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        try:
            with path.open(newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                numeric: dict[str, list[float]] = {field: [] for field in fields}
                rows = 0
                for row in reader:
                    rows += 1
                    for field in fields:
                        value = _num(row.get(field))
                        if value is not None:
                            numeric[field].append(value)
        except OSError:
            continue
        stats = []
        for field, values in numeric.items():
            if not values:
                continue
            stats.append({
                "name": field,
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            })
        return {"path": path, "rows": rows, "columns": len(fields), "fields": fields, "stats": stats[:6]}
    return {"path": None, "rows": None, "columns": None, "fields": [], "stats": []}


def _data_panel(meta: dict[str, Any], manifest: dict[str, Any], quest_path: Path) -> str:
    profile = _dataset_profile(quest_path, manifest)
    source = _data_source_label(manifest)
    rows = profile.get("rows")
    columns = profile.get("columns")
    shape = f"{rows} rows × {columns} columns" if rows is not None and columns is not None else "Not documented"
    path = profile.get("path")
    if isinstance(path, Path) and path.exists():
        try:
            path_label = path.relative_to(quest_path)
        except ValueError:
            path_label = path
    else:
        path_label = "Not documented"
    stats_rows = []
    for stat in profile.get("stats") or []:
        stats_rows.append(
            "<tr>"
            f"<td>{escape(str(stat['name']))}</td>"
            f"<td class='mono'>{_fmt(stat['mean'], 3)}</td>"
            f"<td class='mono'>{_fmt(stat['min'], 3)}</td>"
            f"<td class='mono'>{_fmt(stat['max'], 3)}</td>"
            "</tr>"
        )
    stats_html = "".join(stats_rows) or '<tr><td colspan="4">No numeric descriptive statistics available.</td></tr>'
    return f'''
    <article class="panel data-panel"><h3>The Data</h3>
      <div class="data-facts">
        <div><span>Data source</span><strong>{escape(source)}</strong></div>
        <div><span>Data shape</span><strong class="mono">{escape(shape)}</strong></div>
        <div><span>Dataset path</span><strong>{escape(str(path_label))}</strong></div>
      </div>
      <div class="feature-cols"><div><h4 class="semantic-state">Input features</h4><ul>{_list_items(meta.get('input_features') or [])}</ul></div><div><h4 class="semantic-action">Targets / outcomes</h4><ul>{_list_items(meta.get('target_features') or [])}</ul></div></div>
      <h4>Descriptive statistics</h4>
      <table class="stats-table"><thead><tr><th>Feature</th><th>Mean</th><th>Min</th><th>Max</th></tr></thead><tbody>{stats_html}</tbody></table>
    </article>'''


def _model_abstraction_section(meta: dict[str, Any]) -> str:
    architecture = _operator_explain_text(str(meta.get("model_architecture", "Not documented")))
    task_type = _operator_explain_text(str(meta.get("task_type", "Not documented")))
    validation = _operator_explain_text(str(meta.get("validation_technique", "Not documented")))
    svg = """
<svg id="ResearchModelAbstraction" xmlns="http://www.w3.org/2000/svg" width="1180" height="430" viewBox="0 0 1180 430">
  <defs><marker id="rma-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#22E6FF"/></marker></defs>
  <rect width="1180" height="430" fill="#050B16"/>
  <text x="38" y="42" fill="#EAF7FF" font-size="24" font-family="Space Grotesk, Arial" font-weight="700">SciQuest model abstraction</text>
  <g font-family="Inter, Arial" font-size="14">
    <rect x="55" y="105" width="220" height="120" rx="16" fill="#0f2d3a" stroke="#22E6FF" stroke-width="2"/>
    <text x="78" y="140" fill="#EAF7FF" font-size="18" font-weight="700">State/context x</text>
    <text x="78" y="170" fill="#C9D7E8">observed environment</text>
    <text x="78" y="194" fill="#C9D7E8">history, constraints, covariates</text>
    <rect x="55" y="270" width="220" height="88" rx="16" fill="#3d2a0b" stroke="#FF9F43" stroke-width="2"/>
    <text x="78" y="305" fill="#EAF7FF" font-size="18" font-weight="700">Action a</text>
    <text x="78" y="333" fill="#C9D7E8">candidate intervention</text>
    <path d="M275 165 C340 165 340 230 410 230" stroke="#22E6FF" stroke-width="3" fill="none" marker-end="url(#rma-arrow)"/>
    <path d="M275 314 C340 314 340 260 410 260" stroke="#FF9F43" stroke-width="3" fill="none" marker-end="url(#rma-arrow)"/>
    <rect x="410" y="150" width="300" height="150" rx="16" fill="#123927" stroke="#45FF7A" stroke-width="2"/>
    <text x="435" y="185" fill="#EAF7FF" font-size="18" font-weight="700">Learned world model fθ(x,a)</text>
    <text x="435" y="218" fill="#C9D7E8">predicts counterfactual outcomes</text>
    <text x="435" y="242" fill="#C9D7E8">compared against scientific baselines</text>
    <text x="435" y="266" fill="#D6FF4D">ŷ = fθ(x, a)</text>
    <path d="M710 225 C770 225 770 225 835 225" stroke="#45FF7A" stroke-width="3" fill="none" marker-end="url(#rma-arrow)"/>
    <rect x="835" y="105" width="280" height="120" rx="16" fill="#271b45" stroke="#9B6CFF" stroke-width="2"/>
    <text x="860" y="140" fill="#EAF7FF" font-size="18" font-weight="700">Predicted outcome ŷ</text>
    <text x="860" y="170" fill="#C9D7E8">target variables and artifacts</text>
    <text x="860" y="194" fill="#C9D7E8">plots, reports, validation results</text>
    <rect x="835" y="270" width="280" height="90" rx="16" fill="#451a2b" stroke="#FF4F64" stroke-width="2"/>
    <text x="860" y="304" fill="#EAF7FF" font-size="18" font-weight="700">Validation score</text>
    <text x="860" y="333" fill="#D6FF4D">S = Σ wᵢ sᵢ / Σ wᵢ</text>
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
    output_dir = output_dir or quest_path / "reports" / "dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    quest = read_yaml(quest_path / "quest.yaml", {})
    state = read_yaml(quest_path / "state.yaml", {})
    data_manifest = read_yaml(quest_path / "data_manifest.yaml", {})
    experiments = [p for p in sorted((quest_path / "experiments").glob("exp_*")) if p.is_dir()]
    exp_data: list[dict[str, Any]] = []
    for exp in experiments:
        exp_data.append({
            "path": exp,
            "id": exp.name,
            "meta": read_yaml(exp / "experiment.yaml", {}),
            "validation": read_yaml(exp / "validation_results.yaml", {}),
            "report": (exp / "experiment_report.md").read_text(encoding="utf-8", errors="ignore") if (exp / "experiment_report.md").exists() else "",
        })
    active = exp_data[-1] if exp_data else None
    best_score = max((_num(d["validation"].get("aggregate_score")) for d in exp_data), default=None)
    active_exp_id = active["id"] if active else None
    latest_meta = active["meta"] if active else {}
    nav, sections = [], []
    prev_score: float | None = None
    for d in exp_data:
        exp, exp_id, meta, validation, report_text = d["path"], d["id"], d["meta"], d["validation"], d["report"]
        score = _num(validation.get("aggregate_score"))
        delta = None if score is None or prev_score is None else score - prev_score
        active_class = " active" if exp_id == active_exp_id else ""
        best_badge = '<b class="best-badge semantic-best">BEST</b>' if score is not None and best_score is not None and abs(score - best_score) < 1e-12 else ""
        delta_html = "—" if delta is None else f"{delta:+.4f}"
        short_name = str(meta.get("task_type") or meta.get("hypothesis_id") or "experiment").split("/")[0].strip()
        nav.append(f'''
<button type="button" class="exp-tab{active_class}" data-exp-target="{exp_id}" onclick="showExperiment('{exp_id}')">
  <span class="timeline-dot"></span><span class="timeline-main"><b>{exp_id}</b><small>{escape(short_name)}</small></span>
  <span class="timeline-score mono">{_fmt(score, 4)}</span><span class="score-delta mono">{delta_html}</span>{best_badge}<span class="active-badge">ACTIVE</span>
</button>''')
        interpretation = _extract_interpretation(report_text)
        improvement, failure = _main_improvement_failure(validation)
        prev_label = "—" if delta is None else f"{delta:+.4f}"
        diagrams = sorted((exp / "artifacts" / "diagrams").glob("*.svg")) if (exp / "artifacts" / "diagrams").exists() else []
        plots = sorted((exp / "artifacts" / "plots").glob("*.svg")) if (exp / "artifacts" / "plots").exists() else []
        sections.append(f'''
<section class="experiment{active_class}" id="{exp_id}">
  <section class="verdict-card panel">
    <div><p class="eyebrow">Experiment Verdict</p><h2>{exp_id}</h2><p>{escape(_verdict(score, delta))}</p></div>
    <div class="verdict-grid">
      <div><span>Current score</span><strong class="mono">{_fmt(score, 6)}</strong></div>
      <div><span>Delta vs previous</span><strong class="mono {'semantic-pass' if delta is not None and delta >= 0 else 'semantic-risk'}">{prev_label}</strong></div>
      <div><span>Best score</span><strong class="mono semantic-best">{_fmt(best_score, 6)}</strong></div>
      <div><span>Main improvement</span><p>{escape(improvement)}</p></div>
      <div><span>Main failure</span><p>{escape(failure)}</p></div>
      <div><span>Scientific verdict</span><p>{escape(_verdict(score, delta))}</p></div>
    </div>
  </section>
  <div class="grid two summary-first">
    {_data_panel(meta, data_manifest, quest_path)}
    <article class="panel"><h3>Task + Model</h3><dl>
      <dt>Task type</dt><dd>{escape(str(meta.get('task_type', 'Not documented')))}</dd>
      <dt>Model architecture</dt><dd>{_split_description(str(meta.get('model_architecture', 'Not documented')))}</dd>
      <dt>Validation technique</dt><dd>{_split_description(str(meta.get('validation_technique', 'Not documented')))}</dd>
    </dl></article>
  </div>
  <article class="panel"><h3>Executive Validation Scorecards</h3>{_metric_scorecards(validation)}<h4>Detailed validation metrics</h4><table><thead><tr><th>Metric</th><th>Raw</th><th>Normalized</th><th>Weight</th></tr></thead><tbody>{_metric_table(validation)}</tbody></table></article>
  <article class="panel"><h3>Technical Diagrams</h3><div class="media-grid diagrams-grid">{_svg_cards(diagrams, exp, 'Technical Diagrams', interpretation)}</div></article>
  <article class="panel"><h3>Result Graphs</h3><div class="media-grid result-grid">{_svg_cards(plots[:4], exp, 'Result Graphs', interpretation)}</div></article>
  <article class="panel interpretation"><h3>Graph Interpretation</h3><pre>{escape(interpretation)}</pre></article>
  <article class="panel next-card"><h3>Recommended Next Experiment</h3><p>{_recommended_next_experiment(validation, interpretation)}</p></article>
</section>
''')
        prev_score = score
    html = DASHBOARD_TEMPLATE.format(
        logo=_logo_html(quest_path),
        title=escape(_human_title(quest, quest_path)),
        hero=escape(str(quest.get("hero_statement", ""))),
        problem=escape(str(quest.get("problem_statement", ""))),
        status=escape(str(state.get("quest_status", "unknown"))),
        best=escape(str(state.get("best_score", "pending"))),
        nav="".join(nav) or '<span class="empty">No experiments yet.</span>',
        sections="".join(sections) or '<section class="panel"><h2>No experiments yet</h2></section>',
        operator_glossary=_operator_glossary_section(),
        metric_definitions=METRIC_DEFINITIONS_HTML,
        model_abstraction=_model_abstraction_section(latest_meta),
        semantic_classes=SEMANTIC_CSS_CLASSES,
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<script>window.MathJax = {{tex: {{ inlineMath: [['\\\\(', '\\\\)'], ['$', '$']], displayMath: [['\\\\[', '\\\\]']] }}, svg: {{ fontCache: 'global' }}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {{ color-scheme: dark; --bg:#050B16; --panel:#0A1626; --text:#EAF7FF; --muted:#8CA4BD; --cyan:#22E6FF; --orange:#FF9F43; --green:#45FF7A; --purple:#9B6CFF; --red:#FF4F64; --gold:#D6FF4D; --line:rgba(34,230,255,.22); }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter, sans-serif; background:radial-gradient(circle at 20% 0%, rgba(34,230,255,.15), transparent 32%), radial-gradient(circle at 90% 15%, rgba(155,108,255,.13), transparent 30%), #050B16; color:var(--text); }}
.shell {{ display:grid; grid-template-columns:360px 1fr; min-height:100vh; }} .sidebar {{ position:sticky; top:0; height:100vh; overflow:auto; padding:24px; border-right:1px solid rgba(34,230,255,.18); background:linear-gradient(180deg,rgba(5,11,22,.97),rgba(8,18,31,.92)); }}
.sciquest-logo {{ width:245px; max-width:100%; display:block; margin:0 auto 14px; border-radius:22px; box-shadow:0 0 34px rgba(34,230,255,.24),0 0 70px rgba(166,255,61,.08); }} .logo-fallback {{ font-family:Space Grotesk; font-size:30px; font-weight:800; background:linear-gradient(90deg,var(--cyan),var(--green),var(--gold)); -webkit-background-clip:text; color:transparent; }} .brand-title {{ text-align:center; color:var(--cyan); font-weight:800; letter-spacing:.14em; text-transform:uppercase; font-size:12px; }}
.muted {{ color:var(--muted); }} .mono,.timeline-score,.score-delta {{ font-family:JetBrains Mono, monospace; }} h1,h2,h3,h4,.eyebrow {{ font-family:Space Grotesk, Inter, sans-serif; }} h1 {{ font-size:36px; line-height:1.02; letter-spacing:-.04em; margin:8px 0 10px; }} h2 {{ margin:0 0 10px; }} .eyebrow {{ color:var(--cyan); text-transform:uppercase; letter-spacing:.16em; font-size:12px; }}
.exp-tab {{ position:relative; width:100%; display:grid; grid-template-columns:14px 1fr auto auto; align-items:center; gap:10px; padding:13px 12px; margin:10px 0; color:var(--text); text-align:left; border:1px solid rgba(34,230,255,.18); border-radius:16px; background:rgba(10,22,38,.78); cursor:pointer; }} .exp-tab.active {{ border-color:var(--cyan); box-shadow:0 0 22px rgba(34,230,255,.16); }} .timeline-dot {{ width:10px; height:10px; border-radius:50%; background:var(--purple); box-shadow:0 0 12px var(--purple); }} .timeline-main small {{ display:block; color:var(--muted); max-width:150px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }} .best-badge,.active-badge {{ display:none; position:absolute; top:-8px; right:8px; font-size:9px; border-radius:999px; padding:2px 7px; }} .best-badge {{ display:block; border:1px solid var(--gold); color:var(--gold); background:rgba(214,255,77,.08); }} .exp-tab.active .active-badge {{ display:block; border:1px solid var(--cyan); color:var(--cyan); background:#07111f; }}
main {{ padding:38px; }} .hero,.panel {{ border:1px solid var(--line); border-radius:24px; background:rgba(10,22,38,.82); box-shadow:0 24px 70px rgba(0,0,0,.32),0 0 28px rgba(34,230,255,.05),inset 0 0 20px rgba(155,108,255,.035); backdrop-filter:blur(10px); }} .hero {{ padding:32px; margin-bottom:28px; background:linear-gradient(135deg,rgba(34,230,255,.13),rgba(155,108,255,.09) 55%,rgba(166,255,61,.06)); }} .panel {{ padding:22px; margin-bottom:22px; }} .cycle-strip {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px; }} .cycle-strip span {{ border:1px solid rgba(166,255,61,.35); color:#dfffc6; border-radius:999px; padding:6px 11px; font-size:12px; letter-spacing:.08em; text-transform:uppercase; background:rgba(166,255,61,.06); }}
.experiment {{ display:none; }} .experiment.active {{ display:block; }} .grid.two {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; }} .summary-first {{ align-items:start; }} .verdict-card {{ display:grid; grid-template-columns:1.1fr 2fr; gap:24px; border-color:rgba(214,255,77,.28); }} .verdict-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }} .verdict-grid div,.metric-scorecard {{ border:1px solid rgba(34,230,255,.16); border-radius:16px; padding:14px; background:rgba(5,11,22,.38); }} .verdict-grid span,.metric-scorecard span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:7px; }} .verdict-grid strong,.metric-scorecard strong {{ font-size:22px; }}
dl {{ display:grid; grid-template-columns:150px 1fr; gap:12px 18px; }} dt {{ color:var(--muted); }} dd {{ margin:0; }} .model-field-list {{ list-style:none; padding:0; margin:0; display:grid; gap:9px; }} .model-field-list li {{ border-left:2px solid var(--cyan); padding-left:10px; }} .model-field-list span {{ display:block; color:var(--cyan); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }} .feature-cols {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }} .data-facts {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:12px 0 18px; }} .data-facts div {{ border:1px solid rgba(34,230,255,.15); border-radius:14px; background:rgba(5,11,22,.34); padding:12px; }} .data-facts span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }} .data-facts strong {{ color:var(--text); font-size:13px; overflow-wrap:anywhere; }} .data-panel .stats-table {{ margin-top:8px; font-size:13px; }}
.metric-scorecards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin-bottom:22px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:11px; border-bottom:1px solid rgba(34,230,255,.12); }} th {{ color:var(--cyan); font-family:Space Grotesk; }}
.media-grid {{ display:grid; gap:18px; }} .diagrams-grid {{ grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); }} .result-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} figure {{ margin:0; border:1px solid rgba(34,230,255,.18); background:#07111f; border-radius:18px; padding:12px; overflow:hidden; }} figcaption {{ color:var(--cyan); font-weight:800; margin-bottom:8px; font-family:Space Grotesk; }} .figure-caption {{ color:var(--muted); font-size:13px; margin:.7rem 0 0; }} .svg-wrap svg {{ width:100%; height:auto; display:block; border-radius:12px; }} .themed-svg {{ background:#07111f; }} pre {{ white-space:pre-wrap; font-family:Inter; color:var(--text); line-height:1.55; }}
.metric-defs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; }} .metric-card {{ border:1px solid rgba(155,108,255,.28); border-radius:16px; padding:15px; background:rgba(2,6,23,.45); overflow:hidden; }} .metric-card h4 {{ margin:0 0 8px; color:var(--cyan); }} .equation {{ color:var(--gold); font-family:JetBrains Mono, monospace; font-size:14px; overflow-x:auto; max-width:100%; padding-bottom:4px; }} .next-card {{ border-color:rgba(214,255,77,.35); }}
.operator-mode {{ border-color:rgba(214,255,77,.28); background:linear-gradient(135deg,rgba(214,255,77,.06),rgba(34,230,255,.05)); }} .operator-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }} .operator-explain {{ display:inline-block; vertical-align:baseline; margin:0 2px; }} .operator-explain summary {{ list-style:none; cursor:pointer; color:var(--gold); border:1px solid rgba(214,255,77,.36); border-radius:999px; padding:2px 8px; background:rgba(214,255,77,.07); font-weight:700; }} .operator-explain summary::-webkit-details-marker {{ display:none; }} .operator-explain[open] {{ display:block; margin:8px 0; }} .operator-explain[open] summary {{ border-radius:12px 12px 0 0; }} .operator-explain p {{ margin:0; padding:10px 12px; color:var(--text); border:1px solid rgba(214,255,77,.24); border-top:0; border-radius:0 0 12px 12px; background:rgba(5,11,22,.72); line-height:1.45; }} .operator-card {{ display:block; margin:0; }}
.semantic-state {{ color:var(--cyan)!important; }} .semantic-action {{ color:var(--orange)!important; }} .semantic-pass {{ color:var(--green)!important; }} .semantic-metric {{ color:var(--purple)!important; }} .semantic-risk {{ color:var(--red)!important; }} .semantic-best {{ color:var(--gold)!important; }} .empty {{ color:var(--muted); padding:12px; }} .sr-only {{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }}
@media(max-width:1000px) {{ .shell {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; height:auto; }} .grid.two,.verdict-card,.verdict-grid,.feature-cols,.result-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="shell"><aside class="sidebar"><div class="brand">{logo}<div class="brand-title">SciQuest Dashboard</div></div><p class="muted">Status: {status}<br>Best score: <span class="mono semantic-best">{best}</span></p><nav class="experiment-timeline">{nav}</nav></aside><main><section class="hero"><div class="cycle-strip"><span>Hypothesize</span><span>Model</span><span>Validate</span><span>Iterate</span></div><p class="eyebrow">Quest</p><h1>{title}</h1><h2>{hero}</h2><p>{problem}</p><span class="sr-only">{semantic_classes}</span></section>{operator_glossary}{sections}{metric_definitions}{model_abstraction}</main></div>
<script>
function showExperiment(id) {{ document.querySelectorAll('.experiment').forEach(el => el.classList.toggle('active', el.id === id)); document.querySelectorAll('.exp-tab').forEach(el => el.classList.toggle('active', el.dataset.expTarget === id)); history.replaceState(null, '', '#' + id); }}
window.addEventListener('DOMContentLoaded', () => {{ const requested = location.hash ? location.hash.slice(1) : null; if (requested && document.getElementById(requested)) showExperiment(requested); }});
</script>
</body>
</html>"""

METRIC_DEFINITIONS_HTML = r"""
<section class="panel" id="metric-definitions">
  <h3>Validation Metric Definitions</h3>
  <div class="metric-defs">
    <article class="metric-card"><h4>Weighted aggregate score</h4><p>Normalized metric scores are combined with user/agent-defined weights.</p><p class="equation">\(S = \sum_i w_i s_i / \sum_i w_i\)</p></article>
    <article class="metric-card"><h4>WAPE</h4><p>Weighted absolute percentage error. Lower is better for forecast/counterfactual error.</p><p class="equation">\(WAPE = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i |y_i|}\)</p></article>
    <article class="metric-card"><h4>Relative WAPE lift</h4><p>Improvement versus a baseline simulator or model. Higher is better.</p><p class="equation">\(Lift = (WAPE_{baseline} - WAPE_{model}) / WAPE_{baseline}\)</p></article>
    <article class="metric-card"><h4>RMSE</h4><p>Root mean squared error, used here for demand/unit prediction error. Lower is better.</p><p class="equation">\(RMSE = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}\)</p></article>
    <article class="metric-card"><h4>Rank correlation</h4><p>Spearman correlation of true and predicted price-action rankings. Higher is better.</p><p class="equation">\(\rho = corr(rank(y), rank(\hat{y}))\)</p></article>
    <article class="metric-card"><h4>Law-of-demand pass rate</h4><p>Share of comparable states where predicted demand is non-increasing as price rises.</p><p class="equation">\(PassRate = \frac{1}{|G|}\sum_g 1[\hat{d}_{g,p_1} \geq \hat{d}_{g,p_2}\;\forall p_1&lt;p_2]\)</p></article>
  </div>
</section>
"""
