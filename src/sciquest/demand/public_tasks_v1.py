"""Fixed public engineering tasks, never arbitrary owner instructions or truth.

Task fulfillment is a separate prospective audit. Registering a task does not
change legal operations, optional default delivery, resource rules or scientific
admission; the actor must use the ordinary iterative instrument.
"""

from __future__ import annotations

from sciquest.io import canonical_json, content_hash

VERSION = "registered-public-engineering-task-v1.0"
CHECKED_HANDOFF_ID = "checked-fitted-handoff-v1"
MAX_TASK_BYTES = 8192

_CHECKED_HANDOFF_TEXT = """Perform a software preservation exercise using your own public observations and chosen working model. Collect at least one legal observation through this campaign's query operation. Using only that public evidence and your own assumptions, create and save your fitted working state. Explain what your named finite forecast quantities mean, and save an earlier witness computed from your working state before relying on serialization.

Use the emitted fitted_state_contract and unchanged public probes to obtain a successful fitted-state check through a legal analysis or prepare operation. The Python checker reload program must recompute the declared quantities from saved files without reading the earlier witness. Inspect the actual returned software feedback; authoring a declaration, printing local agreement or publishing opaque bytes alone is not a successful check.

In a later prediction submission, select that earlier successful declaration and its exact checked model files and loader. Consume the saved state for your actual forecasts. Reconstructing executable predictor objects from the SAME frozen fitted values and basis is allowed. Do not re-estimate from observations, change the fitted state, use code search to recreate a missing fit, or ignore the saved state. A data-derived constant working forecast can satisfy this requirement if it actually consumes the preserved fitted state; no particular scientific model is prescribed. Have the original prediction notebook create the exact exported assets and the separate native reload notebook, following native_model_export_contract, and complete native delivery and clean reload. Use the ordinary scientific-decision and prediction flow; stopping still freezes scientific conclusions and does not provide new observations.

Choose your own model, fitted representation and assumptions. Do not fabricate uncertainty or claim that software parity identifies the hidden world, validates calibration or establishes scientific truth. This task supplies no physical law, coefficients, likelihood, rival, reference program, serialized answer or owner scientific critique. The requirements concern this opt-in engineering exercise; ordinary campaigns without this task retain their optional fitted-state route."""


def get_task(task_id):
    """Resolve only the fixed public ID or None; return a fresh bounded object."""
    if task_id is None:
        return None
    if type(task_id) is not str or task_id != CHECKED_HANDOFF_ID:
        raise ValueError("public_task must be None or the registered ID " + CHECKED_HANDOFF_ID)
    task = dict(version=VERSION, id=CHECKED_HANDOFF_ID, text=_CHECKED_HANDOFF_TEXT)
    if len(canonical_json(task).encode("utf-8")) > MAX_TASK_BYTES:
        raise RuntimeError("Registered public task exceeds its fixed object bound")
    return task


def verify_frozen(task, task_sha256):
    """Reject drift or caller-authored text, including changed-and-rehashed text."""
    if task is None:
        if task_sha256 is not None:
            raise ValueError("Absent public_task requires an absent public_task_sha256")
        return None
    if not isinstance(task, dict) or set(task) != {"version", "id", "text"}:
        raise ValueError("Frozen public_task must be the exact registered version/id/text object")
    expected = get_task(task["id"])
    if expected is None or task != expected or task_sha256 != content_hash(expected):
        raise ValueError("Frozen public task content, registry identity or canonical hash changed")
    return expected
