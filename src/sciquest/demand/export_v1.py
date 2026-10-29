"""Byte binding for prospective checked fitted-state native exports.

No model deserialization and no inference about how an arbitrary predictor uses
its assets. Absence of this optional contract preserves legacy export behavior.
"""

from .artifacts_v1 import _read_regular, relative_name, sha

VERSION = "checked-fitted-export-v1.0"


def verify_export(workspace, export):
    commitment = export.get("checked_fitted_export")
    if commitment is None:
        return None
    if (
        not isinstance(commitment, dict)
        or set(commitment) != {"version", "campaign_id", "analysis_id", "check_sha256",
                               "artifact_sha256", "scope"}
        or commitment["version"] != VERSION
        or commitment["scope"] != "exact_declared_assets_and_finite_probes"
        or not isinstance(commitment["artifact_sha256"], dict)
        or not isinstance(export.get("artifacts"), list)
        or not export["artifacts"]
        or len(set(export["artifacts"])) != len(export["artifacts"])
        or set(commitment["artifact_sha256"]) != set(export["artifacts"])
    ):
        raise ValueError("Invalid checked fitted export contract")
    for name, digest in commitment["artifact_sha256"].items():
        relative_name(name)
        if sha(_read_regular(workspace, name, 1048576)) != digest:
            raise ValueError("Native export differs from selected checked fitted asset: " + name)
    return commitment
