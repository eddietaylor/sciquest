"""Public apparatus syntax. Contains no world equations, priors or answers."""

from copy import deepcopy


def action_schemas(public, observations=()):
    replicate_ids = [o["id"] for o in observations if o["kind"] != "measure"]
    measure_ids = [
        o["id"] for o in observations if o["kind"] in ("observe", "intervene")
    ]
    precision = dict(type="string", enum=["standard", "precise"], default="standard")
    controls = dict(
        context=dict(type="number", enum=public["contexts"], default=0),
        precision=precision,
    )
    fields = {
        "observe": controls,
        "replicate": dict(observation_id=dict(type="string", enum=replicate_ids)),
        "measure": dict(observation_id=dict(type="string", enum=measure_ids)),
    }
    if public["interventions_available"]:
        fields["intervene"] = dict(
            **controls,
            price=dict(type="number", enum=public["prices"]),
            capacity=dict(type="number", enum=public["capacities"], default=180),
        )
    if "calibration_references" in public:
        fields["calibrate"] = dict(
            reference=dict(type="number", enum=public["calibration_references"]),
            precision=precision,
        )
    required = dict(
        observe=[],
        intervene=["price"],
        replicate=["observation_id"],
        measure=["observation_id"],
        calibrate=["reference"],
    )
    return {
        kind: dict(
            type="object",
            properties=dict(kind=dict(type="string", const=kind), **deepcopy(props)),
            required=["kind", *required[kind]],
            additionalProperties=False,
        )
        for kind, props in fields.items()
    }


def plan_schema(metrics, *, collecting=False):
    text = dict(type="string", minLength=1)
    fields = {
        k: text
        for k in (
            "question",
            "hypothesis",
            "scope",
            "assumptions",
            "data_access",
            "stopping_rule",
        )
    }
    fields.update(
        action=dict(
            type="string",
            enum=[
                "hypothesis_test",
                "rival_test",
                "replication",
                "measurement",
                "data_acquisition",
            ],
        ),
        confirmatory_status=dict(
            type="string",
            enum=["exploratory"] if collecting else ["exploratory", "confirmatory"],
        ),
        predictions=dict(
            type="array",
            minItems=1,
            items=dict(
                type="object",
                properties=dict(
                    id=text,
                    metric=dict(type="string", enum=list(metrics)),
                    operator=dict(type="string", enum=["<", "<=", ">", ">=", "=="]),
                    threshold=dict(type="number"),
                ),
                required=["id", "metric", "operator", "threshold"],
                additionalProperties=False,
            ),
        ),
    )
    return dict(
        type="object",
        properties=fields,
        required=list(fields),
        additionalProperties=True,
    )


SEMANTICS = """An observe request resamples historical assignment at the requested
context; it cannot fix historical price or capacity. Replicate repeats the original
request on a fresh unit at its original precision: interventions keep their controls,
but observe replications redraw historical assignment and may return another price.
Replicate and measure accept only kind and observation_id; do not add precision.
Measure audits an original sales unit at the apparatus audit precision, with paired
dependence; calibration readings have no demand unit to audit. Reference calibration
measures a known reference. Instrument errors and response laws remain unknown."""
