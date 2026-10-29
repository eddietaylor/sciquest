"""Prospective public software rules, separate from scientific model choices.

Schemas describe shape; explicit runtime and cross-field rules below complete the
contract. They are disclosure, not a second validator or a change to admission.
"""

from __future__ import annotations

from copy import deepcopy

from . import fidelity_v1

VERSION = "instrument-public-software-api-v1.0"


def path_rules():
    return dict(
        encoding="UTF-8", max_bytes=512,
        component_fullmatch=r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        rules="Canonical relative POSIX names only, slash-separated matching components; no empty component, absolute path, '.', '..', backslash or NUL. Referenced files must exist as regular, non-symlink, non-hardlinked files; bounded stable reads apply.")


def fitted_state_contract(*, source_revision, probe_sha256, artifact_limits, archive_limits, policy):
    path = dict(type="string", minLength=1, maxLength=512)
    nonempty = dict(type="string", minLength=1)
    family = dict(type="object", properties=dict(
        id=nonempty, basis=dict(type="array", minItems=0, maxItems=128,
                               uniqueItems=True, items=nonempty)),
        required=["id", "basis"], additionalProperties=False)
    array = dict(type="object", properties=dict(
        file=path, key=nonempty, family_id=nonempty,
        shape=dict(type="array", minItems=1, maxItems=4, items=dict(type="integer", minimum=1)),
        axes=dict(type="array", minItems=1, maxItems=4, uniqueItems=True, items=nonempty),
        basis_axis=dict(type="integer", minimum=-1, maximum=3),
        role=dict(type="string", enum=["parameters", "particle_weights", "other"])),
        required=["file", "key", "family_id", "shape", "axes", "basis_axis", "role"],
        additionalProperties=False)
    props = dict(
        version=dict(type="string", const=fidelity_v1.VERSION),
        instance_id=dict(type="string", minLength=1, maxLength=128),
        source_revision=dict(type="integer", const=source_revision),
        training_cutoff=dict(type="integer", minimum=1, maximum=source_revision),
        families=dict(type="array", minItems=1, maxItems=16, items=family),
        arrays=dict(type="array", minItems=0, maxItems=32, items=array),
        model_files=dict(type="array", minItems=1, maxItems=16, uniqueItems=True, items=path),
        probe_file=path, live_reference=path, reload_program=path,
        measurement_assumptions=nonempty,
    )
    result_props = dict(
        instance_id=deepcopy(props["instance_id"]),
        source_revision=deepcopy(props["source_revision"]),
        training_cutoff=deepcopy(props["training_cutoff"]),
        probe_sha256=dict(type="string", const=probe_sha256),
        families=deepcopy(props["families"]),
        component_weights=dict(type="array", minItems=1, maxItems=16,
                               items=dict(type="number", minimum=0)),
        probe_ids=dict(type="array", minItems=1, maxItems=128, uniqueItems=True, items=nonempty),
        targets=dict(type="array", minItems=1, maxItems=16, uniqueItems=True, items=nonempty),
        predictions=dict(type="array", minItems=1, maxItems=16, items=dict(
            type="array", minItems=1, maxItems=128, items=dict(
                type="array", minItems=1, maxItems=16,
                items=dict(type="number", minimum=-1e12, maximum=1e12)))),
    )
    return dict(
        contract_version=VERSION, version=fidelity_v1.VERSION,
        schema_scope="Structural JSON schemas PLUS every runtime/cross-field rule below; schema validation alone is not complete admission.",
        declaration_fields=sorted(props), family_fields=list(family["required"]),
        array_fields=list(array["required"]), array_roles=list(array["properties"]["role"]["enum"]),
        probe_result_fields=list(result_props),
        declaration_schema=dict(type="object", properties=props, required=list(props), additionalProperties=False),
        probe_result_schema=dict(type="object", properties=result_props,
                                 required=list(result_props), additionalProperties=False),
        source_revision=source_revision, probe_file="probes.csv",
        exact_integer_rule="source_revision, training_cutoff, each shape dimension and basis_axis must decode as Python type int: JSON7 is accepted; true,7.0 and \"7\" are rejected. No coercion.",
        declaration_rules=[
            "Use exactly declaration_fields, with no unknown fields, and the stated version.",
            "instance_id is scientist-chosen, nonblank after Python str.strip(), at most128 characters; no hidden owner ID is required.",
            "source_revision equals the supplied current integer r above/context.json revision, not an analysis-start or publication revision. training_cutoff is a declared integer evidence revision in1..r, not an observation ID/count or descriptive provenance object.",
            "measurement_assumptions is a nonblank string after str.strip(); lists/objects are invalid. This is your own model declaration, not a supplied apparatus explanation.",
            "families has1..16 entries, each exactly id,basis. Family ids are nonblank strings unique across families; basis has0..128 nonblank strings unique within that family, in declared order. Empty basis is allowed.",
            "model_files has1..16 unique safe relative names. probe_file, live_reference, reload_program and all model_files are pairwise distinct; the declaration name is distinct from every asset.",
            "arrays has0..32 entries; empty is allowed. Each entry has exactly array_fields. file names a model_file ending in case-sensitive .npz; key is a nonempty string. Only array-referenced files must be NPZ; other model_files may be opaque formats.",
            "Each array family_id names an existing family; shape has1..4 positive exact-int dimensions and product<=1000000. axes has the same length as shape, with unique nonempty strings (whitespace-only key/axis names are not additionally forbidden).",
            "Each (file,key) may occur once. basis_axis is exact-int -1 or an index in shape; when nonnegative, shape[basis_axis] equals that family's basis length. The same axis label in the same family always has the same width across all arrays.",
            "role is parameters,particle_weights,or other. particle_weights requires shape length1 and axes exactly ['particle']; values are finite nonnegative and sum to1 within host tolerance.",
        ],
        path_rules=path_rules(),
        array_payload_rules=dict(
            loading="Bounded structural worker uses numpy.load(...,allow_pickle=False); supervisor does not deserialize models.",
            dtype_kinds=["boolean", "signed_integer", "unsigned_integer", "floating"],
            finite=True, declared_shapes_exact=True, max_total_declared_array_bytes=16*1024*1024),
        result_rules=[
            "Use exactly probe_result_fields and no unknown fields. instance_id,source_revision,training_cutoff,families exactly copy declaration types, values and order.",
            "probe_sha256 is the actual unchanged probe CSV raw-byte SHA256. probe_ids has1..128 unique nonempty strings in CSV row order. targets has1..16 unique nonempty strings naming your compared quantities; there is no required scientific target vocabulary. Whitespace-only probe IDs/target strings are not additionally forbidden.",
            "component_weights has one finite nonnegative JSON int/float per declared family, excluding bool; its sum is1 within the stated host tolerance.",
            "predictions is component x probe x target, with lengths matching families,probe_ids,targets. Every entry is a finite JSON int/float, excluding bool, with absolute value<=1e12.",
            "The earlier reference and reload must have exactly matching ordered probe_ids/targets and component weights/predictions within host tolerance. The earlier reference's IDs must equal the actual supplied CSV row order.",
        ],
        reload_call=dict(
            language="Python", argv={"1":"probe CSV relative path", "2":"required result JSON relative path"},
            working_directory="fresh checker root, not the reload program's parent directory",
            staged_inputs=["declaration at its declared relative name", "model_files", "probe_file", "reload_program"],
            relative_names_preserved=True, live_reference_staged=False,
            write_result="Write the probe_result_schema JSON to sys.argv[2]; read probes from sys.argv[1]. Do not depend on internal checker filenames, package source, owner files, training data or the earlier live reference.",
            reserved_path_names=list(fidelity_v1.RESERVED_CHECKER_PATHS),
            reserved_path_rule="A staged input's whole name or first path component cannot equal a listed reserved name. These are exclusion rules, not filenames your program should read or invoke."),
        instrument_publication=dict(
            declaration_path="outputs/ relative name, explicitly selected by operation.fitted_bundle",
            required_outputs="Explicitly list the declaration, every model_file, live_reference and reload_program under outputs/ in operation.outputs. probe_file must be the supplied unchanged probes.csv, not an outputs/ asset.",
            operation_max_outputs=16, operation_max_input_ids=64,
            artifact_limits=deepcopy(artifact_limits), archive_limits=deepcopy(archive_limits),
            retained_quotas_scope="Retained artifact quotas are not an arbitrary temporary-disk quota.",
            analysis_timeout_seconds=policy["analysis_timeout_seconds"],
            fidelity_timeout_seconds=policy["fidelity_timeout_seconds"],
            sandbox_file_write_limit_bytes=1024*1024,
            checker_json_and_input_max_bytes=fidelity_v1.MAX_BYTES,
            checker_address_space_bytes=512*1024*1024),
        failure_layers=[
            "Invalid operation before offline launch: format_error, consumes a campaign-global correction; no offline launch.",
            "Invalid declaration or publication after successful offline execution: publication_failed/offline_outcome, consumes one shared offline launch and no format correction; no publication.",
            "Valid declaration but failed bounded fidelity check: may still publish explicit opaque outputs with analysis status succeeded and fidelity.status failed, verified_quantities empty and no checked handoff eligibility.",
        ],
        tolerance=dict(absolute=fidelity_v1.ABS_TOLERANCE, relative=fidelity_v1.REL_TOLERANCE,
                       comparison="math.isclose for each weight/prediction; host-owned, not user-adjustable"),
        quantity_scope="Only declared structure, exact input bytes and named quantities at the finite supplied probes. The reference is the scientist's own earlier witness, not independently established truth. No arbitrary model intent, scientific validity, posterior calibration, omitted uncertainty/dependence/transform or full-distribution certification.",
    )


def final_plan_schema(schema):
    """Clone the instrument's plan schema; leave the legacy apparatus unchanged."""
    result = deepcopy(schema)
    result["properties"]["model_export"] = dict(
        anyOf=[dict(type="null"), dict(type="object", required=["artifacts", "reload_notebook"],
            properties=dict(
                artifacts=dict(type="array", minItems=1, items=dict(type="string", minLength=1)),
                reload_notebook=dict(type="string", minLength=1),
                absolute_tolerance=dict(type="number", minimum=0)), additionalProperties=True)],
        description="Optional; omitted/null preserves basic new_or_unverified delivery. Opting in requires BOTH fields and generated model assets plus a separate valid reload notebook, described by native_model_export_contract.")
    return result


def native_model_export_contract():
    return dict(
        contract_version=VERSION, optional=True, no_export=["omitted", "null"],
        required_if_object=["artifacts", "reload_notebook"],
        artifacts="Nonempty list of nonempty string relative file paths. The unchecked route adds no uniqueness or maximum-count rule. Initial plan validation requires contained paths, not absolute or containing '..'; artifact paths cannot be staged data/..., prediction CSV paths or the reload_notebook path. Subsequent real preflight capture additionally applies capture_path_rules to EVERY artifact AND reload_notebook name.",
        reload_notebook="Nonempty contained relative file path to VALID Jupyter notebook CONTENTS, also satisfying capture_path_rules. No filename extension is required. The ORIGINAL prediction notebook must CREATE this file and all exported artifacts; supplying a plan path or a Python checker reload_program alone does not create a notebook.",
        capture_path_rules=path_rules(),
        path_validation_layers="Initial plan admission alone does not establish valid capture paths. For example, a space-containing name or './name' can pass the contained-path plan predicate but fails canonical ASCII-component capture after notebook execution. This remains a public preflight/format outcome; no path check is moved earlier. Each captured file is additionally bounded to1048576 raw bytes and must be regular, non-symlink and non-hardlinked.",
        original_inputs="Use the actual prediction_files paths and validation_suite in this context. Original prediction receives public training (if any), evaluation features and the selected campaign archive.",
        clean_inputs="Features and exported artifact files at their relative paths, plus validation.yaml and experiment_plan.yaml; no training or campaign archive. The generated reload notebook is copied to notebook.ipynb for preflight and reload.ipynb for native clean replay. CWD is the fresh workspace root.",
        output="Recompute prediction_files.output for every evaluation ID with id,mean_demand,mean_sales,lower_sales,upper_sales. Native reload is a notebook, distinct from the fitted checker's Python reload_program with probe/result argv.",
        absolute_tolerance=fidelity_v1.ABS_TOLERANCE,
        tolerance_rule="If supplied, absolute_tolerance must initially be finite and nonnegative; the instrument then replaces it with host-fixed1e-8. It cannot loosen parity.",
        host_owned_fields=["checked_fitted_export", "instrument_fitted_state"],
        host_metadata_rule="Do not submit checked_fitted_export. The instrument binds successful earlier check identity/hash, campaign/analysis, exact selected artifact hashes and finite scope; it adds instrument_fitted_state to the plan.",
        checked_route="Nonempty predict.fitted_bundle must select the earlier successful declaration and all its checked model files plus checker reload_program. model_export.artifacts must name exactly those selected model files AND checker reload_program using artifacts/ID paths, with unique entries. Export their exact bytes. The separate native reload_notebook is outside that list. A failed finite check never qualifies.",
        scope="Reload parity and selected-asset byte identity are software checks; arbitrary predictor use, model meaning, inference, calibration and scientific truth remain unverified.",
    )
