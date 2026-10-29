"""Append-only custody for one assigned two-stage reviewer trajectory.

This records local release ordering, exact presented content and raw responses.
It cannot prove an external person's identity, prior knowledge, or honesty, or
resist an owner rewriting the entire ledger and every externally held commitment.
An interrupted/invalid attempt remains visible and cannot be silently retried.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import tempfile

from jsonschema import Draft202012Validator
from sciquest.io import utc_now
from . import assessment_v5 as a
from . import assessment_transport_v1 as transport
from .artifacts_v1 import _read_regular

VERSION = "claim-review-custody-v5.0"
TRANSPORT_VERSION = "claim-review-custody-v5.1"
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_ENTRIES = 16


def _write(path, raw):
    if len(raw) > MAX_FILE:
        raise ValueError("Assessment custody object exceeds byte bound")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage2_schema():
    truth = a.response_schema("truth_review",{"annotations","unreviewed_scope"})
    later = a.response_schema("later_critique",{
        "annotations","revision_relations","code_checks","unreviewed_scope"})
    definitions = {**truth.pop("$defs"),**later.pop("$defs")}
    return dict(type="object",additionalProperties=False,required=["truth_review","later_critique"],
                properties=dict(truth_review=truth,later_critique=later),
                **{"$defs":definitions,"$schema":a._DATA["$schema"]})


def _stage2(response, state):
    Draft202012Validator(stage2_schema()).validate(response)
    prep = state["prepared"]
    freeze,release = state["freeze"],state["release"]
    truth = dict(
        document_type=a.KINDS["truth_review"],reviewer=copy.deepcopy(prep["reviewer"]),
        public_freeze_hash=a.H(freeze),truth_release_hash=a.H(release),
        truth_packet_hash=a.H(state["truth_packet"]),**copy.deepcopy(response["truth_review"]),
    )
    later = dict(
        document_type=a.KINDS["later_critique"],reviewer=copy.deepcopy(prep["reviewer"]),
        public_freeze_hash=a.H(freeze),truth_release_hash=a.H(release),
        later_public_packet_hash=a.H(state["later_packet"]),
        target_manifest_hash=a.H(prep["public_packet"]["targets"]),
        exposure="private_informed",**copy.deepcopy(response["later_critique"]),
    )
    a.validate_truth_review(truth,state["truth_packet"],freeze,release,
                            public_review=state["public_review"],reviewer=prep["reviewer"])
    a.validate_later_critique(later,prep["public_packet"],state["later_packet"],freeze,
                              release=release,reviewer=prep["reviewer"])
    return truth,later


def _assembled_public(raw, prep):
    if "annotation_transport" in prep:
        return transport.assemble_public(raw, prep["public_packet"], reviewer=prep["reviewer"],
                                         rubric_hash=a.H(prep["rubric"]))
    return a.assemble_public_review(a.loads(raw), prep["public_packet"],
                                   reviewer=prep["reviewer"], rubric_hash=a.H(prep["rubric"])), None


def _assembled_stage2(raw, state):
    if "annotation_transport" not in state["prepared"]:
        truth, later = _stage2(a.loads(raw), state)
        return truth, later, None
    response, origin = transport.assemble_private_payload(raw, state["truth_packet"], state["later_packet"])
    truth, later = _stage2(response, state)  # Unchanged semantic validators.
    return truth, later, transport.seal_assembly(origin, dict(truth_review=truth, later_critique=later))


def _private_bundle(state, truth, later):
    prep = state["prepared"]
    a.validate_truth_packet(truth, public_packet=prep["public_packet"])
    a.validate_later_packet(prep["public_packet"], later)
    result = dict(frozen_public_review=state["public_review"], truth_packet=truth,
                  later_public_packet=later, rubric=copy.deepcopy(prep["rubric"]),
                  response_schema=prep["stage2_schema"],
                  exposure="private_informed; original meaning/support cannot be changed")
    if "annotation_transport" in prep:
        result.update(response_schema=transport.private_schema(truth, later),
                      annotation_transport=transport.stage2_contract(truth, later))
    return result


class ReviewLedger:
    def __init__(self,path):
        self.path = Path(path)

    @classmethod
    def create(cls,path,packet,*,rubric,reviewer,annotation_transport=None):
        transport.mode(annotation_transport)
        if packet["selection_rule"] == "first_nested_operation_complete_stop_v1;stop" and annotation_transport is None:
            raise ValueError("Complete-stop packets require the source-location annotation transport")
        a.actor_presentation(packet)
        a._shape("reviewer",reviewer)
        prepared = dict(
            version=VERSION,public_packet=copy.deepcopy(packet),rubric=copy.deepcopy(rubric),
            reviewer=copy.deepcopy(reviewer),public_schema=a.public_response_schema(),
            stage2_schema=stage2_schema(),recorded_at=utc_now(),
        )
        if annotation_transport is not None:
            prepared.update(version=TRANSPORT_VERSION,
                            annotation_transport=transport.public_contract(packet),
                            public_schema=transport.public_schema(packet), stage2_schema=None)
        raw = a.encoded(prepared)
        root = Path(path)
        root.mkdir(mode=0o700,parents=False,exist_ok=False)
        _write(root/"prepared.json",raw)
        _write(root/"lock",b"")
        (root/"entries").mkdir(mode=0o700)
        _sync(root)
        result = cls(root)
        result.verify()
        return result

    @contextmanager
    def _lock(self):
        fd = os.open(self.path/"lock",os.O_RDONLY|os.O_NOFOLLOW)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def verify(self):
        """Read bytes only. Never executes a model, subject program or SQL."""
        return self._verify()

    def verify_prefix(self,sequence,expected_head):
        """Authenticate an anchored prefix while unrelated later writes run.

        This deliberately does not certify any suffix, pending or completed.
        Full terminal audits must still call verify().
        """
        if type(sequence) is not int or not 0<=sequence<=MAX_ENTRIES:
            raise ValueError("Invalid anchored custody prefix length")
        result=self._verify(prefix_sequence=sequence)
        if result["head"]!=expected_head:
            raise ValueError("Anchored custody prefix changed")
        return result

    def _verify(self,*,prefix_sequence=None):
        if self.path.is_symlink() or set(p.name for p in self.path.iterdir()) != {"prepared.json","lock","entries"}:
            raise ValueError("Unknown/incomplete custody root")
        if _read_regular(self.path,"lock",0) != b"":
            raise ValueError("Changed custody lock")
        raw = _read_regular(self.path,"prepared.json",MAX_FILE)
        prep = a.loads(raw)
        expected_fields = {
            "version","public_packet","rubric","reviewer","public_schema","stage2_schema","recorded_at"
        }
        new = prep.get("version") == TRANSPORT_VERSION
        if new:
            expected_fields.add("annotation_transport")
        if raw != a.encoded(prep) or set(prep) != expected_fields or prep["version"] not in {VERSION, TRANSPORT_VERSION}:
            raise ValueError("Invalid prepared custody record")
        a.actor_presentation(prep["public_packet"])
        a._shape("reviewer",prep["reviewer"])
        if new:
            if (prep["annotation_transport"] != transport.public_contract(prep["public_packet"])
                or prep["public_schema"] != transport.public_schema(prep["public_packet"])
                or prep["stage2_schema"] is not None):
                raise ValueError("Prepared annotation transport/schema identity changed")
        elif prep["public_schema"] != a.public_response_schema() or prep["stage2_schema"] != stage2_schema():
            raise ValueError("Prepared reviewer schema changed")
        directory = self.path/"entries"
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Invalid custody entries directory")
        names = sorted(p.name for p in directory.iterdir())
        if prefix_sequence is None:
            if len(names)>MAX_ENTRIES or names != [f"e{i:06d}" for i in range(1,len(names)+1)]:
                raise ValueError("Custody gaps, pending writes or unknown entries")
        else:
            required=[f"e{i:06d}" for i in range(1,prefix_sequence+1)]
            if not set(required)<=set(names):
                raise ValueError("Anchored custody prefix is missing")
            names=required
        state = dict(prepared=prep,status="prepared",entries=[],head=a.raw_sha(raw),sequence=0)
        total = len(raw)
        for name in names:
            folder = directory/name
            if folder.is_symlink() or not folder.is_dir():
                raise ValueError("Invalid custody entry directory")
            entryraw = _read_regular(directory,name+"/entry.json",MAX_FILE)
            entry = a.loads(entryraw)
            if entryraw != a.encoded(entry) or set(entry) != {"sequence","previous","kind","payload","objects"}:
                raise ValueError("Invalid custody entry")
            if type(entry["sequence"]) is not int or entry["sequence"] != state["sequence"]+1 or entry["previous"] != state["head"]:
                raise ValueError("Custody sequence/previous commitment changed")
            objects = {}
            if not isinstance(entry["objects"],dict):
                raise ValueError("Invalid custody objects")
            for filename,commitment in entry["objects"].items():
                if not re.fullmatch(r"[a-z_]+\.(?:json|bin)",filename) or set(commitment) != {"sha256","bytes"}:
                    raise ValueError("Invalid custody object commitment")
                data = _read_regular(directory,name+"/"+filename,MAX_FILE)
                if commitment != dict(sha256=a.raw_sha(data),bytes=len(data)):
                    raise ValueError("Custody object raw bytes changed")
                objects[filename] = data
                total += len(data)
            total += len(entryraw)
            if total>MAX_TOTAL or set(p.name for p in folder.iterdir()) != {"entry.json",*objects}:
                raise ValueError("Unknown/oversized custody evidence")
            self._apply(state,entry,objects)
            state.update(sequence=entry["sequence"],head=a.raw_sha(entryraw))
            state["entries"].append(entry)
        return state

    def _apply(self,state,entry,objects):
        kind,payload = entry["kind"],entry["payload"]
        prep = state["prepared"]
        if kind in {"public_attempt","stage2_attempt"}:
            expected = "prepared" if kind=="public_attempt" else "truth_released"
            if state["status"] != expected or set(objects) != {"response.bin","receipt.bin"} or set(payload) != {"recorded_at"}:
                raise ValueError("Duplicate/out-of-order review attempt")
            # Raw response is intentionally not parsed before it is committed.
            state.update(status=kind,attempt=objects)
        elif kind in {"public_rejection","stage2_rejection"}:
            expected = "public_attempt" if kind=="public_rejection" else "stage2_attempt"
            if state["status"] != expected or objects or set(payload) != {"error","recorded_at"}:
                raise ValueError("Invalid review rejection")
            state["status"] = "rejected"
        elif kind=="public_freeze":
            new = "annotation_transport" in prep
            if state["status"] != "public_attempt" or set(objects) != ({"review.json", "assembly.json"} if new else {"review.json"}):
                raise ValueError("Freeze lacks its actual unique public attempt")
            review = a.loads(objects["review.json"])
            expected, assembly = _assembled_public(state["attempt"]["response.bin"], prep)
            if new and objects["assembly.json"] != a.encoded(assembly):
                raise ValueError("Public raw-to-derived assembly/origin changed")
            a._shape("freeze",payload)
            if review != expected or payload != self._freeze(state,review,entry["sequence"],entry["previous"],payload["recorded_at"]):
                raise ValueError("Freeze changed the actual public response or stage commitment")
            state.update(status="public_frozen",public_review=review,freeze=payload)
        elif kind=="truth_release":
            new = "annotation_transport" in prep
            required = {"truth.json", "later.json"} | ({"private_schema.json", "private_contract.json"} if new else set())
            if state["status"] != "public_frozen" or set(objects) != required:
                raise ValueError("Truth release requires an actual accepted public freeze")
            truth,later = a.loads(objects["truth.json"]),a.loads(objects["later.json"])
            a.validate_truth_packet(truth,public_packet=prep["public_packet"])
            a.validate_later_packet(prep["public_packet"],later)
            a._shape("release",payload)
            if payload != self._release(state,truth,entry["sequence"],entry["previous"],payload["recorded_at"]):
                raise ValueError("Truth release changed the actual preceding freeze")
            if new:
                expected_schema = transport.private_schema(truth, later)
                expected_contract = transport.stage2_contract(truth, later)
                if objects["private_schema.json"] != a.encoded(expected_schema) or objects["private_contract.json"] != a.encoded(expected_contract):
                    raise ValueError("Released stage2 schema/source-domain identity changed")
                state.update(stage2_schema=expected_schema, stage2_annotation_transport=expected_contract)
            state.update(status="truth_released",release=payload,truth_packet=truth,later_packet=later)
        elif kind=="stage2_accepted":
            new = "annotation_transport" in prep
            required = {"truth_review.json", "later_critique.json"} | ({"assembly.json"} if new else set())
            if state["status"] != "stage2_attempt" or set(objects) != required or payload != {"acceptance":"atomic_truth_and_private_informed_critique"}:
                raise ValueError("Invalid two-stage completion")
            truth, later, assembly = _assembled_stage2(state["attempt"]["response.bin"], state)
            expected = (truth, later)
            if new and objects["assembly.json"] != a.encoded(assembly):
                raise ValueError("Stage2 raw-to-derived assembly/origin changed")
            reviews = (a.loads(objects["truth_review.json"]),a.loads(objects["later_critique.json"]))
            if reviews != expected:
                raise ValueError("Stage2 annotations changed from their raw response")
            state.update(status="complete",truth_review=reviews[0],later_critique=reviews[1])
        else:
            raise ValueError("Unknown custody entry kind")

    @staticmethod
    def _freeze(state,review,sequence,previous,recorded_at):
        prep = state["prepared"]
        return dict(document_type=a.KINDS["freeze"],reviewer_id=prep["reviewer"]["id"],
                    public_packet_hash=a.H(prep["public_packet"]),
                    target_manifest_hash=a.H(prep["public_packet"]["targets"]),
                    rubric_hash=a.H(prep["rubric"]),public_review_hash=a.H(review),
                    presentation_sha256=review["presentation_sha256"],ledger_sequence=sequence,
                    previous_record_hash=previous,recorded_at=recorded_at)

    @staticmethod
    def _release(state,truth,sequence,previous,recorded_at):
        return dict(document_type=a.KINDS["release"],
                    reviewer_id=state["prepared"]["reviewer"]["id"],
                    public_freeze_hash=a.H(state["freeze"]),truth_packet_hash=a.H(truth),
                    ledger_sequence=sequence,previous_record_hash=previous,recorded_at=recorded_at)

    def _append(self,kind,payload,objects):
        state = self.verify()
        sequence = state["sequence"]+1
        if sequence>MAX_ENTRIES:
            raise ValueError("Custody entry quota exhausted")
        entry = dict(sequence=sequence,previous=state["head"],kind=kind,payload=payload,
                     objects={k:dict(sha256=a.raw_sha(v),bytes=len(v)) for k,v in objects.items()})
        # Validate semantics before beginning an atomic entry write. Attempts are
        # committed first, so even a later validation failure retains their bytes.
        self._apply(copy.deepcopy(state),entry,objects)
        pending = Path(tempfile.mkdtemp(prefix="pending-",dir=self.path/"entries"))
        for name,data in objects.items():
            _write(pending/name,data)
        _write(pending/"entry.json",a.encoded(entry))
        _sync(pending)
        pending.rename(self.path/"entries"/f"e{sequence:06d}")
        _sync(self.path/"entries")
        return self.verify()

    def public_bundle(self):
        state = self.verify()
        prep = state["prepared"]
        result = dict(presentation=a.actor_presentation(prep["public_packet"]),
                      rubric=copy.deepcopy(prep["rubric"]),response_schema=prep["public_schema"])
        if "annotation_transport" in prep:
            result["annotation_transport"] = copy.deepcopy(prep["annotation_transport"])
        return result

    def export_public_bundle(self,path):
        """Safe standalone bundle for a prospective human/agent reviewer."""
        root = Path(path)
        bundle = self.public_bundle()
        root.mkdir(mode=0o700,parents=False,exist_ok=False)
        for name,value in bundle.items():
            _write(root/(name+".json"),a.encoded(value))
        return root

    def submit_public(self,raw_response,*,raw_receipt=b""):
        with self._lock():
            if self.verify()["status"] != "prepared":
                raise ValueError("Public reviewer attempt already allocated/used")
            state = self._append("public_attempt",dict(recorded_at=utc_now()),
                                 {"response.bin":raw_response,"receipt.bin":raw_receipt})
            try:
                prep = state["prepared"]
                review, assembly = _assembled_public(raw_response, prep)
            except Exception as error:
                self._append("public_rejection",dict(error=str(error)[:2000],recorded_at=utc_now()),{})
                raise
            freeze = self._freeze(state,review,state["sequence"]+1,state["head"],utc_now())
            objects = {"review.json":a.encoded(review)}
            if assembly is not None:
                objects["assembly.json"] = a.encoded(assembly)
            return self._append("public_freeze",freeze,objects)

    def release_truth(self,truth_packet,later_packet):
        with self._lock():
            state = self.verify()
            if state["status"] != "public_frozen":
                raise ValueError("Truth cannot be released before an actual public freeze")
            payload = self._release(state,truth_packet,state["sequence"]+1,state["head"],utc_now())
            objects = {"truth.json":a.encoded(truth_packet),"later.json":a.encoded(later_packet)}
            if "annotation_transport" in state["prepared"]:
                objects.update({"private_schema.json":a.encoded(transport.private_schema(truth_packet, later_packet)),
                                "private_contract.json":a.encoded(transport.stage2_contract(truth_packet, later_packet))})
            return self._append("truth_release",payload,objects)

    def preview_stage2_bundle(self, truth_packet, later_packet):
        """Pure released-input construction after public freeze, before dispatch.

        The group barrier is external to one ledger. Actual release later binds
        these exact packets/schema/domain declarations before any private call.
        """
        state = self.verify()
        if "public_review" not in state:
            raise ValueError("Stage2 preview requires an actual public freeze")
        if "truth_packet" in state and (truth_packet != state["truth_packet"] or later_packet != state["later_packet"]):
            raise ValueError("Stage2 preview differs from actual released packets")
        return _private_bundle(state, truth_packet, later_packet)

    def stage2_bundle(self):
        state = self.verify()
        if state["status"] != "truth_released":
            raise ValueError("Stage2 requires an actual truth release")
        return _private_bundle(state, state["truth_packet"], state["later_packet"])

    def submit_stage2(self,raw_response,*,raw_receipt=b""):
        with self._lock():
            if self.verify()["status"] != "truth_released":
                raise ValueError("Stage2 attempt unavailable/already used")
            state = self._append("stage2_attempt",dict(recorded_at=utc_now()),
                                 {"response.bin":raw_response,"receipt.bin":raw_receipt})
            try:
                truth,later,assembly = _assembled_stage2(raw_response,state)
            except Exception as error:
                self._append("stage2_rejection",dict(error=str(error)[:2000],recorded_at=utc_now()),{})
                raise
            objects = {"truth_review.json":a.encoded(truth),"later_critique.json":a.encoded(later)}
            if assembly is not None:
                objects["assembly.json"] = a.encoded(assembly)
            return self._append("stage2_accepted",dict(acceptance="atomic_truth_and_private_informed_critique"),objects)
