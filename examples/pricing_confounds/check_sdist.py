"""Verify that the source archive preserves the complete published example."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def check(path):
    with tarfile.open(path) as archive:
        names = archive.getnames()
        prefix = names[0].split("/")[0] + "/"
        root = prefix + "docs/examples/pricing-confounds/run/"
        manifest = json.load(archive.extractfile(root + "manifest.json"))
        reference = json.loads((Path(__file__).resolve().parents[2] /
                                "docs/examples/pricing-confounds/run/manifest.json").read_text())
        if manifest != reference:
            raise ValueError("Archive manifest differs from the published source manifest")
        for relative, expected in manifest["sha256"].items():
            data = archive.extractfile(root + relative).read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source archive altered published evidence: {relative}")
        for relative in ["examples/pricing_confounds/run.py",
                         "examples/pricing_confounds/test_end_to_end.py",
                         "examples/pricing_confounds/requirements.txt"]:
            archive.getmember(prefix + relative)
        if any("/__pycache__/" in name or name.endswith(".pyc") for name in names):
            raise ValueError("Source archive contains Python caches")
        return len(manifest["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    args = parser.parse_args()
    print(f"Verified {check(args.archive)} saved-run hashes inside the source archive.")
