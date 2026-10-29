"""Minimal subprocess entry point. This process never receives supervisor write access."""

from __future__ import annotations

import os
from pathlib import Path
import resource
import sys
import json
import tempfile


def notebook(source: Path, output: Path, cwd: Path, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    nb = nbformat.read(source, as_version=4)
    nbformat.validate(nb)
    if not any(cell.cell_type == "code" and cell.source.strip() for cell in nb.cells):
        raise ValueError("Notebook contains no executable cells")
    with tempfile.TemporaryDirectory(prefix="sciquest-kernelspec-") as temp:
        spec = Path(temp) / "sciquest"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "SciQuest registered interpreter",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="sciquest",
            kernel_spec_manager=KernelSpecManager(
                kernel_dirs=[temp], ensure_native_kernel=False
            ),
        )
        client = NotebookClient(
            nb, km=manager, timeout=timeout, resources={"metadata": {"path": str(cwd)}}
        )
        try:
            client.execute()
        finally:
            # Failed cells and earlier output are evidence too.
            output.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, output)


def main() -> None:
    if sys.argv[1] == "command":
        limit = int(sys.argv[2])
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        command = sys.argv[4:]
        if not command:
            raise ValueError("Missing worker command")
        os.execvpe(command[0], command, dict(os.environ))
    elif sys.argv[1] == "notebook":
        notebook(Path(sys.argv[2]), Path(sys.argv[3]), Path.cwd(), int(sys.argv[4]))
    else:
        raise ValueError("Unknown worker task")


if __name__ == "__main__":
    main()
