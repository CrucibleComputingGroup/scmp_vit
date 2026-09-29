"""Make ``vit_sc/kernels`` the scmp_kernels that vit_sc imports.

vit_sc maintains its OWN ``scmp_kernels`` checkout at ``vit_sc/kernels`` (branch
``sc/precision-trace``). That copy must win over any other on the machine — in
this environment a *global* editable install points at ``scmp_cnn/kernels``
(branch ``main`` + local edits), so a bare ``import scmp_kernels`` from a neutral
cwd resolves to the CNN sibling's copy, which lacks trace.py / the ρ-metric
dispatch this repo relies on. Prepending our path fixes the resolution order.

Two rules inherited from the sibling repos, do not break them:

* The checkout directory MUST be named ``kernels/``, never ``scmp_kernels/``.
  A directory named ``scmp_kernels/`` next to the entry script shadows the real
  package as an empty PEP-420 namespace package (``__file__ = None``, no
  ``sc_matmul``) and SC silently degrades to FP.
* Import this module (or ``model.sc_vit``) before anything imports
  ``scmp_kernels``, otherwise the first importer wins.

Set ``VIT_SC_USE_INSTALLED_KERNELS=1`` to opt out and use whatever
``scmp_kernels`` is installed instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VIT_SC_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = VIT_SC_ROOT / "kernels"


def ensure_kernels_on_path() -> Path | None:
    """Prepend ``vit_sc/kernels`` to ``sys.path``. Returns the path used."""
    if os.environ.get("VIT_SC_USE_INSTALLED_KERNELS") == "1":
        return None
    if not (KERNELS_DIR / "scmp_kernels" / "__init__.py").is_file():
        raise RuntimeError(
            f"vit_sc kernels checkout missing at {KERNELS_DIR}. Restore it with:\n"
            f"  git clone git@github.com:CrucibleComputingGroup/scmp_kernels.git "
            f"{KERNELS_DIR} && git -C {KERNELS_DIR} checkout sc/precision-trace")
    p = str(KERNELS_DIR)
    if sys.path[:1] != [p]:
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return KERNELS_DIR


def kernels_provenance() -> str:
    """One-line description of which scmp_kernels actually got imported."""
    import scmp_kernels
    got = Path(scmp_kernels.__file__).resolve().parents[1]
    tag = "vit_sc-local" if got == KERNELS_DIR.resolve() else "EXTERNAL"
    return f"{tag}: {scmp_kernels.__file__}"


ensure_kernels_on_path()
