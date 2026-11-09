"""Fourier Occupancy Field (FOF) encoding and decoding.

    mesh_to_fof      meshes -> FOF coefficients [B, C, H, W]          (encode.py)
    render_fof       FOF + front/back depth and normal maps            (encode.py)
    rasterize        exact per-pixel lists of all surface crossings    (encode.py)
    ray_diagnostics  per-ray defect report of a ground-truth mesh       (encode.py)
    fof_to_occupancy FOF -> occupancy volume [B, D, H, W]              (decode.py)
    fof_to_mesh      FOF -> triangle mesh (optionally "lossless")      (decode.py)
    volume_to_mesh   occupancy volume -> triangle mesh                 (decode.py)
    marching_cubes   crack-free marching cubes on tensors (GPU)        (mc.py)
    orient_outward   offline orientation repair of ground-truth meshes  (repair.py)

``FOF``, ``FOF_Normal`` and ``render`` keep the original call signatures.
"""

from .decode import fof_basis, fof_to_mesh, fof_to_occupancy, volume_to_mesh
from .encode import (
    FILL_RULES,
    FOFRender,
    Fragments,
    decode_keys,
    fragments_to_fof,
    mesh_to_fof,
    rasterize,
    ray_diagnostics,
    render_fof,
)
from .legacy import FOF, FOF_Normal, render
from .mc import marching_cubes
from .repair import orient_outward

__all__ = [
    "FILL_RULES",
    "FOF",
    "FOFRender",
    "FOF_Normal",
    "Fragments",
    "decode_keys",
    "fof_basis",
    "fof_to_mesh",
    "fof_to_occupancy",
    "fragments_to_fof",
    "marching_cubes",
    "mesh_to_fof",
    "orient_outward",
    "rasterize",
    "ray_diagnostics",
    "render",
    "render_fof",
    "volume_to_mesh",
]
