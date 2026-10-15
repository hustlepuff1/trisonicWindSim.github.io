3D full-volume Mach data preparation
====================================

Run from this directory with Python 3, NumPy, and SciPy:

  python3 extract_volume.py --allow-different-topology
  python3 derive_native_geometry.py
  python3 register_cached_volumes.py --reference-case M17

For a portable checkout, place the compact source registry at
results/source_cases.json and supply the raw-data directory explicitly:

  python3 extract_volume.py --allow-different-topology --raw-dir /path/to/cfd

An alternative registry can be passed with --cases-json /path/to/cases.json.
It has the same list format as results/cache/cases.json. --raw-dir changes only
the directory; raw filenames and required original SHA256 digests are retained.
The downstream consumers read the exact registry written in their cache.
The original framework/phase4_CFD/audit_reduced_volumes.py parser must accompany
these scripts at the same relative framework location. No original thesis or
external validation directory is needed when the portable registry is present.
The optional verify_cached_volumes.py check additionally requires the retained
2D research extracts and is not part of the minimal portable rebuild.

The geometry and registration consumers also accept --wait and may run while
the native extractor completes later cases. The final extraction_manifest.json
must have complete=true and topology_correspondence_verified=true before model
training. Partial memory-mapped arrays are deliberately not training-ready.

Source membership
-----------------
The 27 canonical training cases and their (L_mm, P0_Pa) inputs come from the
frozen v05 training_dataset.csv and dataset_provenance.json. Both metadata hashes
are checked against the frozen manifest. M175 and M325 are appended as held-out
rows from validation/raw_source_index.json and actual_inputs_confirmed.json;
they never become training rows. The complete original raw-file SHA256 is
checked against the archived expected digest for every case. Raw CFD files
remain at their external source locations and are not copied.

Every source has 3,639,168 nodes, 10,705,806 quadrilateral faces, and 3,533,741
hexahedral cells. All original cells, including the downstream diffuser, are
retained. The domain extends approximately from x=-5.8412 m to x=8.42 m.
The native CFD domain is one quarter; reflection at y=0 and z=0 is reserved for
visualization and does not multiply the training snapshots.

Why registration is necessary
-----------------------------
The raw face-node, left-cell, and right-cell connectivity hashes differ between
cases despite equal counts. Native cell indices therefore cannot be compared
directly. register_volume.py recovers a complete material-coordinate lattice
from mesh-edge distances to the inlet and the two symmetry boundaries:

  nodes: (702, 72, 72)
  cells: (701, 71, 71)
  flattened cell index: (i * 71 + j) * 71 + k

For every case the algorithm verifies a bijection for all nodes and all cells,
unit quadrilateral faces, six-face unit hexahedra, and correct neighbouring-cell
adjacency. Registration is an exact permutation of the stored float32 Mach
values; it performs no field interpolation or deletion. Coordinates are parsed
and geometrically audited as float64. This is a common material mesh for the
moving geometry, not a claim that all cases share the same physical coordinates.
In particular, a constant logical i section is not always a constant physical
x plane; x can vary across a logical section.

Reference physical volumes
--------------------------
The ordinary source exports round coordinates enough to collapse some very
small boundary-layer cells. No volume floor or coordinate repair is applied.
native/<case>/geometry.json and valid_geometry.npy record these defects; unsigned
tetrahedral volume diagnostics for invalid cells must not be treated as physical
quadrature weights. The Mach values of those cells remain in the training data.

The M17 training export retains more coordinate digits. Its complete mesh has
positive volumes for all 3,533,741 cells, no collapsed faces, no inconsistent
interior face orientations, and a maximum relative surface-closure residual of
about 1.76e-13. Its total quarter-domain volume is 2.426060656578175 m^3. M17 is
therefore used as the fixed physical reference quadrature for weighted POD.
These reference weights are fixed across conditions; they are not each case's
own physical volumes. Selecting this training case avoids deleting cells or
inventing positive weights for collapsed rounded exports.

Cache contract
--------------
results/cache/cases.json:
  Ordered 27 training cases followed by held-out M175 and M325.
results/cache/mach.npy:
  Native export order; NOT corresponding across cases. Do not train from it.
results/cache/registered_mach.npy:
  Float32 [29, 3533741] in the verified common material-cell order.
results/cache/registered_reference_centers.npy:
  M17 cell-centre coordinates, float64 [3533741, 3], same material-cell order.
results/cache/registered_reference_volumes.npy:
  M17 positive physical cell volumes, float64 [3533741].
results/cache/registered_reference_nodes.npy:
  M17 node coordinates, float64 [3639168, 3], common material-node order.
results/cache/native/<case>/structured_to_native_cells.npy:
results/cache/native/<case>/structured_to_native_nodes.npy:
  Exact per-case permutation maps for reconstructing actual source geometry.
results/cache/extraction_audit.json:
  Raw source hashes and native connectivity hashes.
results/cache/geometry_audit.json:
  Per-case physical geometry and rounded-coordinate diagnostics.
results/cache/extraction_manifest.json:
  Final registration proof, reference geometry, row membership, and readiness.

The large cache is local scientific working data, not a browser payload or a
GitHub Pages asset. Browser display reduction is performed separately after
the model has been trained and validated on every registered source cell.
