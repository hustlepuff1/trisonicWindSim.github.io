Inlet-to-test-section 3D Mach training domain
===========================================

This is a new training dataset and model domain. The v10 full-volume datasets
and models remain unchanged. The downstream diffuser is excluded from the new
POD basis, GPR training fields, error integration, and display geometry.

Run:
  python3 prepare_domain.py --source-cache ../v10_volume_rom/results/cache
  python3 verify_domain.py

The source cache must contain the previously audited 29 native meshes and
registered Mach fields: 27 canonical training cases followed by held-out M175
and M325. No raw CFD re-export or parsing of the 27 GB source archive is needed.

Physical boundary and correspondence
------------------------------------
The requested terminal section is the full physical x=0 plane through the
origin, rather than one point. No original source has a node plane exactly at
x=0. Depending on condition and transverse position, x=0 crosses native axial
cell layers 286 and/or 287. A common index crop alone would either omit part of
the nozzle or retain downstream fluid.

All 29 cases share 286 wholly upstream native cell layers, i=0,...,285. Their
1,441,726 cell-centred Mach values are retained exactly in the original common
material order. The remaining upstream portions of native layers 286 and 287
are clipped geometrically to x<=0, then conservatively combined within each of
the 5,041 corresponding transverse cell columns. This yields 287*71*71 =
1,446,767 training control volumes covering the complete upstream domain.

The source Mach representation is piecewise constant in each CFD cell. A cell
straddling x=0 contributes only its upstream clipped physical volume; a cell
entirely downstream contributes zero. The terminal training value is the
physical-volume average of retained source contributions. All original cells
intersecting the upstream domain are represented. The common terminal layer
merges at most two source layers, rather than using downstream source values to
extrapolate a field to the boundary.

Clipping and quadrature
----------------------
Clipping uses the actual native quadrilateral face connectivity and its original
diagonal triangulation. Each face triangle and the audited native cell interior
point form a tetrahedron. Tetrahedra are intersected exactly with the x<=0
half-space; their retained volumes and first moments are summed. Their full
unclipped volumes agree with the previous native geometry audit to floating
point precision. Every terminal source cell in all 29 cases is geometrically
valid and has positive volume. No fallback or artificial volume floor was used.

M17 remains the fixed physical reference quadrature. Its unchanged prefix
weights are followed by the physical clipped terminal weights, all positive.
The reference quarter-domain volume is 0.48335330184715153 m^3. These are fixed
reference-domain weights, not each operating condition's own physical volumes.

Display geometry uses actual source nodes through node layer286 and intersects
every actual axial node track with x=0 for the terminal plane. The resulting
grid has shape (288,72,72,3), remains strictly at x<0 before its final layer, and
has x=0 exactly on that layer. The single terminal rendering cell simplifies
intermediate native edges; the scientific training weights and averages use
the complete clipped native geometry rather than that display simplification.

Cache contract
--------------
registered_mach.npy: float32 [29,1446767], ordered by ((i*71)+j)*71+k.
registered_reference_volumes.npy: positive float64 M17 reference weights.
registered_reference_centers.npy: M17 upstream centres, including clipped
  terminal first-moment centres.
registered_reference_nodes.npy: flattened M17 display geometry.
geometry/<case_id>.npy: float64 [288,72,72,3], preserving case-ID capitalization.
terminal/<case_id>_clipped_source_volumes.npy: float64 [2,71,71], exact retained
  volume contribution from each of the two original terminal source layers.
cases.json: unchanged 27 training and two held-out conditions.
domain_manifest.json and extraction_manifest.json: completion gates, source
  hashes, domain dimensions, per-case clipping checks, and final array hashes.
domain_verification.json: independent prefix-preservation, conservative-average,
  downstream-exclusion, exact-cap, reference-weight, and clipping-algorithm tests.

The verification tests also change every entirely downstream terminal source
value to 1e20 and confirm that no training value changes. An analytic clipped
cube and 10,000 random tetrahedra verify partitioned volumes and first moments.
