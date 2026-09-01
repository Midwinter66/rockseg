# 10 mm Empty-Surface Diagnostic

Sampled meshes: 12; empty at 10 mm: 6; recovered at 5 mm: 6.

The current mesh rasterizer tests grid-cell centres against triangle XY projections. Thus a coarse grid can return an empty surface when no centre lands within any projected triangle, even though the mesh itself is valid.

| Dataset | Sample | Cached 10 mm | 2.5 mm cells | 5 mm cells | 10 mm cells | Face area < 100 mm2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| T01 | T01F005a | error | 7 | 1 | 0 | 1.000 |
| T01 | T01F011a | error | 18 | 5 | 0 | 1.000 |
| T01 | T01F012a | error | 14 | 2 | 0 | 1.000 |
| T01 | T01E143a | success | 67 | 16 | 3 | 1.000 |
| T01 | T01E144a | success | 122 | 30 | 7 | 1.000 |
| T01 | T01F000a | success | 44 | 9 | 2 | 1.000 |
| L01 | L01C027a | error | 23 | 5 | 0 | 1.000 |
| L01 | L01C031a | error | 18 | 3 | 0 | 1.000 |
| L01 | L01C035a | error | 19 | 3 | 0 | 1.000 |
| L01 | L01C010a | success | 53 | 13 | 2 | 1.000 |
| L01 | L01C012a | success | 39 | 9 | 2 | 1.000 |
| L01 | L01C014a | success | 77 | 18 | 3 | 1.000 |

## Decision

Do not train from the 10 mm cache. A revised mesh-to-grid coverage rule must first be defined and validated separately against the existing 0.5 mm method; changing it would create a different scientific measurement pipeline.
