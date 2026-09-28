# License scope

- `src/rt_csi/` and `pyproject.toml`: MIT License, see
  [`LICENSE-CODE`](../LICENSE-CODE). Copyright © 2026 Tang Shunpu.
- `data/`: Creative Commons Attribution 4.0 International (CC BY 4.0), see
  [`LICENSE-DATA`](../LICENSE-DATA) and https://creativecommons.org/licenses/by/4.0/ . This is
  the dataset author's license for their generated CSI tensors, position
  arrays, split annotations, and metadata. Cite the benchmark and retain
  OpenStreetMap attribution when reusing the dataset.
- `scenes/`: custom scene geometry includes information derived from
  OpenStreetMap data © OpenStreetMap contributors under ODbL 1.0. No MIT or
  CC BY grant is made here for third-party geographic data. The origin and
  redistribution terms of terrain input still need confirmation before
  these files are publicly uploaded. See [`third-party-notices.md`](third-party-notices.md) and
  https://www.openstreetmap.org/copyright .
- The NVIDIA city scenes are loaded from the separately installed Sionna RT
  package and are not copied into this repository. Refer to NVIDIA's scene
  documentation and the license accompanying that package.
- `uv.lock` records third-party dependencies and does not relicense them.

Attribution for any published dataset description or scene visualization:
“© OpenStreetMap contributors” linked to
https://www.openstreetmap.org/copyright .
