# Custom scene construction records

The three Python files in `src/rt_csi/blender/` were retained from the original ZJU/SUTD and
Shenzhen scene packages. They document the OpenStreetMap and Blosm import,
material assignment, Blender export, and Shenzhen ground-plane step that
produced the bundled Mitsuba XML/PLY scenes. The corresponding import
settings and geographic extents are in `../scenes/<scene>/*.toml`.

These are historical Blender recipes. They require Blender, the matching
Blosm and Mitsuba-Blender add-ons, and paths in the TOML `[paths]` sections
to be set for the machine running them. The add-ons and raw OSM/terrain
downloads are not bundled. Re-importing live OSM data can produce different
geometry, so the XML/PLY scenes in `../scenes/` are the canonical inputs to
the CSI generator.

The imported terrain is documented by Blosm as Mapzen-hosted terrain tiles
assembled from several elevation sources. Exact source tiles were not
recorded in the original scene metadata. See [`third-party-notices.md`](third-party-notices.md).
