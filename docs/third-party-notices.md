# Third-party scene data

## OpenStreetMap

The Munich, Paris Étoile, and Florence city scenes shipped with Sionna RT,
and the bundled ZJU, SUTD, and Shenzhen custom geometry was derived from
OpenStreetMap data. Attribution: **© OpenStreetMap contributors**.
OpenStreetMap data is licensed under ODbL 1.0:
https://www.openstreetmap.org/copyright .

The NVIDIA city scenes are not copied into this repository. NVIDIA's scene
documentation identifies their OSM source and ODbL terms:
https://nvlabs.github.io/sionna/rt/api/scene.html .

## Terrain

The custom Blender scene recipes import terrain through Blosm. Blosm's
terrain documentation says it uses Mapzen-prepared terrain tiles compiled
from multiple open elevation sources and requires attribution:
https://github.com/vvoovv/blosm/wiki/Terrain .
The upstream terrain attribution guide is:
https://github.com/tilezen/joerd/blob/master/docs/attribution.md .
The original scene records do not identify the exact downloaded tiles or
their component sources. Confirm those sources and give the applicable
attribution before public distribution of `scenes/`.

## Tools

The Blosm authoring add-on is not bundled. Its documentation and source
license are at https://github.com/vvoovv/blosm/wiki/Documentation . Sionna RT
is installed separately at the pinned version in `pyproject.toml`.
