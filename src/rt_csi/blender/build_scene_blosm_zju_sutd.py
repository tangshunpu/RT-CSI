#!/usr/bin/env python3
"""Build a Sionna RT scene from OpenStreetMap data inside Blender.

Run with:

    blender --background --python src/rt_csi/blender/build_scene_blosm_zju_sutd.py -- \
        --config scenes/ZJU/ZJU.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Blender 3.x fallback
    import tomli as tomllib

try:
    import bpy
    import bmesh
    from mathutils import Vector
except ModuleNotFoundError as exc:  # pragma: no cover - requires Blender
    raise SystemExit(
        "This script must run inside Blender. Use: "
        "blender --background --python src/rt_csi/blender/build_scene_blosm_zju_sutd.py -- --config ..."
    ) from exc

_homebrew_llvm = Path("/opt/homebrew/opt/llvm/lib/libLLVM.dylib")
if _homebrew_llvm.exists():
    os.environ.setdefault("DRJIT_LIBLLVM_PATH", str(_homebrew_llvm))


ITU_COLORS = {
    "itu_concrete": (0.54, 0.54, 0.54, 1.0),
    "itu_brick": (0.54, 0.12, 0.08, 1.0),
    "itu_marble": (0.70, 0.64, 0.49, 1.0),
    "itu_metal": (0.22, 0.22, 0.25, 1.0),
    "itu_glass": (0.55, 0.80, 0.95, 0.45),
    "itu_wood": (0.38, 0.24, 0.12, 1.0),
    "itu_wet_ground": (0.45, 0.36, 0.20, 1.0),
    "itu_very_dry_ground": (0.62, 0.50, 0.31, 1.0),
    "itu_medium_dry_ground": (0.52, 0.43, 0.28, 1.0),
}


def blender_user_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def load_config(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def add_external_blender_packages(project_root: Path) -> None:
    package_dir = project_root / "vendor" / "blender-python-packages"
    if package_dir.exists() and str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    mitsuba_blender_dir = project_root / "vendor" / "build" / "mitsuba-blender-patched"
    if mitsuba_blender_dir.exists() and str(mitsuba_blender_dir) not in sys.path:
        sys.path.insert(0, str(mitsuba_blender_dir))


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def sanitize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    if not ascii_name:
        ascii_name = "obj_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    ascii_name = re.sub(r"[^\w.\-]+", "_", ascii_name)
    ascii_name = ascii_name.strip("._")
    return ascii_name or "unnamed_object"


def unique_name(name: str, used: set[str]) -> str:
    base = sanitize_name(name)
    candidate = base
    index = 1
    while candidate in used:
        index += 1
        candidate = f"{base}_{index:03d}"
    used.add(candidate)
    return candidate


def enable_addon(zip_path: Path | None, module_names: list[str]) -> str:
    installed = bpy.context.preferences.addons
    for module_name in module_names:
        if module_name in installed:
            print(f"[INFO] Add-on already enabled: {module_name}")
            return module_name

    if zip_path:
        if not zip_path.exists():
            raise FileNotFoundError(f"Add-on zip does not exist: {zip_path}")
        print(f"[INFO] Installing add-on from {zip_path}")
        bpy.ops.preferences.addon_install(filepath=str(zip_path), overwrite=True)

    last_error: Exception | None = None
    for module_name in module_names:
        try:
            bpy.ops.preferences.addon_enable(module=module_name)
            print(f"[INFO] Enabled add-on: {module_name}")
            return module_name
        except Exception as exc:  # Blender raises RuntimeError for bad module IDs
            last_error = exc

    names = ", ".join(module_names)
    raise RuntimeError(f"Could not enable any add-on module from [{names}]") from last_error


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    if "World" not in bpy.data.worlds:
        bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds.get("World")
    print("[INFO] Cleared Blender scene")


def configure_blosm(config: dict, data_dir: Path) -> None:
    scene = bpy.context.scene
    if not hasattr(scene, "blosm"):
        raise RuntimeError("Blosm add-on is not available on bpy.context.scene")

    prefs = bpy.context.preferences
    if "blosm" in prefs.addons:
        prefs.addons["blosm"].preferences.dataDir = str(data_dir)

    extent = config["extent"]
    scene.blosm.minLon = extent["min_lon"]
    scene.blosm.minLat = extent["min_lat"]
    scene.blosm.maxLon = extent["max_lon"]
    scene.blosm.maxLat = extent["max_lat"]


def import_blosm_layer(config: dict, data_type: str) -> None:
    scene = bpy.context.scene
    import_cfg = config.get("import", {})
    scene.blosm.dataType = data_type

    if data_type == "osm":
        if hasattr(scene.blosm, "mode"):
            scene.blosm.mode = import_cfg.get("osm_mode", "3Dsimple")
        attr_map = {
            "water": "water",
            "forests": "forests",
            "vegetation": "vegetation",
            "highways": "highways",
            "railways": "railways",
            "single_object": "singleObject",
        }
        for config_key, blosm_attr in attr_map.items():
            if hasattr(scene.blosm, blosm_attr):
                setattr(scene.blosm, blosm_attr, bool(import_cfg.get(config_key, False)))

    print(f"[INFO] Importing Blosm data type: {data_type}")
    bpy.ops.blosm.import_data()


def ensure_material(name: str, rgba: tuple[float, float, float, float] | None = None):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)

    rgba = rgba or ITU_COLORS.get(name, (0.50, 0.50, 0.50, 1.0))
    material.diffuse_color = rgba
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf and "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = rgba
    return material


def classify_material_name(obj_name: str, slot_name: str, material_cfg: dict) -> str:
    text = f"{obj_name} {slot_name}".lower()
    if "water" in text or "river" in text or "lake" in text:
        return material_cfg.get("water", material_cfg.get("terrain", "itu_wet_ground"))
    if (
        "forest" in text
        or "vegetation" in text
        or "grass" in text
        or "meadow" in text
        or "wood" in text
        or "tree" in text
    ):
        return material_cfg.get("vegetation", "itu_wood")
    if "roof" in text:
        return material_cfg.get("roof", "itu_metal")
    if "wall" in text or "building" in text:
        return material_cfg.get("wall", "itu_concrete")
    if "terrain" in text or "ground" in text:
        return material_cfg.get("terrain", "itu_wet_ground")
    if "road" in text or "highway" in text or "street" in text:
        return material_cfg.get("road", "itu_concrete")
    return material_cfg.get("default", "itu_concrete")


def assign_materials(config: dict) -> dict[str, int]:
    material_cfg = config.get("materials", {})
    assigned: dict[str, int] = {}

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.data.materials:
            for index, old_material in enumerate(obj.data.materials):
                old_name = old_material.name if old_material else ""
                new_name = classify_material_name(obj.name, old_name, material_cfg)
                obj.data.materials[index] = ensure_material(new_name)
                assigned[new_name] = assigned.get(new_name, 0) + 1
        else:
            new_name = classify_material_name(obj.name, "", material_cfg)
            obj.data.materials.append(ensure_material(new_name))
            assigned[new_name] = assigned.get(new_name, 0) + 1

    print(f"[INFO] Assigned materials: {assigned}")
    return assigned


def convert_curve_objects_to_meshes() -> int:
    curves = [obj for obj in bpy.context.scene.objects if obj.type == "CURVE"]
    if not curves:
        return 0

    bpy.ops.object.select_all(action="DESELECT")
    for obj in curves:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = curves[0]
    bpy.ops.object.convert(target="MESH")
    print(f"[INFO] Converted {len(curves)} curve objects to meshes")
    return len(curves)


def sanitize_object_and_mesh_names() -> None:
    used_objects: set[str] = set()
    used_meshes: set[str] = set()
    for obj in bpy.context.scene.objects:
        obj.name = unique_name(obj.name, used_objects)
        if obj.type == "MESH":
            obj.data.name = unique_name(obj.data.name, used_meshes)
    print("[INFO] Sanitized object and mesh names")


def triangulate_meshes() -> int:
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        count += 1
    print(f"[INFO] Triangulated {count} mesh objects")
    return count


def mesh_world_bbox() -> tuple[Vector, Vector] | None:
    min_corner = Vector((math.inf, math.inf, math.inf))
    max_corner = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
            found = True
    if not found:
        return None
    return min_corner, max_corner


def recenter_xy() -> dict[str, list[float]] | None:
    bbox = mesh_world_bbox()
    if bbox is None:
        return None
    min_corner, max_corner = bbox
    center = Vector(((min_corner.x + max_corner.x) / 2.0, (min_corner.y + max_corner.y) / 2.0, 0.0))

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.location.x -= center.x
            obj.location.y -= center.y

    new_bbox = mesh_world_bbox()
    print(f"[INFO] Recentered scene by XY offset: {[center.x, center.y]}")
    return {
        "original_bbox_min": [min_corner.x, min_corner.y, min_corner.z],
        "original_bbox_max": [max_corner.x, max_corner.y, max_corner.z],
        "applied_xy_offset": [center.x, center.y],
        "new_bbox_min": list(new_bbox[0]) if new_bbox else [],
        "new_bbox_max": list(new_bbox[1]) if new_bbox else [],
    }


def add_camera_and_sun(config: dict) -> None:
    view_cfg = config.get("view", {})
    if view_cfg.get("add_camera", True):
        bpy.ops.object.camera_add(location=tuple(view_cfg.get("camera_position", [0.0, -450.0, 260.0])))
        camera = bpy.context.object
        target = Vector(tuple(view_cfg.get("camera_look_at", [0.0, 0.0, 0.0])))
        direction = target - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.camera = camera

    if view_cfg.get("add_sun", True):
        bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 500.0))
        sun = bpy.context.object
        sun.name = "Sun"
        sun.data.energy = 2.0


def export_mitsuba_xml(xml_path: Path) -> None:
    import importlib

    from bpy_extras.io_utils import axis_conversion

    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("scalar_rgb")

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Exporting Mitsuba XML: {xml_path}")

    try:
        exporter = importlib.import_module("mitsuba-blender.io.exporter")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Mitsuba-Blender did not register its exporter. "
            "Enable the add-on and install its Mitsuba Python dependencies."
        ) from exc

    converter = exporter.SceneConverter()
    converter.export_ctx.axis_mat = axis_conversion(to_forward="Y", to_up="Z").to_4x4()
    converter.export_ctx.export_ids = True
    converter.use_selection = False
    converter.set_path(str(xml_path), split_files=False)

    window_manager = bpy.context.window_manager
    deps_graph = bpy.context.evaluated_depsgraph_get()
    window_manager.progress_begin(0, len(deps_graph.object_instances))
    try:
        converter.scene_to_dict(deps_graph, window_manager)
        converter.dict_to_xml()
    finally:
        window_manager.progress_end()


def patch_xml(xml_path: Path, add_face_normals: bool) -> dict[str, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    material_count = 0
    shape_count = 0
    for bsdf in root.findall(".//bsdf"):
        if "id" in bsdf.attrib:
            material_count += 1

    for shape in root.findall(".//shape"):
        shape_count += 1
        if add_face_normals and shape.attrib.get("type") in {"ply", "obj"}:
            has_face_normals = any(
                elem.tag == "boolean" and elem.attrib.get("name") == "face_normals"
                for elem in list(shape)
            )
            if not has_face_normals:
                ET.SubElement(shape, "boolean", name="face_normals", value="true")

        filename = shape.find("string[@name='filename']")
        if filename is not None:
            old = filename.attrib.get("value", "")
            filename.attrib["value"] = old.replace("\\", "/")

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] Patched XML: {shape_count} shapes, {material_count} material definitions")
    return {"shape_count": shape_count, "material_count": material_count}


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] Wrote metadata: {path}")


def cleanup_blosm_cache(data_dir: Path) -> None:
    for child in ("osm", "terrain"):
        path = data_dir / child
        if path.exists():
            shutil.rmtree(path)
            print(f"[INFO] Removed cache directory: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="TOML scene config")
    return parser.parse_args(blender_user_args())


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    add_external_blender_packages(config_path.parent.parent)
    config = load_config(config_path)
    base_dir = config_path.parent

    path_cfg = config.get("paths", {})
    data_dir = resolve_path(path_cfg.get("data_dir", "output/scene"), base_dir)
    assert data_dir is not None
    xml_path = data_dir / path_cfg.get("export_filename", "scene.xml")
    blend_path = data_dir / path_cfg.get("blend_filename", "scene.blend")
    blosm_zip = resolve_path(path_cfg.get("blosm_zip"), base_dir)
    mitsuba_zip = resolve_path(path_cfg.get("mitsuba_blender_zip"), base_dir)

    addon_cfg = config.get("addons", {})
    data_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    enable_addon(blosm_zip, addon_cfg.get("blosm_modules", ["blosm"]))
    # Do not enable the Mitsuba-Blender add-on here. Registering it in Blender
    # calls ensurepip inside Blender.app, which modifies the signed app bundle
    # and makes macOS report Blender as damaged. The exporter module is loaded
    # directly from vendor/build/mitsuba-blender-patched instead.
    configure_blosm(config, data_dir)

    import_cfg = config.get("import", {})
    if import_cfg.get("terrain", True):
        import_blosm_layer(config, "terrain")
    if import_cfg.get("osm", True):
        import_blosm_layer(config, "osm")

    post_cfg = config.get("postprocess", {})
    if post_cfg.get("sanitize_names", True):
        sanitize_object_and_mesh_names()
    convert_curve_objects_to_meshes()
    assigned_materials = assign_materials(config)
    if post_cfg.get("triangulate", True):
        triangulate_meshes()
    offset_metadata = recenter_xy() if post_cfg.get("recenter_xy", True) else None
    add_camera_and_sun(config)

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    export_mitsuba_xml(xml_path)
    xml_stats = patch_xml(xml_path, add_face_normals=post_cfg.get("add_face_normals", True))

    metadata = {
        "config": str(config_path),
        "extent": config["extent"],
        "xml": str(xml_path),
        "blend": str(blend_path),
        "assigned_materials": assigned_materials,
        "xml_stats": xml_stats,
        "offset": offset_metadata,
    }
    write_metadata(data_dir / "scene_metadata.json", metadata)

    if post_cfg.get("cleanup_blosm_cache", False):
        cleanup_blosm_cache(data_dir)

    print("[INFO] Done")


if __name__ == "__main__":
    main()
