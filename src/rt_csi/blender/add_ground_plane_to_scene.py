#!/usr/bin/env python3
"""Create a Sionna/Mitsuba XML copy with a continuous ground plane mesh."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


GROUND_MATERIAL_COLORS = {
    "itu_wet_ground": "0.45, 0.36, 0.20",
    "itu_medium_dry_ground": "0.52, 0.43, 0.28",
    "itu_very_dry_ground": "0.62, 0.50, 0.31",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="Input Sionna/Mitsuba XML scene")
    parser.add_argument("--out-xml", type=Path, required=True, help="Output XML path")
    parser.add_argument("--name", default="GroundPlane", help="Ground object and mesh name")
    parser.add_argument("--material", default="itu_wet_ground", help="ITU radio material name")
    parser.add_argument("--z", type=float, help="Ground plane height. Defaults to scene min z.")
    parser.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--size", nargs=2, type=float, metavar=("X_SIZE", "Y_SIZE"))
    parser.add_argument("--margin", type=float, default=80.0, help="Extra extent around metadata bounds")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Scene metadata JSON. Defaults to scene_metadata.json next to the input XML.",
    )
    parser.add_argument(
        "--remove-terrain",
        action="store_true",
        help="Remove Terrain.ply shapes for a flat Sionna-style visualization copy.",
    )
    return parser.parse_args()


def load_metadata_bounds(xml_path: Path, metadata_path: Path | None) -> tuple[list[float], list[float]]:
    path = metadata_path or xml_path.parent / "scene_metadata.json"
    if not path.exists():
        raise SystemExit(
            "No metadata bounds found. Pass --center and --size, or provide --metadata."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    offset = data.get("offset") or {}
    mins = offset.get("new_bbox_min") or offset.get("original_bbox_min")
    maxs = offset.get("new_bbox_max") or offset.get("original_bbox_max")
    if not mins or not maxs:
        raise SystemExit(f"Metadata does not contain scene bounds: {path}")
    return [float(value) for value in mins], [float(value) for value in maxs]


def write_ground_ply(path: Path, center: tuple[float, float], size: tuple[float, float], z: float) -> None:
    half_x = size[0] * 0.5
    half_y = size[1] * 0.5
    x0, x1 = center[0] - half_x, center[0] + half_x
    y0, y1 = center[1] - half_y, center[1] + half_y
    vertices = [
        (x0, y0, z),
        (x1, y0, z),
        (x1, y1, z),
        (x0, y1, z),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write("element vertex 4\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property float nx\n")
        fh.write("property float ny\n")
        fh.write("property float nz\n")
        fh.write("element face 2\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for vertex in vertices:
            fh.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} 0.0 0.0 1.0\n")
        fh.write("3 0 1 2\n")
        fh.write("3 0 2 3\n")


def ensure_material(root: ET.Element, material: str) -> str:
    material_id = material if material.startswith("mat-") else f"mat-{material}"
    if root.find(f".//bsdf[@id='{material_id}']") is not None:
        return material_id

    color = GROUND_MATERIAL_COLORS.get(material.removeprefix("mat-"), "0.45, 0.36, 0.20")
    outer = ET.Element("bsdf", type="twosided", id=material_id, name=material_id)
    inner = ET.SubElement(outer, "bsdf", type="diffuse")
    ET.SubElement(inner, "rgb", name="reflectance", value=color)
    root.insert(0, outer)
    return material_id


def remove_existing_ground(root: ET.Element, ground_id: str) -> int:
    removed = 0
    for shape in list(root.findall("shape")):
        if shape.attrib.get("id") == ground_id or shape.attrib.get("name") == ground_id:
            root.remove(shape)
            removed += 1
    return removed


def remove_terrain(root: ET.Element) -> int:
    removed = 0
    for shape in list(root.findall("shape")):
        filename = shape.find("string[@name='filename']")
        filename_value = filename.attrib.get("value", "") if filename is not None else ""
        shape_text = " ".join(
            value for value in (shape.attrib.get("id"), shape.attrib.get("name"), filename_value) if value
        ).lower()
        if "terrain" in shape_text:
            root.remove(shape)
            removed += 1
    return removed


def append_ground_shape(root: ET.Element, ground_id: str, mesh_filename: str, material_id: str) -> None:
    shape = ET.SubElement(root, "shape", type="ply", id=ground_id, name=ground_id)
    ET.SubElement(shape, "string", name="filename", value=mesh_filename)
    ET.SubElement(shape, "ref", id=material_id, name="bsdf")
    ET.SubElement(shape, "boolean", name="face_normals", value="true")


def main() -> None:
    args = parse_args()
    xml_path = args.xml.expanduser().resolve()
    out_xml = args.out_xml.expanduser().resolve()
    mesh_dir = out_xml.parent / "meshes"
    mesh_name = f"{args.name}.ply"
    mesh_path = mesh_dir / mesh_name

    mins, maxs = load_metadata_bounds(xml_path, args.metadata)
    center = tuple(args.center) if args.center else ((mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5)
    size = tuple(args.size) if args.size else (
        maxs[0] - mins[0] + 2.0 * args.margin,
        maxs[1] - mins[1] + 2.0 * args.margin,
    )
    z = args.z if args.z is not None else mins[2]

    if xml_path.parent != out_xml.parent:
        raise SystemExit("Output XML must be written next to the input XML so mesh references stay valid.")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    material_id = ensure_material(root, args.material)
    ground_id = args.name
    removed_ground = remove_existing_ground(root, ground_id)
    removed_terrain = remove_terrain(root) if args.remove_terrain else 0

    write_ground_ply(mesh_path, center, size, z)
    append_ground_shape(root, ground_id, f"meshes/{mesh_name}", material_id)

    ET.indent(tree, space="\t")
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)
    print(f"Wrote ground mesh: {mesh_path}")
    print(f"Wrote XML: {out_xml}")
    print(f"Ground center={center}, size={size}, z={z}")
    print(f"Removed existing ground shapes: {removed_ground}")
    print(f"Removed terrain shapes: {removed_terrain}")


if __name__ == "__main__":
    main()
