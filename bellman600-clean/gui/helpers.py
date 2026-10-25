from __future__ import annotations
import numpy as np
import trimesh
import viser
import yaml
import yourdfpy

def load_urdf(urdf_path: str) -> yourdfpy.URDF:
    """Load URDF with yourdfpy."""
    return yourdfpy.URDF.load(
        urdf_path,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )

def load_scene_yaml(path: str) -> list[dict]:
    """Load scene YAML and return list of cuboid dicts."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("cuboids", [])

def load_zones_yaml(path: str) -> dict:
    """Load the optional `zones:` block from a scene YAML (empty dict if absent)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("zones", {}) or {}

def load_viz_table_yaml(path: str) -> dict | None:
    """Load the optional visual-only table from a scene YAML."""
    tables = load_viz_tables_yaml(path)
    return tables[0] if tables else None

def load_viz_tables_yaml(path: str) -> list[dict]:
    """Load optional visual-only tables from a scene YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    tables = data.get("viz_tables") or []
    if tables:
        return [
            {
                "name": table.get("name", f"viz_table_{i}"),
                "pose": table["pose"],
                "dims": table["dims"],
            }
            for i, table in enumerate(tables)
        ]

    table = data.get("viz_table")
    if not table:
        return []
    return [{
        "name": table.get("name", "viz_table"),
        "pose": table["pose"],
        "dims": table["dims"],
    }]

def add_scene_to_viser(
    server: viser.ViserServer,
    cuboids: list[dict],
) -> list[viser.BoxHandle]:
    """Add cuboid obstacles to viser scene."""
    handles = []
    for cuboid in cuboids:
        name = cuboid["name"]
        pose = cuboid["pose"]  # [x, y, z, qw, qx, qy, qz]
        dims = cuboid["dims"]
        position = (pose[0], pose[1], pose[2])
        wxyz = (pose[3], pose[4], pose[5], pose[6])
        is_table = "table" in name
        color = (139, 90, 43) if is_table else (235, 72, 42)
        handle = server.scene.add_box(
            name=f"/scene/{name}",
            dimensions=(dims[0], dims[1], dims[2]),
            color=color,
            opacity=0.6 if is_table else 0.88,
            wxyz=wxyz,
            position=position,
        )
        handles.append(handle)
        if not is_table:
            label = server.scene.add_label(
                name=f"/scene/{name}_label",
                text=(
                    f"NEW BOX: {name} | center "
                    f"({pose[0]:+.2f}, {pose[1]:+.2f}, {pose[2]:+.3f}) m | "
                    f"size {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} m"
                ),
                position=(
                    pose[0], pose[1], pose[2] + dims[2] / 2.0 + 0.04
                ),
            )
            handles.append(label)
    return handles

def add_viz_table_to_viser(
    server: viser.ViserServer,
    table: dict,
) -> viser.BoxHandle:
    """Add the visual-only tabletop to viser without changing collision cuboids."""
    name = table.get("name", "viz_table")
    pose = table["pose"]
    dims = table["dims"]
    return server.scene.add_box(
        name=f"/scene/{name}",
        dimensions=(dims[0], dims[1], dims[2]),
        color=(125, 120, 110),
        opacity=0.28,
        wxyz=(pose[3], pose[4], pose[5], pose[6]),
        position=(pose[0], pose[1], pose[2]),
    )

def scene_cuboids_for_drawing(cuboids: list[dict], viz_tables: list[dict]) -> list[dict]:
    """Return cuboids to draw when visual-only tables are present."""
    if not viz_tables:
        return cuboids
    return [cuboid for cuboid in cuboids if "table" not in cuboid["name"]]

def scene_table_for_zones(cuboids: list[dict], viz_tables: list[dict]) -> dict | None:
    """Return the table metadata used to size zone overlays."""
    if viz_tables:
        return viz_tables[0]
    return next((cuboid for cuboid in cuboids if "table" in cuboid["name"]), None)

def add_scene_yaml_to_viser(
    server: viser.ViserServer,
    scene_path: str,
    draw_zones: bool = True,
) -> dict:
    """Load a scene YAML, draw scene geometry/zones, and return scene metadata."""
    cuboids = load_scene_yaml(scene_path)
    viz_tables = load_viz_tables_yaml(scene_path)
    for table_viz in viz_tables:
        add_viz_table_to_viser(server, table_viz)
    viz_table = viz_tables[0] if viz_tables else None

    draw_cuboids = scene_cuboids_for_drawing(cuboids, viz_tables)
    scene_handles = add_scene_to_viser(server, draw_cuboids)

    zones = load_zones_yaml(scene_path)
    table = scene_table_for_zones(cuboids, viz_tables)
    zone_handles = []
    if draw_zones and zones and zones.get("type") == "corner_table":
        zone_handles = add_corner_table_zones_to_viser(server, zones, viz_tables, cuboids)
    elif draw_zones and zones and table is not None:
        zone_handles = add_zone_planes_to_viser(
            server,
            y_divide=zones.get("y_divide", 0.0),
            overlap=zones.get("overlap", 0.05),
            x_center=zones.get("x_center", table["pose"][0]),
            cup_side=zones.get("cup_side", 1),
            width=table["dims"][0],
            y_extent=table["dims"][1] / 2.0,
        )

    return {
        "cuboids": cuboids,
        "viz_table": viz_table,
        "viz_tables": viz_tables,
        "draw_cuboids": draw_cuboids,
        "zones": zones,
        "table": table,
        "scene_handles": scene_handles,
        "zone_handles": zone_handles,
    }

def add_zone_planes_to_viser(
    server: viser.ViserServer,
    y_divide: float = -0.2,
    overlap: float = 0.05,
    x_center: float = 0.30,
    z_center: float = 0.10,
    width: float = 0.50,    # X extent
    height: float = 0.30,   # Z extent
    cup_side: int = -1,     # -1: cup on lower-y side (G1); +1: cup on higher-y side (UR5e)
    y_extent: float = 0.35, # how far the floor strips reach away from each boundary
) -> list:
    """Add zone boundary lines (not solid boxes) to viser scene.

    Draws thin colored lines on the ground plane at zone boundaries,
    plus small labels. No solid/transparent boxes to avoid z-fighting and occlusion.

    `cup_side` selects which side of `y_divide` is the cupping zone:
      -1 → cupping occupies y < y_divide (G1: arm hangs to lower y),
      +1 → cupping occupies y > y_divide (UR5e: cupping on +y / LEFT).
    The pouring zone is always the opposite side.
    """
    import numpy as _np
    handles = []
    cup_side = 1 if cup_side >= 0 else -1
    # cup_bound is the inner edge of the cupping band; pour_bound the inner edge of pouring.
    cup_bound = y_divide + cup_side * overlap
    pour_bound = y_divide - cup_side * overlap
    z_lo = z_center - height / 2
    z_hi = z_center + height / 2
    x_lo = x_center - width / 2
    x_hi = x_center + width / 2

    def _add_boundary_line(name, y_val, color, line_width=2.0):
        """Draw a vertical rectangle outline at a given Y plane using line segments."""
        corners = _np.array([
            [x_lo, y_val, z_lo],
            [x_hi, y_val, z_lo],
            [x_hi, y_val, z_hi],
            [x_lo, y_val, z_hi],
        ], dtype=_np.float32)
        # Draw each edge as a separate line_segments call (1, 2, 3) to avoid color shape issues
        edge_handles = []
        edges = [(0,1), (1,2), (2,3), (3,0)]
        for ei, (a, b) in enumerate(edges):
            pts = _np.array([[[corners[a][0], corners[a][1], corners[a][2]],
                              [corners[b][0], corners[b][1], corners[b][2]]]], dtype=_np.float32)
            cols = _np.array([color], dtype=_np.uint8)
            edge_handles.append(server.scene.add_line_segments(
                name=f"{name}_{ei}",
                points=pts, colors=cols, line_width=line_width,
            ))
        return edge_handles

    # y_divide boundary (red, thickest)
    handles.append(_add_boundary_line("/zones/y_divide", y_divide, (220, 50, 50), line_width=3.0))

    # Cup boundary (blue)
    handles.append(_add_boundary_line("/zones/cup_boundary", cup_bound, (70, 130, 210), line_width=2.0))

    # Pour boundary (orange)
    handles.append(_add_boundary_line("/zones/pour_boundary", pour_bound, (210, 160, 30), line_width=2.0))

    # Ground-level colored strips to indicate zones (thin lines on the floor).
    # Enough strips (0.02 spacing) to march the full y_extent so the bands fill
    # the table rather than stopping in a small central patch.
    z_floor = z_lo - 0.005
    strip_step = 0.02
    n_strips = max(12, int(_np.ceil(y_extent / strip_step)) + 1)
    # Cupping strips march away from cup_bound in the +cup_side direction.
    cup_limit = cup_bound + cup_side * y_extent
    for i in range(n_strips):
        y_c = cup_bound + cup_side * (0.01 + i * strip_step)
        if (cup_side < 0 and y_c < cup_limit) or (cup_side > 0 and y_c > cup_limit):
            break
        pts = _np.array([[[x_lo, y_c, z_floor], [x_hi, y_c, z_floor]]], dtype=_np.float32)
        colors = _np.array([(70, 130, 210)], dtype=_np.uint8)
        server.scene.add_line_segments(f"/zones/cup_strip_{i}", pts, colors, line_width=1.0)

    # Pouring strips march away from pour_bound in the -cup_side direction.
    pour_limit = pour_bound - cup_side * y_extent
    for i in range(n_strips):
        y_p = pour_bound - cup_side * (0.01 + i * strip_step)
        if (cup_side > 0 and y_p < pour_limit) or (cup_side < 0 and y_p > pour_limit):
            break
        pts = _np.array([[[x_lo, y_p, z_floor], [x_hi, y_p, z_floor]]], dtype=_np.float32)
        colors = _np.array([(210, 160, 30)], dtype=_np.uint8)
        server.scene.add_line_segments(f"/zones/pour_strip_{i}", pts, colors, line_width=1.0)

    # Labels
    server.scene.add_label(
        name="/zones/cup_label",
        text="CUP",
        position=(x_center, cup_bound + cup_side * 0.08, z_hi + 0.01),
    )
    server.scene.add_label(
        name="/zones/overlap_label",
        text="OVERLAP",
        position=(x_center, y_divide, z_hi + 0.01),
    )
    server.scene.add_label(
        name="/zones/pour_label",
        text="POUR",
        position=(x_center, pour_bound - cup_side * 0.05, z_hi + 0.01),
    )

    return handles

def _table_top_z(tables: list[dict], cuboids: list[dict], default: float = 0.0) -> float:
    """Return the highest tabletop z from visual tables or collision table cuboids."""
    candidates = tables or [cuboid for cuboid in cuboids if "table" in cuboid["name"]]
    if not candidates:
        return default
    return max(float(t["pose"][2]) + float(t["dims"][2]) / 2.0 for t in candidates)

def _add_rect_outline(server: viser.ViserServer, name: str, x_range, y_range, z: float,
                      color, line_width: float) -> list:
    x_lo, x_hi = float(x_range[0]), float(x_range[1])
    y_lo, y_hi = float(y_range[0]), float(y_range[1])
    corners = np.array([
        [x_lo, y_lo, z],
        [x_hi, y_lo, z],
        [x_hi, y_hi, z],
        [x_lo, y_hi, z],
    ], dtype=np.float32)
    handles = []
    for i, (a, b) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
        points = np.array([[corners[a], corners[b]]], dtype=np.float32)
        colors = np.array([color], dtype=np.uint8)
        handles.append(server.scene.add_line_segments(
            name=f"{name}_{i}",
            points=points,
            colors=colors,
            line_width=line_width,
        ))
    return handles

def _add_region_strips(server: viser.ViserServer, prefix: str, x_range, y_range, z: float,
                       color, line_width: float = 1.0) -> list:
    x_lo, x_hi = float(x_range[0]), float(x_range[1])
    y_lo, y_hi = float(y_range[0]), float(y_range[1])
    height = max(y_hi - y_lo, 0.0)
    strip_step = 0.035
    n_strips = max(3, int(np.ceil(height / strip_step)) + 1)
    handles = []
    for i in range(n_strips):
        y = y_lo + (i + 0.5) * height / n_strips
        points = np.array([[[x_lo, y, z], [x_hi, y, z]]], dtype=np.float32)
        colors = np.array([color], dtype=np.uint8)
        handles.append(server.scene.add_line_segments(
            name=f"{prefix}_{i}",
            points=points,
            colors=colors,
            line_width=line_width,
        ))
    return handles

def add_corner_table_zones_to_viser(
    server: viser.ViserServer,
    zones: dict,
    viz_tables: list[dict],
    cuboids: list[dict],
) -> list:
    """Draw rectangular task overlays for the UR5e corner-table prototype."""
    z = float(zones.get("z", _table_top_z(viz_tables, cuboids) + 0.012))
    label_z = z + float(zones.get("label_height", 0.13))
    handles = []

    region_specs = [
        ("cup", zones["cup_region"], (70, 130, 210), "CUP"),
        ("pour", zones["pour_region"], (210, 160, 30), "POUR"),
    ]
    if zones.get("overlap_region"):
        region_specs.append(("overlap", zones["overlap_region"], (220, 50, 50), "CORNER"))

    for key, region, color, default_label in region_specs:
        x_range = region["x"]
        y_range = region["y"]
        label = region.get("label", default_label)
        handles.extend(_add_rect_outline(
            server, f"/zones/{key}_region", x_range, y_range, z, color, line_width=2.5
        ))
        handles.extend(_add_region_strips(
            server, f"/zones/{key}_strip", x_range, y_range, z, color, line_width=1.0
        ))
        x_mid = (float(x_range[0]) + float(x_range[1])) / 2.0
        y_mid = (float(y_range[0]) + float(y_range[1])) / 2.0
        handles.append(server.scene.add_label(
            name=f"/zones/{key}_label",
            text=label,
            position=(x_mid, y_mid, label_z),
        ))

    if zones.get("draw_routing_map", True) and hasattr(server.scene, "add_point_cloud"):
        cup = zones["cup_region"]
        pour = zones["pour_region"]
        overlap = zones.get("overlap_region")

        def _contains(region, x, y):
            return (
                float(region["x"][0]) <= x <= float(region["x"][1])
                and float(region["y"][0]) <= y <= float(region["y"][1])
            )

        def _dist(region, x, y):
            dx = max(float(region["x"][0]) - x, x - float(region["x"][1]), 0.0)
            dy = max(float(region["y"][0]) - y, y - float(region["y"][1]), 0.0)
            return float(np.sqrt(dx * dx + dy * dy))

        xs = [
            float(cup["x"][0]), float(cup["x"][1]),
            float(pour["x"][0]), float(pour["x"][1]),
        ]
        ys = [
            float(cup["y"][0]), float(cup["y"][1]),
            float(pour["y"][0]), float(pour["y"][1]),
        ]
        if overlap is not None:
            xs.extend([float(overlap["x"][0]), float(overlap["x"][1])])
            ys.extend([float(overlap["y"][0]), float(overlap["y"][1])])
        gx = np.arange(min(xs), max(xs) + 1e-6, float(zones.get("routing_map_step", 0.05)))
        gy = np.arange(min(ys), max(ys) + 1e-6, float(zones.get("routing_map_step", 0.05)))
        points = []
        colors = []
        for x in gx:
            for y in gy:
                in_overlap = overlap is not None and _contains(overlap, x, y)
                if in_overlap:
                    color = (220, 50, 50)
                elif _contains(cup, x, y):
                    color = (70, 130, 210)
                elif _contains(pour, x, y):
                    color = (210, 160, 30)
                else:
                    color = (120, 120, 120) if _dist(cup, x, y) <= _dist(pour, x, y) else (150, 130, 90)
                points.append((x, y, z + 0.006))
                colors.append(color)
        handles.append(server.scene.add_point_cloud(
            "/zones/routing_map",
            points=np.asarray(points, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
            point_size=float(zones.get("routing_map_point_size", 0.008)),
        ))

    return handles

def get_link_transform(urdf: yourdfpy.URDF, link_name: str) -> np.ndarray | None:
    """Get the 4x4 world-frame transform for a link. Returns None if not found.

    Lookup order:
    1. link_name directly in scene.graph.nodes  (works for any URDF: UR5e DAE, G1 STL, etc.)
    2. <link_name>.STL in scene.graph.nodes_geometry  (legacy G1 path, kept for safety)
    3. first geometry node that starts with link_name  (legacy fallback)
    """
    scene = urdf.scene
    # Primary: link frames are always nodes in the kinematic scene graph
    if link_name in scene.graph.nodes:
        transform, _ = scene.graph.get(link_name)
        return transform
    # Legacy fallback: geometry nodes named <link>.STL (G1-style)
    node_name = f"{link_name}.STL"
    if node_name not in scene.graph.nodes_geometry:
        for n in scene.graph.nodes_geometry:
            if n.startswith(link_name):
                node_name = n
                break
        else:
            return None
    transform, _ = scene.graph.get(node_name)
    return transform

JOINT_NAMES=("shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint")

def add_robot_scaled(server: viser.ViserServer, urdf) -> dict:
    """Render mixed-unit UR5e/Robotiq visual meshes in meters."""
    handles = {}
    scene = urdf.scene
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh):
            try:
                mesh = mesh.to_mesh()
            except Exception:
                continue
        mesh = mesh.copy()
        if float(mesh.extents.max()) > 10.0:
            mesh.apply_scale(0.001)
        quaternion = trimesh.transformations.quaternion_from_matrix(transform)
        handles[node_name] = server.scene.add_mesh_trimesh(
            f"/robot/{node_name}",
            mesh=mesh,
            wxyz=tuple(quaternion),
            position=tuple(transform[:3, 3]),
        )
    return handles

def set_urdf(urdf, q: np.ndarray) -> None:
    cfg = urdf.zero_cfg.copy()
    for name, value in zip(JOINT_NAMES, q, strict=True):
        cfg[urdf.actuated_joint_names.index(name)] = float(value)
    urdf.update_cfg(cfg)

def update_robot(urdf, mesh_handles: dict, q: np.ndarray) -> None:
    set_urdf(urdf, q)
    for node_name, handle in mesh_handles.items():
        transform, _ = urdf.scene.graph.get(node_name)
        quaternion = trimesh.transformations.quaternion_from_matrix(transform)
        handle.wxyz = tuple(quaternion)
        handle.position = tuple(transform[:3, 3])

def ee_position(urdf, q: np.ndarray) -> np.ndarray:
    set_urdf(urdf, q)
    transform = get_link_transform(urdf, "grasp_frame")
    if transform is None:
        transform = get_link_transform(urdf, "tool0")
    if transform is None:
        raise RuntimeError("URDF has neither grasp_frame nor tool0")
    return np.asarray(transform[:3, 3], dtype=np.float32)

def ee_path(urdf, trajectory: np.ndarray, maximum: int = 500) -> np.ndarray:
    if len(trajectory) == 0:
        return np.empty((0, 3), dtype=np.float32)
    indices = np.unique(
        np.linspace(0, len(trajectory) - 1, min(len(trajectory), maximum))
        .round().astype(int)
    )
    return np.stack([ee_position(urdf, trajectory[i]) for i in indices])

def segments(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
