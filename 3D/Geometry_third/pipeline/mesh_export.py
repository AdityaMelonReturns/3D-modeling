"""Stage [I]: Meshing + Export.

Poisson surface reconstruction (Open3D) on the fused cloud, vertex-colored
by signed distance to the reference plane (a depth-deviation heatmap) -
per the spec, often the single most useful artifact for a human reviewer,
independent of whether the numeric measurements above are fully trusted.
"""
import numpy as np
import open3d as o3d
import trimesh

from .plane import signed_distance


def heatmap_color(d, d_max=None):
    """Signed distance -> red/blue diverging color (red = protrudes toward
    camera, blue = recedes / is a hole, per typical sign convention of the
    fitted plane normal pointing toward the camera)."""
    d_max = d_max or (np.percentile(np.abs(d), 98) + 1e-9)
    t = np.clip(d / d_max, -1, 1)
    colors = np.zeros((len(d), 3))
    pos = t > 0
    colors[pos, 0] = t[pos]           # red channel for protrusion
    colors[~pos, 2] = -t[~pos]        # blue channel for recession
    colors[:, 1] = 1 - np.abs(t)      # green fades out toward extremes
    return np.clip(colors, 0, 1)


def build_mesh(fused_pcd, normal, point_on_plane, alpha_multiplier=3.0):
    """Alpha-shape mesh from the fused (defect + margin) point cloud.

    NOT Poisson anymore. Poisson needs consistently-oriented per-point
    normals first (estimate_normals + orient_normals_consistent_tangent_plane
    / orient_normals_towards_camera_location), and confirmed live across
    multiple real defect sets - `ceiling` (67k pts), `gaps` (a mere 1655
    pts) - that this can hang for many minutes regardless of cloud size or
    which orientation method was used; it isn't a size problem, something
    about that whole code path is unreliable in this Open3D build. A hard
    wall-clock signal timeout didn't reliably save it either (Python signal
    handlers only run when control returns to the interpreter, and this call
    can sit inside a single uninterruptible C++ call the whole time).

    Alpha-shape reconstruction sidesteps all of that - it needs no normals
    and no orientation step at all, just raw point positions and one alpha
    (search radius) parameter, estimated here from the cloud's own point
    spacing. It's a Delaunay-based combinatorial computation, not an
    iterative solver, so it's a fundamentally different (fast, bounded) code
    path rather than a tuned version of the same slow one."""
    distances = fused_pcd.compute_nearest_neighbor_distance()
    avg_spacing = float(np.mean(distances)) if len(distances) else 0.01
    alpha = max(avg_spacing * alpha_multiplier, 1e-6)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(fused_pcd, alpha)
    mesh.compute_vertex_normals()  # cheap, mesh-topology-based - not the point-cloud orientation step above

    verts = np.asarray(mesh.vertices)
    d = signed_distance(verts, normal, point_on_plane) if len(verts) else np.array([])
    mesh.vertex_colors = o3d.utility.Vector3dVector(heatmap_color(d) if len(d) else np.zeros((0, 3)))
    return mesh, d


def export_pointcloud(fused_pcd, out_dir, prefix="defect"):
    """Just the raw fused point cloud - cheap, can't hang (no Poisson
    involved). Call this BEFORE attempting build_mesh(), not after - if
    meshing later hangs/gets killed, the point cloud is already safely on
    disk instead of being lost with it (this exact loss happened on 7 of 8
    real defect runs before this split existed: killing a stuck meshing
    process took export_all()'s point-cloud write down with it, since it
    was bundled after the mesh build in one function)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    ply_path = os.path.join(out_dir, f"{prefix}_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, fused_pcd)
    return ply_path


def export_mesh(mesh, out_dir):
    """Mesh-only export (ply + glb) - call after build_mesh() succeeds."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    mesh_ply_path = os.path.join(out_dir, "mesh.ply")
    glb_path = os.path.join(out_dir, "mesh.glb")

    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)

    # glb export via trimesh (Open3D's own glb export is less reliable for
    # vertex colors across versions)
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
        process=False,
    )
    tm.export(glb_path)
    return {"mesh_ply": mesh_ply_path, "mesh_glb": glb_path}


def export_all(mesh, fused_pcd, out_dir, prefix="defect"):
    """Kept for backward compat (Geometry_Second still calls this as one
    step) - Geometry_third's run_defect.py calls export_pointcloud() and
    export_mesh() separately instead, see their docstrings for why."""
    ply_path = export_pointcloud(fused_pcd, out_dir, prefix)
    paths = export_mesh(mesh, out_dir)
    paths["pointcloud_ply"] = ply_path
    return paths

    return {"pointcloud_ply": ply_path, "mesh_ply": mesh_ply_path, "mesh_glb": glb_path}
