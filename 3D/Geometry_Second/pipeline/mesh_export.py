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


def build_mesh(fused_pcd, normal, point_on_plane, poisson_depth=9):
    """Poisson mesh from the fused (defect + margin) point cloud. Requires
    per-point normals - estimate them if the cloud doesn't have any yet."""
    if not fused_pcd.has_normals():
        fused_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        fused_pcd.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        fused_pcd, depth=poisson_depth
    )
    # trim low-density (low-confidence) vertices - avoids the "789 sparse
    # points -> near-empty mesh" failure mode seen with defaults in the
    # COLMAP pilot (Geometry side, delaunay_mesher at depth=13 needed
    # depth=9/trim=4 for the sparser sets; same idea here, applied up front)
    densities = np.asarray(densities)
    density_thresh = np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(densities < density_thresh)

    verts = np.asarray(mesh.vertices)
    d = signed_distance(verts, normal, point_on_plane)
    mesh.vertex_colors = o3d.utility.Vector3dVector(heatmap_color(d))
    return mesh, d


def export_all(mesh, fused_pcd, out_dir, prefix="defect"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    ply_path = os.path.join(out_dir, f"{prefix}_pointcloud.ply")
    mesh_ply_path = os.path.join(out_dir, "mesh.ply")
    glb_path = os.path.join(out_dir, "mesh.glb")

    o3d.io.write_point_cloud(ply_path, fused_pcd)
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

    return {"pointcloud_ply": ply_path, "mesh_ply": mesh_ply_path, "mesh_glb": glb_path}
