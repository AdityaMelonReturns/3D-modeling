"""Build a real mesh from an already-downloaded defect_pointcloud.ply -
entirely locally, no pod needed. Uses alpha-shape reconstruction (fast,
no normal-orientation step - see pipeline/mesh_export.py's build_mesh
docstring for why Poisson kept hanging on the pod regardless of cloud size).

Colors the mesh with the REAL photo colors already baked into the point
cloud (from SAM2 backprojection) via nearest-neighbor transfer, rather than
the signed-distance heatmap - we don't have the plane normal/point_on_plane
saved to disk for the 7 sets that were killed before mesh export, so this
is what's actually available, and it's arguably more useful anyway (an
actual photo-textured 3D shape, not an abstract heatmap).

Usage: python build_meshes_locally.py <defect_id> [alpha_multiplier]
Writes output/<defect_id>/mesh.ply and mesh.glb.
"""
import sys
import os

import numpy as np
import open3d as o3d
import trimesh

HERE = os.path.dirname(__file__)


def transfer_colors(mesh, source_pcd):
    """Alpha-shape mesh vertices are a subset of the original point cloud's
    own points (not new interpolated ones), so nearest-neighbor lookup is an
    exact match, not an approximation."""
    tree = o3d.geometry.KDTreeFlann(source_pcd)
    src_colors = np.asarray(source_pcd.colors)
    verts = np.asarray(mesh.vertices)
    out = np.zeros((len(verts), 3))
    for i, v in enumerate(verts):
        _, idx, _ = tree.search_knn_vector_3d(v, 1)
        out[i] = src_colors[idx[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(out)


def main():
    defect_id = sys.argv[1]
    alpha_multiplier = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    out_dir = os.path.join(HERE, "output", defect_id)
    ply_path = os.path.join(out_dir, "defect_pointcloud.ply")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"{defect_id}: {len(pcd.points)} points loaded, has_colors={pcd.has_colors()}")

    distances = pcd.compute_nearest_neighbor_distance()
    avg_spacing = float(np.mean(distances)) if len(distances) else 0.01
    alpha = max(avg_spacing * alpha_multiplier, 1e-6)
    print(f"  avg point spacing={avg_spacing:.5f}, alpha={alpha:.5f}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    print(f"  mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    if len(mesh.triangles) == 0:
        # alpha too small for this cloud's density - widen once and retry
        alpha *= 3
        print(f"  0 triangles, retrying with wider alpha={alpha:.5f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        print(f"  mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    mesh.compute_vertex_normals()
    if pcd.has_colors():
        transfer_colors(mesh, pcd)

    mesh_ply_path = os.path.join(out_dir, "mesh.ply")
    glb_path = os.path.join(out_dir, "mesh.glb")
    o3d.io.write_triangle_mesh(mesh_ply_path, mesh)

    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None,
        process=False,
    )
    tm.export(glb_path)
    print(f"  exported: {mesh_ply_path}, {glb_path}")


if __name__ == "__main__":
    main()
