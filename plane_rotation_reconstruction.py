import bpy
import bmesh
from mathutils import Vector, Matrix, Quaternion
from math import atan2, cos, sin

def newell_normal(verts):
    """Compute polygon normal using Newell's method. verts: list of Vector (in same coord space)."""
    nx = ny = nz = 0.0
    n = len(verts)
    for i in range(n):
        v_curr = verts[i]
        v_next = verts[(i + 1) % n]
        nx += (v_curr.y - v_next.y) * (v_curr.z + v_next.z)
        ny += (v_curr.z - v_next.z) * (v_curr.x + v_next.x)
        nz += (v_curr.x - v_next.x) * (v_curr.y + v_next.y)
    nvec = Vector((nx, ny, nz))
    if nvec.length_squared == 0.0:
        return None
    return nvec.normalized()

def face_major_axis_by_pca(verts, normal):
    """
    Project vertices into plane orthonormal basis and compute 2x2 covariance.
    Return a 3D tangent vector (normalized) representing the major PCA axis in world/local coords.
    """
    # build any orthonormal basis (u, v, normal) for the plane
    # pick helper that is not parallel to normal
    helper = Vector((1.0, 0.0, 0.0))
    if abs(normal.dot(helper)) > 0.999:
        helper = Vector((0.0, 1.0, 0.0))
    u = helper.cross(normal).normalized()  # first in-plane axis
    v = normal.cross(u).normalized()       # second in-plane axis

    # compute centroid
    centroid = Vector((0.0, 0.0, 0.0))
    for p in verts:
        centroid += p
    centroid /= len(verts)

    # build 2D coordinates in (u,v)
    xs = []
    ys = []
    for p in verts:
        d = p - centroid
        xs.append(d.dot(u))
        ys.append(d.dot(v))

    # compute covariance elements
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov_xx = cov_xy = cov_yy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        cov_xx += dx * dx
        cov_xy += dx * dy
        cov_yy += dy * dy

    # if degenerate (all points same), fallback to first edge vector
    if cov_xx + cov_yy <= 1e-12:
        # fallback vector: first edge direction projected to plane
        fallback = (verts[1] - verts[0]) - normal * (verts[1] - verts[0]).dot(normal)
        if fallback.length_squared == 0:
            # absolute fallback
            return u
        return fallback.normalized()

    # covariance matrix is [cov_xx cov_xy; cov_xy cov_yy]
    # principal eigenvector angle:
    theta = 0.5 * atan2(2.0 * cov_xy, cov_xx - cov_yy)
    # principal vector in (u,v) coords
    cx = cos(theta)
    cy = sin(theta)
    # map to 3D
    major = (u * cx) + (v * cy)
    if major.length_squared == 0:
        return u
    return major.normalized()

def compute_face_basis_from_vertices(verts_local, eval_obj):
    """
    verts_local: list of vertices in evaluated object's local coordinates (Vector)
    Returns: basis_mat (Matrix 3x3) in evaluated local space where columns are X (tangent), Y, Z (normal)
    """
    # compute robust normal with Newell
    normal = newell_normal(verts_local)
    if normal is None:
        # fallback to Blender face normal via cross of first two edges
        e = verts_local[1] - verts_local[0]
        f = verts_local[2] - verts_local[0]
        cz = e.cross(f)
        if cz.length_squared == 0:
            cz = Vector((0.0, 0.0, 1.0))
        normal = cz.normalized()

    # compute major in-plane axis using PCA
    tangent = face_major_axis_by_pca(verts_local, normal)

    # build orthonormal basis: x = tangent (in-plane), z = normal, y = z cross x
    z = normal.normalized()
    x = tangent - z * tangent.dot(z)
    if x.length_squared == 0:
        # fallback choose axis
        helper = Vector((1.0, 0.0, 0.0))
        if abs(z.dot(helper)) > 0.999:
            helper = Vector((0.0, 1.0, 0.0))
        x = helper.cross(z).normalized()
    else:
        x = x.normalized()
    y = z.cross(x).normalized()

    # columns are x,y,z -> construct matrix and transpose so columns become basis vectors
    mat = Matrix((x, y, z)).transposed()
    return mat

def compute_face_quaternion_world(eval_obj, face):
    # collect face vertices in evaluated local coordinates
    verts_local = [v.co.copy() for v in face.verts]
    basis_mat = compute_face_basis_from_vertices(verts_local, eval_obj)
    # convert basis to world space
    world_basis = eval_obj.matrix_world.to_3x3() @ basis_mat
    return world_basis.to_quaternion()

def average_quaternions(quats):
    if not quats:
        return None
    q_sum = Quaternion((0.0, 0.0, 0.0, 0.0))
    ref = quats[0]
    for q in quats:
        q_al = q
        if ref.dot(q) < 0.0:
            q_al = -q
        q_sum.w += q_al.w
        q_sum.x += q_al.x
        q_sum.y += q_al.y
        q_sum.z += q_al.z
    # Blender Quaternion has .magnitude instead of .length
    if q_sum.magnitude == 0:
        return None
    q_sum.normalize()
    return q_sum


def process_object_apply_rotation(obj):
    if obj.type != 'MESH':
        print(f"Skipping non-mesh object: {obj.name}")
        return 0

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bm.faces.ensure_lookup_table()

    face_quats = []
    quad_count = 0
    for face in bm.faces:
        if len(face.verts) < 3:
            continue
        q_world = compute_face_quaternion_world(eval_obj, face)
        face_quats.append(q_world)
        quad_count += 1

    bm.free()
    eval_obj.to_mesh_clear()

    if quad_count == 0:
        print(f"No valid faces for object {obj.name}; skipping.")
        return 0

    mean_world_q = average_quaternions(face_quats)
    if mean_world_q is None:
        print(f"Failed to average quaternions for {obj.name}")
        return 0

    # convert world mean to local quaternion (respect parent)
    if obj.parent:
        parent_world_q = obj.parent.matrix_world.to_quaternion()
        local_q = parent_world_q.inverted() @ mean_world_q
    else:
        local_q = mean_world_q.copy()

    # set rotation mode and apply rotation
    prev_mode = obj.rotation_mode
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = local_q

    # rotate base mesh vertices by inverse local rotation so world geometry remains
    inv_local_mat = local_q.to_matrix().inverted()
    mesh = obj.data

    # If multiple objects share the mesh, this will modify it for all users.
    # If you don't want that, uncomment the following to make the mesh single-user:
    # if mesh.users > 1:
    #     obj.data = mesh.copy()
    #     mesh = obj.data

    for v in mesh.vertices:
        v.co = inv_local_mat @ v.co

    mesh.update()

    # Recompute origin to geometry median (optional; keeps origin at geometry center)
    prev_active = bpy.context.view_layer.objects.active
    prev_selected = list(bpy.context.selected_objects)
    bpy.ops.object.mode_set(mode='OBJECT')
    for o in prev_selected:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    # restore
    for o in prev_selected:
        o.select_set(True)
    bpy.context.view_layer.objects.active = prev_active

    # keep quaternion rotation mode (we set it above)
    obj.rotation_mode = 'QUATERNION'

    print(f"Processed {obj.name}: evaluated {quad_count} faces; applied rotation (world) {mean_world_q}.")
    return quad_count

def main():
    selected_meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not selected_meshes:
        print("Select at least one mesh object.")
        return

    total = 0
    for obj in selected_meshes:
        total += process_object_apply_rotation(obj)

    print(f"\nDone — processed {len(selected_meshes)} objects, {total} faces total.\n")

if __name__ == "__main__":
    main()
