import bpy
import bmesh
from mathutils import Vector, Matrix, Quaternion

def projected_tangent(v_from, v_to, normal):
    e = (v_to - v_from)
    e_proj = e - normal * e.dot(normal)
    if e_proj.length_squared == 0:
        return None
    return e_proj.normalized()

def make_basis_from_normal_and_tangent(normal, tangent):
    z = normal.normalized()
    if tangent is None or tangent.length_squared == 0:
        helper = Vector((1, 0, 0))
        if abs(z.dot(helper)) > 0.999:
            helper = Vector((0, 1, 0))
        x = helper.cross(z).normalized()
    else:
        x = tangent - z * tangent.dot(z)
        if x.length_squared == 0:
            helper = Vector((1, 0, 0))
            if abs(z.dot(helper)) > 0.999:
                helper = Vector((0, 1, 0))
            x = helper.cross(z).normalized()
        else:
            x = x.normalized()
    y = z.cross(x).normalized()
    mat = Matrix((x, y, z)).transposed()
    return mat

def quad_size_along_basis(face, basis_x, basis_y):
    verts = [v.co for v in face.verts]
    vcount = len(verts)
    sizes_x, sizes_y = [], []
    for i in range(vcount):
        a = verts[i]
        b = verts[(i+1) % vcount]
        edge = b - a
        sizes_x.append(abs(edge.dot(basis_x)))
        sizes_y.append(abs(edge.dot(basis_y)))
    sx = max(sizes_x) or 0.01
    sy = max(sizes_y) or 0.01
    return sx, sy

def average_quaternions(quats):
    """
    Compute a simple averaged quaternion by summing (with sign alignment) and normalizing.
    This is not a geodesic Slerp average but works well as a pragmatic mean for nearby rotations.
    """
    if not quats:
        return None
    # Start with first quaternion
    q_sum = quats[0].copy()
    for q in quats[1:]:
        # Align sign to avoid antipodal cancellation
        if q_sum.dot(q) < 0.0:
            q = -q
        q_sum = Quaternion((q_sum.w + q.w, q_sum.x + q.x, q_sum.y + q.y, q_sum.z + q.z))
    q_sum.normalize()
    return q_sum

def compute_face_quaternion_world(eval_obj, face):
    # compute basis for face in local (evaluated) coords
    normal = face.normal.copy()
    v0, v1 = face.verts[0].co, face.verts[1].co
    tangent = projected_tangent(v0, v1, normal)
    basis_mat = make_basis_from_normal_and_tangent(normal, tangent)
    # convert basis into world (3x3)
    world_basis = eval_obj.matrix_world.to_3x3() @ basis_mat
    # convert to quaternion (world space)
    q = world_basis.to_quaternion()
    return q

def process_object_apply_rotation(obj):
    """ Compute mean rotation from quads, set object rotation, rotate base mesh vertices by inverse local rotation,
        and recompute origin to geometry median.
    """
    if obj.type != 'MESH':
        print(f"Skipping non-mesh object: {obj.name}")
        return 0

    # Use evaluated mesh to compute face normals/orientations (includes modifiers)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bm.faces.ensure_lookup_table()

    face_quats = []
    quad_count = 0
    for face in bm.faces:
        if len(face.verts) != 4:
            continue
        q_world = compute_face_quaternion_world(eval_obj, face)
        face_quats.append(q_world)
        quad_count += 1

    # cleanup evaluated mesh objects
    bm.free()
    eval_obj.to_mesh_clear()

    if quad_count == 0:
        print(f"No quads found for object {obj.name}; skipping rotation change.")
        return 0

    # Average quaternions (world space)
    mean_world_q = average_quaternions(face_quats)
    if mean_world_q is None:
        print(f"Failed to compute mean quaternion for {obj.name}")
        return 0

    # Determine local quaternion that produces this world rotation
    # If object has a parent, need to convert from world to local space
    if obj.parent:
        parent_world_q = obj.parent.matrix_world.to_quaternion()
        local_q = parent_world_q.inverted() @ mean_world_q
    else:
        local_q = mean_world_q.copy()

    # Ensure object uses quaternion rotation mode
    prev_mode = obj.rotation_mode
    obj.rotation_mode = 'QUATERNION'

    # Apply the local rotation to the object (this will change the object's matrix_world)
    obj.rotation_quaternion = local_q

    # Now rotate the object's base mesh vertices by inverse of the *local* rotation
    # so that world-space vertex positions remain (approximately) identical.
    inv_local_mat = local_q.to_matrix().inverted()

    # Work on the object's actual mesh datablock (base mesh)
    mesh = obj.data

    # If mesh is linked/shared between multiple objects, we operate in-place (modify shared mesh).
    # Optionally, if you want per-object unique mesh, you could make a copy here.

    # Rotate each vertex coordinate in object-local space
    # NOTE: If the mesh has shape keys, modifiers, etc., this modifies base geometry only.
    for v in mesh.vertices:
        v.co = inv_local_mat @ v.co

    # Update mesh to ensure changes are reflected
    mesh.update()

    # Recompute object origin to geometry median (this will shift object.location so geometry stays put)
    # Make this object active & selected for the operator to work correctly.
    prev_active = bpy.context.view_layer.objects.active
    prev_selected = list(bpy.context.selected_objects)

    # select only this object and make it active
    bpy.ops.object.mode_set(mode='OBJECT')  # ensure object mode
    for o in prev_selected:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Origin set to geometry median
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

    # restore selection / active
    for o in prev_selected:
        o.select_set(True)
    bpy.context.view_layer.objects.active = prev_active

    # restore rotation mode if it was different (keep quaternion as the stored rotation though)
    obj.rotation_mode = 'QUATERNION'  # keep quaternion mode because we explicitly set quaternion
    # If you want to restore prev_mode uncomment next line (but it may convert quaternion to Euler)
    # obj.rotation_mode = prev_mode

    print(f"Processed {obj.name}: {quad_count} quads -> applied rotation (world) {mean_world_q}.")

    return quad_count

def main():
    selected_meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not selected_meshes:
        print("Select at least one mesh object.")
        return

    total_quads = 0
    for obj in selected_meshes:
        total_quads += process_object_apply_rotation(obj)

    print(f"\nDone — processed {len(selected_meshes)} objects, {total_quads} quads total.\n")

if __name__ == "__main__":
    main()
