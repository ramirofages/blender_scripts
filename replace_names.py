import bpy

for obj in bpy.context.scene.objects:
    obj.name = obj.name.replace(".", "-")