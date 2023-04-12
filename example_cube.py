#!/usr/bin/env python
# coding=utf-8

import taichi as ti
from srtool import SRTool

ti.init(arch = ti.cuda, device_memory_GB = 5)

bounding_box = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
voxel_size = 0.01
particle_radius = 0.005
sr = SRTool(bounding_box, voxel_size=voxel_size, particle_radius = particle_radius, max_num_vertices = 20000, max_num_indices = 600000)

# This is a cube of size 100 * 100 * 100
cube_point_clouds = ti.Vector.field(3, ti.f32, 100 * 100 * 100)

@ti.kernel
def make_cube(point_cloud : ti.template()):
    base_coord = ti.Vector([0.25, 0.25, 0.25])
    for i, j, k in ti.ndrange(100, 100, 100):
        index = i + j * 100 + k * 100 * 100
        cube_point_clouds[index] = base_coord + ti.Vector([i, j, k]) * particle_radius



if __name__ == "__main__":
    make_cube(cube_point_clouds)
    isovalue = 0
    smooth_range = 0.03

    window = ti.ui.Window("Example for Surface Reconstruction GUI", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5, 2, 2)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        sr.dual_contouring(isovalue, smooth_range, num_particles = 100 * 100 * 100, pos = cube_point_clouds)
        scene.mesh(sr.mesh_vertex, sr.mesh_index, vertex_count = sr.num_vertices[None], index_count = sr.num_indices[None])

        # Draw 3d-lines in the scene
        canvas.scene(scene)
        window.show()











