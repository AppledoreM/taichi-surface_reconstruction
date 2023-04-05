#!/usr/bin/env python
# coding=utf-8

import taichi as ti
from srtool import SRTool

ti.init(arch = ti.cuda, device_memory_GB = 5)

bounding_box = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
voxel_size = 0.01
particle_radius = 0.005
sr = SRTool(bounding_box, voxel_size=voxel_size, particle_radius = particle_radius)

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
    for i in range(1000):
        sr.dual_contouring(isovalue, smooth_range, num_particles = 100 * 100 * 100, pos = cube_point_clouds)
        sr.export_mesh("./test.ply")









