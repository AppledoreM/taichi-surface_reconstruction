#!/usr/bin/env python
# coding=utf-8
import taichi as ti


@ti.kernel
def copy_index_field_to_array(src: ti.template(), dst: ti.types.ndarray(), length: ti.i32):
    for i in range(length // 3):
        dst[i, 0] = src[i * 3]
        dst[i, 1] = src[i * 3 + 1]
        dst[i, 2] = src[i * 3 + 2]

@ti.kernel
def copy_vertex_field_to_array(src: ti.template(), dst: ti.types.ndarray(), length: ti.i32):
    for i in range(length):
        dst[i, 0] = src[i][0]
        dst[i, 1] = src[i][1]
        dst[i, 2] = src[i][2]


@ti.kernel
def array_to_field_copy(src: ti.types.ndarray(), dst: ti.template(), length: ti.i32):
    for i in range(length):
        dst[i][0] = src[i, 0]
        dst[i][1] = src[i, 1]
        dst[i][2] = src[i, 2]







