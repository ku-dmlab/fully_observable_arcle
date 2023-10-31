
import numpy as np
cimport numpy as np
cimport cython

cdef (int, int, int, int) _get_bbox(unsigned char[:, ::1] img) noexcept nogil

cdef rotate(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selected,
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel_area, 
    np.ndarray[np.uint8_t, ndim=2] objsel_bg, 
    tuple[int, int] objsel_coord, 
    char objsel_active, 
    char objsel_rot,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection,
    k)

cdef move(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selected,
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel_area, 
    np.ndarray[np.uint8_t, ndim=2] objsel_bg, 
    tuple[int, int] objsel_coord, 
    char objsel_active, 
    char objsel_rot,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection,
    char d)

cdef flip(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selected,
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel_area, 
    np.ndarray[np.uint8_t, ndim=2] objsel_bg, 
    tuple[int, int] objsel_coord, 
    char objsel_active, 
    char objsel_rot,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection,
    axis)


cdef copy(np.ndarray[np.uint8_t, ndim=2] inp, inp_dim, np.ndarray[np.uint8_t, ndim=2] grid, clip, clip_dim, np.ndarray[np.npy_bool, ndim=2, cast=True] selection, source)

cdef void paste(grid, np.ndarray[np.uint8_t, ndim=2] clip, clip_dim, np.ndarray[np.npy_bool, ndim=2, cast=True] selection)
