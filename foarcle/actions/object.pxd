
import numpy as np
cimport numpy as np
cimport cython

cdef (int, int, int, int) _get_bbox(unsigned char[:, ::1] img) noexcept nogil

cdef rotate(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selected,
    char active, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_,
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_sel, 
    tuple[int, int] object_dim, 
    tuple[int, int] object_pos, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] background, 
    char rotation_parity,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection,
    char k)

cdef move(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selected,
    char active, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_,
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_sel, 
    tuple[int, int] object_dim, 
    tuple[int, int] object_pos, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] background, 
    char rotation_parity,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection,
    char d)

cdef flip(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selected,
    char active, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_,
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_sel, 
    tuple[int, int] object_dim, 
    tuple[int, int] object_pos, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] background, 
    char rotation_parity,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection,
    char axis)

cdef copy(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] inp, 
    tuple[int, int] inp_dim, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid, 
    tuple[int, int] grid_dim, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] clip, 
    tuple[int, int] clip_dim,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection,
    char source)

cdef void paste(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] clip, 
    tuple[int, int] clip_dim, 
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection)