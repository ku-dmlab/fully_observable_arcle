
import numpy as np
cimport numpy as np
from foarcle.actions.object cimport _get_bbox
cdef extern from "limits.h":
    cdef int INT_MAX
    cdef int INT_MIN

cdef (int, int) crop_grid(
    np.ndarray[np.uint8_t, ndim=2] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection):


    cdef int xmin, xmax, ymin, ymax, H, W

    xmin, xmax, ymin, ymax = _get_bbox(selection)
    if xmax == INT_MIN:
        return grid_dim

    H = xmax - xmin + 1
    W = ymax - ymin + 1

    patch = np.zeros((H, W), dtype=np.uint8)
    np.copyto(dst=patch, src=grid[xmin:xmax + 1, ymin:ymax + 1], where=selection[xmin:xmax + 1, ymin:ymax + 1])
    grid[:, :] = 0
    np.copyto(grid[:H, :W], patch)
    return H, W
