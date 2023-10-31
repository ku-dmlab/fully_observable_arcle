
import numpy as np
cimport numpy as np

cdef (int, int) crop_grid(np.ndarray[np.uint8_t, ndim=2] grid, tuple[int, int] grid_dim, np.ndarray[np.npy_bool, ndim=2, cast=True] selection)
