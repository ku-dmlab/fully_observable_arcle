import numpy as np
cimport numpy as np
cimport cython

cdef void color(unsigned char[:, ::1] grid, unsigned char[:, ::1] selection, int color) noexcept nogil

cdef void floodfill(unsigned char[:, ::1] grid, tuple[int, int] grid_dim, unsigned char[:, ::1] selection, int color) noexcept nogil
