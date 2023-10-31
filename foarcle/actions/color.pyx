
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
cdef void flood_fill_recursive(unsigned char[:, ::1] grid, int grid_dim_x, int grid_dim_y, int x, int y, int color, int start_color) noexcept nogil:
    if x < 0 or x >= grid_dim_x: return
    if y < 0 or y >= grid_dim_y: return
    
    if grid[x][y] == color: return
    if grid[x][y] != start_color: return
    
    grid[x][y] = color

    flood_fill_recursive(grid, grid_dim_x, grid_dim_y, x - 1, y, color, start_color)
    flood_fill_recursive(grid, grid_dim_x, grid_dim_y, x + 1, y, color, start_color)
    flood_fill_recursive(grid, grid_dim_x, grid_dim_y, x, y + 1, color, start_color)
    flood_fill_recursive(grid, grid_dim_x, grid_dim_y, x, y - 1, color, start_color)

@cython.boundscheck(False)
cdef void color(unsigned char[:, ::1] grid, unsigned char[:, ::1] selection, int color) noexcept nogil:
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = grid[i, j] * (1 - selection[i, j]) + color * selection[i, j]

@cython.boundscheck(False)
cdef void floodfill(unsigned char[:, ::1] grid, tuple[int, int] grid_dim, unsigned char[:, ::1] selection, int color) noexcept nogil:
    cdef int x = -1, y = -1, 
    cdef int i, j
    for i in range(selection.shape[0]):
        for j in range(selection.shape[1]):
            if selection[i, j] == 1:
                if  x == -1 and y == -1:
                    x = i
                    y = j
                else:
                    return
    if x == -1 and y == -1:
        return
    cdef int start_color = grid[x][y]

    flood_fill_recursive(grid, grid_dim[0], grid_dim[1], x, y, color, start_color)
