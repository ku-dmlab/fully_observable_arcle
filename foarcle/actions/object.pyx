
import numpy as np
cimport numpy as np
cimport cython
cdef extern from "limits.h":
    cdef int INT_MAX
    cdef int INT_MIN

@cython.boundscheck(False)
cdef (int, int, int, int) _get_bbox(unsigned char[:, ::1] img) noexcept nogil:
    cdef int rmin=INT_MAX, rmax=INT_MIN, cmin=INT_MAX, cmax=INT_MIN
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j]:
                if rmin > i:
                    rmin = i
                if rmax < i:
                    rmax = i
                if cmin > j:
                    cmin = j
                if cmax < j:
                    cmax = j
    return rmin, rmax, cmin, cmax

cdef _init_objsel(
    np.ndarray[np.uint8_t, ndim=2] grid,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selected, 
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection, 
    np.ndarray[np.uint8_t, ndim=2] objsel, 
    np.ndarray[np.uint8_t, ndim=2] objsel_area, 
    np.ndarray[np.uint8_t, ndim=2] objsel_bg, 
    tuple[int, int] objsel_coord, 
    char objsel_active, 
    char objsel_rot):

    cdef int xmin, xmax, ymin, ymax
    cdef int H, W

    xmin, xmax, ymin, ymax = _get_bbox(selection)
    if xmax != INT_MIN:

        H = xmax - xmin + 1
        W = ymax - ymin + 1

        objsel = np.zeros((H, W), dtype=np.uint8)

        np.copyto(dst=objsel, src=grid[xmin:xmax+1, ymin:ymax+1], where=selection[xmin:xmax+1, ymin:ymax+1].astype(np.bool_))

        objsel_area = np.copy(selection[xmin:xmax+1, ymin:ymax+1])

        objsel_bg = np.copy(grid)
        np.copyto(dst=objsel_bg, src=0, where=selection.astype(np.bool_))

        objsel_coord = (int(xmin), int(ymin)) 
        objsel_active = True
        objsel_rot = 0

        selected = np.copy(selection)

    elif objsel_active: 
        xmin, ymin = objsel_coord
        xmax = xmin + objsel.shape[0] - 1
        ymax = ymin + objsel.shape[1] - 1

    else:
        xmin = INT_MIN
    
    return xmin, xmax, ymin, ymax, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot

cdef void get_it_copied(
    tuple[int, int] objsel_coord,
    tuple[int, int] grid_dim,
    unsigned char[:, ::1] grid,
    unsigned char[:, ::1] p) noexcept nogil:

    cdef int x, y, h, w, gh, gw
    cdef int stx, edx, sty, edy, i , j
    x, y = objsel_coord
    h = p.shape[0]
    w = p.shape[1]
    gh, gw = grid_dim

    if x + h <= 0 or x >= gh or y + w <= 0 or y >= gw:
        return

    stx = 0 if x < 0 else x
    sty = 0 if y < 0 else y
    edx = gh if gh < x + h else x + h
    edy = gw if gw < y + w else y + w
    for i in range(stx, edx):
        for j in range(sty, edy):
            grid[i, j] = p[i - x, j - y]


cdef void _apply_patch(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    tuple[int, int] grid_dim, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] objsel,
    tuple[int, int] objsel_coord,
    np.ndarray[np.uint8_t, ndim=2] objsel_bg):

    grid[:] = 0
    get_it_copied(objsel_coord, grid_dim, grid, objsel)
    np.copyto(grid, objsel_bg, where=(grid == 0))

cdef void _apply_sel(
    unsigned char[:, ::1] selected,
    tuple[int, int] grid_dim, 
    unsigned char[:, ::1] objsel_area,
    tuple[int, int] objsel_coord):
    
    selected[:] = 0
    get_it_copied(objsel_coord, grid_dim, selected, objsel_area)

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
    k):

    assert 0 < k < 4

    xmin, xmax, ymin, ymax, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = _init_objsel(
        grid, selected, selection, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot)

    if xmin == INT_MIN:
        return grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot

    if k == 2:
        objsel = objsel[::-1, ::-1].copy(order='C')

    elif objsel.shape[0] % 2 == objsel.shape[1] % 2:
        cx = (xmax + xmin) * 0.5
        cy = (ymax + ymin) * 0.5
        x, y = objsel_coord
        objsel_coord = ( int(np.floor(cx - cy + y)), int(np.floor(cy - cx + x)))

    else:
        cx = (xmax + xmin) * 0.5
        cy = (ymax + ymin) * 0.5
        objsel_rot += k
        sig = (k + 2) % 4 - 2
        mod = 1 - objsel_rot % 2
        mx = min(cx + sig * (cy - ymin), cx + sig * (cy - ymax)) + mod
        my = min(cy - sig * (cx - xmin), cy - sig * (cx - xmax)) + mod
        objsel_coord = (int(np.floor(mx)), int(np.floor(my)))
    
    if k == 1:
        objsel = objsel.swapaxes(0, 1)[::-1, :].copy(order='C')
        objsel_area = objsel_area.swapaxes(0, 1)[::-1, :].copy(order='C')
    elif k == 3:
        objsel = objsel.swapaxes(0, 1)[:, ::-1].copy(order='C')
        objsel_area = objsel_area.swapaxes(0, 1)[:, ::-1].copy(order='C')
    _apply_patch(grid, grid_dim, objsel, objsel_coord, objsel_bg)
    _apply_sel(selected, grid_dim, objsel_area, objsel_coord)

    return grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot

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
    char d):
    assert 0 <= d < 4, f"d is {d}"

    xmin, xmax, ymin, ymax, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = _init_objsel(
        grid, selected, selection, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot)

    if xmin == INT_MIN:
        return grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot
    
    if d == 0:
        dirX, dirY = -1, 0
    elif d == 1:
        dirX, dirY = 1, 0
    elif d == 2:
        dirX, dirY = 0, 1
    elif d == 3:
        dirX, dirY = 0, -1

    x, y = objsel_coord
    objsel_coord = (int(x + dirX), int(y + dirY))
    _apply_patch(grid, grid_dim, objsel, objsel_coord, objsel_bg)
    _apply_sel(selected, grid_dim, objsel_area, objsel_coord)

    return grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot

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
    axis):
    
    assert axis == "H" or axis == "V"
        
    xmin, xmax, ymin, ymax, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = _init_objsel(
        grid, selected, selection, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot)

    if xmin == INT_MIN:
        return grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot
    
    if axis == "H":
        objsel = np.fliplr(objsel).copy(order='C')
        objsel_area = np.fliplr(objsel_area).copy(order='C')
    else:
        objsel = np.flipud(objsel).copy(order='C')
        objsel_area = np.flipud(objsel_area).copy(order='C')

    _apply_patch(grid, grid_dim, objsel, objsel_coord, objsel_bg)
    _apply_sel(selected, grid_dim, objsel_area, objsel_coord)

    return grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot

cdef copy(np.ndarray[np.uint8_t, ndim=2] inp, inp_dim, np.ndarray[np.uint8_t, ndim=2] grid, clip, clip_dim, np.ndarray[np.npy_bool, ndim=2, cast=True] selection, source):
    assert source == "I" or source == "O"
    
    xmin, xmax, ymin, ymax = _get_bbox(selection)
    if xmax == INT_MIN:
        return clip_dim

    H = xmax - xmin + 1
    W = ymax - ymin + 1
    clip[:] = 0

    if source == "I":
        np.copyto(clip[:H, :W], inp[xmin:xmin + H, ymin:ymin + W], where=np.logical_and(inp[xmin:xmin + H, ymin:ymin + W] > 0, selection[xmin:xmin + H, ymin:ymin + W]))
    elif source == "O":
        np.copyto(clip[:H, :W], grid[xmin:xmin + H, ymin:ymin + W], where=np.logical_and(grid[xmin:xmin + H, ymin:ymin + W] > 0, selection[xmin:xmin + H, ymin:ymin + W]))
    
    return H, W

cdef void paste(grid, np.ndarray[np.uint8_t, ndim=2] clip, clip_dim, np.ndarray[np.npy_bool, ndim=2, cast=True] selection):
    xmin, xmax, ymin, _ = _get_bbox(selection)
    if xmax == INT_MIN:
        return

    cdef int max_H, max_W, H, W
    max_H = clip.shape[0]
    max_W = clip.shape[1]

    if xmin >= max_H or ymin >= max_W:
        return 
    
    H, W = clip_dim

    if H == 0 or W == 0:
        return 
    
    edx = min(xmin + H, max_H)
    edy = min(ymin + W, max_W)
    np.copyto(grid[xmin:edx, ymin:edy], clip[:edx - xmin, :edy - ymin], where=clip[:edx - xmin, :edy - ymin] > 0)
