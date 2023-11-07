
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
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid,
    np.ndarray[np.npy_bool, ndim=2, cast=True, mode='c'] selected, 
    np.ndarray[np.npy_bool, ndim=2, cast=True, mode='c'] selection, 
    char active, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_sel, 
    tuple[int, int] object_dim, 
    tuple[int, int] object_pos, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] background, 
    char rotation_parity):

    cdef int xmin, xmax, ymin, ymax
    cdef int H, W

    xmin, xmax, ymin, ymax = _get_bbox(selection)
    if xmax != INT_MIN:

        H = xmax - xmin + 1
        W = ymax - ymin + 1

        object_ = np.zeros_like(grid)
        np.copyto(dst=object_[:H, :W], src=grid[xmin:xmax+1, ymin:ymax+1], where=selection[xmin:xmax+1, ymin:ymax+1].astype(np.bool_))

        object_sel = np.zeros_like(selection)
        np.copyto(dst=object_sel[:H, :W], src=selection[xmin:xmax+1, ymin:ymax+1])

        background = np.copy(grid)
        np.copyto(dst=background, src=0, where=selection.astype(np.bool_))

        object_dim = (H, W)
        object_pos = (xmin, ymin)
        active = True
        rotation_parity = 0

        selected = np.copy(selection)

    elif active: 
        xmin, ymin = object_pos
        xmax = xmin + object_dim[0] - 1
        ymax = ymin + object_dim[1] - 1

    else:
        xmin = INT_MIN
    
    return (xmin, xmax, ymin, ymax, selected, 
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity)

cdef void get_it_copied(
    tuple[int, int] object_dim,
    tuple[int, int] object_pos,
    tuple[int, int] grid_dim,
    unsigned char[:, ::1] grid,
    unsigned char[:, ::1] p) noexcept nogil:

    cdef int x, y, h, w, gh, gw
    cdef int stx, edx, sty, edy, i , j
    x, y = object_pos
    h, w = object_dim
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
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_,
    tuple[int, int] object_dim,
    tuple[int, int] object_pos,
    np.ndarray[np.uint8_t, ndim=2] background):

    grid[:] = 0
    get_it_copied(object_dim, object_pos, grid_dim, grid, object_)
    np.copyto(grid, background, where=(grid == 0))

cdef void _apply_sel(
    unsigned char[:, ::1] selected,
    tuple[int, int] grid_dim, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] object_sel,
    tuple[int, int] object_dim,
    tuple[int, int] object_pos):
    
    selected[:] = 0
    get_it_copied(object_dim, object_pos, grid_dim, selected, object_sel)

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
    char k):

    assert 0 < k < 4

    (xmin, xmax, ymin, ymax, selected, 
    active, object_, object_sel, object_dim, object_pos, background, rotation_parity) = _init_objsel(
        grid, selected, selection, 
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity)

    if xmin == INT_MIN:
        return grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity

    if k == 2:
        object_[:object_dim[0], :object_dim[1]] = object_[:object_dim[0]:-1, :object_dim[1]:-1].copy(order='C')

    elif object_dim[0] % 2 == object_dim[1] % 2:
        cx = (xmax + xmin) * 0.5
        cy = (ymax + ymin) * 0.5
        x, y = object_pos
        object_pos = ( int(np.floor(cx - cy + y)), int(np.floor(cy - cx + x)))

    else:
        cx = (xmax + xmin) * 0.5
        cy = (ymax + ymin) * 0.5
        rotation_parity += k
        rotation_parity %= 2
        sig = (k + 2) % 4 - 2
        mod = 1 - rotation_parity
        mx = min(cx + sig * (cy - ymin), cx + sig * (cy - ymax)) + mod
        my = min(cy - sig * (cx - xmin), cy - sig * (cx - xmax)) + mod
        object_pos = (int(np.floor(mx)), int(np.floor(my)))
    
    if k == 1:
        object_new = np.zeros_like(object_)
        object_new[:object_dim[1], :object_dim[0]] = object_[
            :object_dim[0], :object_dim[1]].swapaxes(0, 1)[::-1, :].copy(order='C')
        object_sel_new = np.zeros_like(object_sel)
        object_sel_new[:object_dim[1], :object_dim[0]] = object_sel[
            :object_dim[0], :object_dim[1]].swapaxes(0, 1)[::-1, :].copy(order='C')
        object_ = object_new
        object_sel = object_sel_new
        object_dim = (object_dim[1], object_dim[0])
    elif k == 3:
        object_new = np.zeros_like(object_)
        object_new[:object_dim[1], :object_dim[0]] = object_[
            :object_dim[0], :object_dim[1]].swapaxes(0, 1)[:, ::-1].copy(order='C')
        object_sel_new = np.zeros_like(object_sel)
        object_sel_new[:object_dim[1], :object_dim[0]] = object_sel[
            :object_dim[0], :object_dim[1]].swapaxes(0, 1)[:, ::-1].copy(order='C')
        object_ = object_new
        object_sel = object_sel_new
        object_dim = (object_dim[1], object_dim[0])

    _apply_patch(grid, grid_dim, object_, object_dim, object_pos, background)
    _apply_sel(selected, grid_dim, object_sel, object_dim, object_pos)

    return grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity

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
    char d):
    assert 0 <= d < 4, f"d is {d}"

    (xmin, xmax, ymin, ymax, selected, 
    active, object_, object_sel, object_dim, object_pos, background, rotation_parity) = _init_objsel(
        grid, selected, selection, 
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity)

    if xmin == INT_MIN:
        return grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity
    
    if d == 0:
        dirX, dirY = -1, 0
    elif d == 1:
        dirX, dirY = 1, 0
    elif d == 2:
        dirX, dirY = 0, 1
    elif d == 3:
        dirX, dirY = 0, -1

    x, y = object_pos
    object_pos = (int(x + dirX), int(y + dirY))
    _apply_patch(grid, grid_dim, object_, object_dim, object_pos, background)
    _apply_sel(selected, grid_dim, object_sel, object_dim, object_pos)

    return grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity

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
    char axis):
    
    cdef int h, w

    assert axis == "H" or axis == "V"
        
    (xmin, xmax, ymin, ymax, selected, 
    active, object_, object_sel, object_dim, object_pos, background, rotation_parity) = _init_objsel(
        grid, selected, selection, 
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity)

    if xmin == INT_MIN:
        return grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity
    
    h, w = object_dim
    if axis == "H":
        object_[:h, :w] = object_[:h, :w][:, ::-1]
        object_sel[:h, :w] = object_sel[:h, :w][:, ::-1]
    else:
        object_[:h, :w] = object_[:h, :w][::-1]
        object_sel[:h, :w] = object_sel[:h, :w][::-1]

    _apply_patch(grid, grid_dim, object_, object_dim, object_pos, background)
    _apply_sel(selected, grid_dim, object_sel, object_dim, object_pos)

    return grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity

cdef copy(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] inp, 
    tuple[int, int] inp_dim, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid, 
    tuple[int, int] grid_dim, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] clip, 
    tuple[int, int] clip_dim,
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection,
    char source):
    assert source == "I" or source == "O"
    
    xmin, xmax, ymin, ymax = _get_bbox(selection)
    if xmax == INT_MIN:
        return clip_dim

    H = xmax - xmin + 1
    W = ymax - ymin + 1

    if source == "I":
        if xmax >= inp_dim[0] or ymax > inp_dim[1]:
            return clip_dim
        clip[:] = 0
        np.copyto(clip[:H, :W], inp[xmin:xmin + H, ymin:ymin + W], where=np.logical_and(inp[xmin:xmin + H, ymin:ymin + W] > 0, selection[xmin:xmin + H, ymin:ymin + W]))
    elif source == "O":
        if xmax >= grid_dim[0] or ymax > grid_dim[1]:
            return clip_dim
        clip[:] = 0
        np.copyto(
            clip[:H, :W], grid[xmin:xmin + H, ymin:ymin + W],
            where=np.logical_and(
                grid[xmin:xmin + H, ymin:ymin + W] > 0,
                selection[xmin:xmin + H, ymin:ymin + W]))
    
    return H, W

cdef void paste(
    np.ndarray[np.uint8_t, ndim=2, mode='c'] grid, 
    np.ndarray[np.uint8_t, ndim=2, mode='c'] clip, 
    tuple[int, int] clip_dim, 
    np.ndarray[np.npy_bool, ndim=2, mode='c', cast=True] selection):
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
