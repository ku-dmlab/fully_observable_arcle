
from foarcle.actions.color cimport color, floodfill
from foarcle.actions.object cimport move, rotate, flip, copy, paste
from foarcle.actions.critical cimport crop_grid
import numpy as np
cimport numpy as np
cimport cython


cpdef act(
    np.ndarray[np.uint8_t, ndim=2] inp,
    tuple[int, int] inp_dim,
    np.ndarray[np.uint8_t, ndim=2] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selected,
    np.ndarray[np.uint8_t, ndim=2] clip,
    tuple[int, int] clip_dim, 
    char terminated, 
    char objsel_active, 
    np.ndarray[np.uint8_t, ndim=2] objsel, 
    np.ndarray[np.uint8_t, ndim=2] objsel_area, 
    np.ndarray[np.uint8_t, ndim=2] objsel_bg, 
    tuple[int, int] objsel_coord, 
    char objsel_rot,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection,
    char operation):

    if operation >= 0 and operation <= 9:
        color(grid, selection, operation)
        selected[:] = 0
        objsel_active = False
    elif operation <= 19:
        floodfill(grid, grid_dim, selection, operation - 10)
        selected[:] = 0
        objsel_active = False
    elif operation <= 23:
        grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = move(
            grid, grid_dim, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot, selection, operation - 20
        )
    elif operation == 24:
        grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = rotate(
            grid, grid_dim, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot, selection, 1
        )
    elif operation == 25:
        grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = rotate(
            grid, grid_dim, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot, selection, 3
        )
    elif operation == 26:
        grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = flip(
            grid, grid_dim, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot, selection, "H"
        )
    elif operation == 27:
        grid, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot = flip(
            grid, grid_dim, selected, objsel, objsel_area, objsel_bg, objsel_coord, objsel_active, objsel_rot, selection, "V"
        )
    elif operation == 28:
        clip_dim = copy(inp, inp_dim, grid, clip, clip_dim, selection, "I")
        selected[:] = 0
        objsel_active = False
    elif operation == 29:
        clip_dim = copy(inp, inp_dim, grid, clip, clip_dim, selection, "O")
        selected[:] = 0
        objsel_active = False
    elif operation == 30:
        paste(grid, clip, clip_dim, selection)
        selected[:] = 0
        objsel_active = False
    elif operation == 31:
        np.copyto(grid, inp)
        grid_dim = inp_dim
        selected[:] = 0
        objsel_active = False
    elif operation == 32:
        grid[:] = 0
        selected[:] = 0
        objsel_active = False
    elif operation == 33:
        grid_dim = crop_grid(grid, grid_dim, selection)
        selected[:] = 0
        objsel_active = False
    elif operation == 34:
        terminated = 1
    else:
        assert False

    return grid, grid_dim, selected, clip, clip_dim, terminated, objsel_active, objsel, objsel_area, objsel_bg, objsel_coord, objsel_rot

cpdef batch_act(
    b_inp, b_inp_dim,
    b_grid, b_grid_dim, b_selected, b_clip, b_clip_dim, b_terminated, b_objsel_active, b_objsel, b_objsel_area, b_objsel_bg, b_objsel_coord, b_objsel_rot,
    b_selection, b_operation):

    nb_grid = b_grid.copy()
    nb_grid_dim = b_grid_dim.copy()
    nb_selected = b_selected.copy()
    nb_clip = b_clip.copy()
    nb_clip_dim = b_clip_dim.copy()
    nb_terminated = b_terminated.copy()
    nb_objsel_active = b_objsel_active.copy()
    nb_objsel = b_objsel.copy()
    nb_objsel_area = b_objsel_area.copy()
    nb_objsel_bg = b_objsel_bg.copy()
    nb_objsel_coord = b_objsel_coord.copy()
    nb_objsel_rot = b_objsel_rot.copy()

    for i, (
        inp, inp_dim, grid, grid_dim, selected, clip, clip_dim, terminated, objsel_active, objsel, objsel_area, objsel_bg, objsel_coord, objsel_rot, selection, operation
    ) in enumerate(zip(
        b_inp,
        b_inp_dim,
        b_grid, 
        b_grid_dim, 
        b_selected, 
        b_clip, 
        b_clip_dim, 
        b_terminated, 
        b_objsel_active, 
        b_objsel, 
        b_objsel_area, 
        b_objsel_bg, 
        b_objsel_coord, 
        b_objsel_rot, 
        b_selection, 
        b_operation)):

        (nb_grid[i], 
        nb_grid_dim[i], 
        nb_selected[i], 
        nb_clip[i], 
        nb_clip_dim[i], 
        nb_terminated[i], 
        nb_objsel_active[i], 
        nb_objsel[i], 
        nb_objsel_area[i], 
        nb_objsel_bg[i], 
        nb_objsel_coord[i], 
        nb_objsel_rot[i]) = act(
            inp, 
            inp_dim, 
            grid, 
            grid_dim, 
            selected, 
            clip, 
            clip_dim, 
            terminated, 
            objsel_active, 
            objsel, 
            objsel_area, 
            objsel_bg, 
            objsel_coord, 
            objsel_rot, 
            selection, 
            operation)

    return (
        nb_grid, 
        nb_grid_dim, 
        nb_selected, 
        nb_clip, 
        nb_clip_dim, 
        nb_terminated, 
        nb_objsel_active, 
        nb_objsel, 
        nb_objsel_area, 
        nb_objsel_bg, 
        nb_objsel_coord, 
        nb_objsel_rot
    )
