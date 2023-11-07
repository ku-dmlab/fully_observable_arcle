
from foarcle.actions.color cimport color, floodfill
from foarcle.actions.object cimport move, rotate, flip, copy, paste
from foarcle.actions.critical cimport crop_grid
import numpy as np
cimport numpy as np
cimport cython


cpdef act(
    np.ndarray[np.uint8_t, ndim=2] inp,
    tuple[int, int] inp_dim,
    np.ndarray[np.uint8_t, ndim=2] answer,
    np.ndarray[np.uint8_t, ndim=2] grid,
    tuple[int, int] grid_dim,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selected,
    np.ndarray[np.uint8_t, ndim=2] clip,
    tuple[int, int] clip_dim, 
    char terminated, 
    char trials_remain, 
    char active, 
    np.ndarray[np.uint8_t, ndim=2] object_, 
    np.ndarray[np.uint8_t, ndim=2] object_sel, 
    tuple[int, int] object_dim, 
    tuple[int, int] object_pos, 
    np.ndarray[np.uint8_t, ndim=2] background,
    char rotation_parity,
    np.ndarray[np.npy_bool, ndim=2, cast=True] selection,
    char operation):

    cdef int reward = 0

    if operation >= 0 and operation <= 9:
        color(grid, selection, operation)
        selected[:] = 0
        active = False
    elif operation <= 19:
        floodfill(grid, grid_dim, selection, operation - 10)
        selected[:] = 0
        active = False
    elif operation <= 23:
        grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity = move(
            grid, grid_dim, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity, selection, operation - 20
        )
    elif operation == 24:
        grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity = rotate(
            grid, grid_dim, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity, selection, 1
        )
    elif operation == 25:
        grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity = rotate(
            grid, grid_dim, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity, selection, 3
        )
    elif operation == 26:
        grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity = flip(
            grid, grid_dim, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity, selection, "H"
        )
    elif operation == 27:
        grid, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity = flip(
            grid, grid_dim, selected, active, object_, object_sel, object_dim, object_pos, background, rotation_parity, selection, "V"
        )
    elif operation == 28:
        clip_dim = copy(inp, inp_dim, grid, grid_dim, clip, clip_dim, selection, "I")
        selected[:] = 0
        active = False
    elif operation == 29:
        clip_dim = copy(inp, inp_dim, grid, grid_dim, clip, clip_dim, selection, "O")
        selected[:] = 0
        active = False
    elif operation == 30:
        paste(grid, clip, clip_dim, selection)
        selected[:] = 0
        active = False
    elif operation == 31:
        np.copyto(grid, inp)
        grid_dim = inp_dim
        selected[:] = 0
        active = False
    elif operation == 32:
        grid[:] = 0
        selected[:] = 0
        active = False
    elif operation == 33:
        grid_dim = crop_grid(grid, grid_dim, selection)
        selected[:] = 0
        active = False
    elif operation == 34:
        if trials_remain > 0:
            trials_remain -= 1
        if grid_dim[0] == answer.shape[0] and grid_dim[1] == answer.shape[1] and np.all(
            grid[:grid_dim[0], :grid_dim[1]] == answer):
            if trials_remain > 0:
                terminated = 1
            reward = 1
        if trials_remain == 0:
            terminated = 1
    else:
        assert False

    return (grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain, 
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity, reward)

cpdef batch_act(
    b_inp, b_inp_dim, b_answer,
    b_grid, b_grid_dim, b_selected, b_clip, b_clip_dim, b_terminated, b_trials_remain,
    b_active, b_object_, b_object_sel, b_object_dim, b_object_pos, b_background, b_rotation_parity,
    b_selection, b_operation):

    nb_grid = b_grid.copy()
    nb_grid_dim = b_grid_dim.copy()
    nb_selected = b_selected.copy()
    nb_clip = b_clip.copy()
    nb_clip_dim = b_clip_dim.copy()
    nb_terminated = b_terminated.copy()
    nb_trials_remain = b_trials_remain.copy()
    nb_active = b_active.copy()
    nb_object_ = b_object_.copy()
    nb_object_sel = b_object_sel.copy()
    nb_object_dim = b_object_dim.copy()
    nb_object_pos = b_object_pos.copy()
    nb_background = b_background.copy()
    nb_rotation_parity = b_rotation_parity.copy()

    for i, (
        inp, inp_dim, answer, grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain,
        active, object_, object_sel, object_dim, object_pos, background, rotation_parity,
        selection, operation
    ) in enumerate(zip(
        b_inp, b_inp_dim, b_answer, b_grid, b_grid_dim, b_selected, b_clip, b_clip_dim, b_terminated, b_trials_remain,
        b_active, b_object_, b_object_sel, b_object_dim, b_object_pos, b_background, b_rotation_parity, 
        b_selection, b_operation)):

        (nb_grid[i], nb_grid_dim[i], nb_selected[i], nb_clip[i], nb_clip_dim[i], nb_terminated[i], nb_trials_remain[i],
        nb_active[i], nb_object_[i], nb_object_sel[i], nb_object_dim[i], nb_object_pos[i],
        nb_background[i], nb_rotation_parity[i]) = act(
            inp, inp_dim, answer, grid, grid_dim, selected, clip, clip_dim, terminated, trials_remain,
            active, object_, object_sel, object_dim, object_pos, background, rotation_parity, 
            selection, operation)

    return (
        nb_grid, nb_grid_dim, nb_selected, nb_clip, nb_clip_dim, nb_terminated, nb_trials_remain,
        nb_active, nb_object_, nb_object_sel, nb_object_dim, nb_object_pos, 
        nb_background, nb_rotation_parity
    )
