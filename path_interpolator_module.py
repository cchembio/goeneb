import numpy as np
import logging

import struct_aligner_module as sam
import IDPP_module as idpp
import geodesic_module as geo
from neb_exceptions import NEBError
from helper import interpolate_linear

logger = logging.getLogger(__name__)

# This module contains functions used to generate
# interpolated paths between two end structures.
# Used for starting path generation and reinterpolating
# failed images.


def do_interpolation(start_pvec,
                     end_pvec,
                     labels,
                     interpolations,
                     mode='internal'):
    """
    This function does the basic interpolation,
    with option for cartesian, internal or geodesic.
    Expects:
    - start_pvec and end_pvec: Structures
    - labels: A list with types of atoms
    - mode: internal, cartesian or geodesic
    - interpolations: Either the number of new images, or the specific positions of the interpolation
    """
    # Get the interpolation space
    if isinstance(interpolations, int):
        if mode != 'geodesic':
            interpolations = np.linspace(0.0, 1.0, num=interpolations+2)
    else:
        if mode == 'geodesic':
            mode = 'internal'
            logger.warning('Setting intepolation mode to internal, since geodesic cant handle specific input.')

    if mode == 'cartesian':
        interp_path = interpolate_linear(start_pvec, end_pvec, interpolations)

    elif mode == 'internal':
        # only import now, to keep functionality without ChemChoord
        import ChemCoord_interface as cci

        # construction table based on each end once, then compare
        interp1, ctable1 = cci.zmat_interpolate(start_pvec,
                                                end_pvec,
                                                labels,
                                                interpolations,
                                                use_2nd_constr_table=False)

        interp2, ctable2 = cci.zmat_interpolate(start_pvec,
                                                end_pvec,
                                                labels,
                                                interpolations,
                                                use_2nd_constr_table=True)

        # compare total atom movement to find better path
        tam1 = cci.total_atom_movement(interp1)
        tam2 = cci.total_atom_movement(interp2)
        if tam1 <= tam2:
            interp_path = interp1
        else:
            interp_path = interp2

    elif mode == 'geodesic':
        level = logger.level
        if logger.level < 30:
            logger.setLevel('WARNING')      # geodesic has some output we don't need
        interp_path = geo.interpolate_geodesic(start_pvec, end_pvec, labels, interpolations)
        logger.setLevel(level)              # set the level back after geodesic use
    else:
        raise NEBError('Error in path_interpolator_module: "' + str(mode) +
                       '" is not a valid interpolation mode.')
    return interp_path


def interpolate_path(start_pvec,
                     end_pvec,
                     n_new_interps,
                     labels,
                     interp_mode,
                     rot_align_mode,
                     **IDPP_kwargs):
    """
    This function does the interpolation of the starting path (including alignment and IDPP),
    with option for cartesian, internal or geodesic.
    Expects:
    - start_pvec and end_pvec: Structures
    - n_new_interps: How many new interpolation images should be produced
    - labels: A list with types of atoms
    - interp_mode: internal, cartesian or geodesic
    - rot_align_mode: None, full or pairwise
    - IDPP_kwargs: all IDPP kwargs including: IDPP and SIDPP (True/False), IDPP_maxiter, IDPP_max_RMSF, IDPP_max_AbsF, max_step
    """
    if IDPP_kwargs['SIDPP']:
        path_pvecs = idpp.do_SIDPP_opt_pass(start_pvec,
                                            end_pvec,
                                            n_new_interps,
                                            IDPP_kwargs['IDPP_maxiter'],
                                            IDPP_kwargs['IDPP_max_RMSF'],
                                            IDPP_kwargs['IDPP_max_AbsF'],
                                            IDPP_kwargs['max_step'])
    else:
        # initial interpolations
        path_pvecs = do_interpolation(start_pvec,
                                      end_pvec,
                                      labels,
                                      n_new_interps,
                                      interp_mode)

        # translational and rotational alignement
        path_pvecs = sam.align_path(path_pvecs, rot_align_mode)

        # IDPP
        if IDPP_kwargs['IDPP']:
            if interp_mode == 'geodesic':
                logger.warning('IDPP pass should not be selected, when using geodesic interpolation, '
                              + 'IDPP pass will be skipped in favor of geodesic interpolation.')
            else:
                path_pvecs = idpp.do_IDPP_opt_pass(path_pvecs,
                                                   IDPP_kwargs['IDPP_maxiter'],
                                                   IDPP_kwargs['IDPP_max_RMSF'],
                                                   IDPP_kwargs['IDPP_max_AbsF'],
                                                   IDPP_kwargs['max_step'])

    # translational and rotational alignement (again after IDPP)
    path_pvecs = sam.align_path(path_pvecs, rot_align_mode)
    return path_pvecs

def interpolate_TS(coords, energies, labels, mode='internal'):
    """
    Interpolate a TS structure starting from the three highest energy images. 
    Assumes that we are already in the quadratic region of the PES.
    Expects:
    - coords of the whole path
    - energies of the whole path
    - labels of atomtypes
    - interpolation mode
    """
    energies = np.array(energies)
    hei_index = np.nanargmax(energies)
    left_coords = coords[hei_index-1]
    hei_coords = coords[hei_index]
    right_coords = coords[hei_index+1]
    left_e = energies[hei_index-1]
    hei_e = energies[hei_index]
    right_e = energies[hei_index+1]

    ld = np.linalg.norm(left_coords-hei_coords)
    logger.debug(f'Distance of HEI to left neighbor: {ld} A.')
    rd = np.linalg.norm(right_coords-hei_coords)
    logger.debug(f'Distance of HEI to right neighbor: {rd} A.')

    a = (1/(ld**2 + ld*rd)) * (left_e - hei_e + (ld/rd)*(right_e - hei_e))
    b = 1/rd * (right_e - hei_e - a*rd**2)
    x = -b/(2*a)
    logger.debug(f'Quadratic equation: {a} x^2 + {b} x + {hei_e}')
    logger.debug(f'TS guess at x = {x}')

    if x < 0:
        x = np.abs(x)
        logger.debug(f'Doing interpolation to left neighbor with fraction: {x/ld}')
        ts_coords = do_interpolation(hei_coords, left_coords, labels, [x/ld], mode)[0]
    elif x > 0:
        logger.debug(f'Doing interpolation to right neighbor with fraction: {x/rd}')
        ts_coords = do_interpolation(hei_coords, right_coords, labels, [x/rd], mode)[0]
    else:
        ts_coords = hei_coords
    return ts_coords
