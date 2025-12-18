import numpy as np
import pandas as pd
import logging

# The chemcoord module imports numba, which produces
# bytecode logging output on the debug level that is not useful
logging.getLogger('numba').setLevel(logging.WARNING)
import chemcoord as cc

from helper import pvec_to_crdmat, crdmat_to_pvec


def total_atom_movement(path_pvecs):
    result = 0.0
    for i in range(1, len(path_pvecs)):
        result += np.linalg.norm(path_pvecs[i] - path_pvecs[i-1])
    return result


def pvec_to_cartesian(labels, pvec):
    coords = pvec_to_crdmat(pvec)

    xvals = [coord[0] for coord in coords]
    yvals = [coord[1] for coord in coords]
    zvals = [coord[2] for coord in coords]

    framedict = {'atom' : labels,
                 'x' : xvals,
                 'y' : yvals,
                 'z' : zvals}

    frame = pd.DataFrame.from_dict(framedict)
    return cc.Cartesian(frame=frame)


def cartesian_to_pvec(cartesian):
    crdmat = np.array(cartesian[['x', 'y', 'z']])
    return crdmat_to_pvec(crdmat)


def pvec_to_zmat(labels, pvec, constr_table=None):
    cc_cart = pvec_to_cartesian(labels, pvec)

    if constr_table is None:
        zmat = cc_cart.get_zmat()
        constr_table = zmat.loc[:, ['b', 'a', 'd']]

    else:
        zmat = cc_cart.get_zmat(constr_table)

    return zmat, constr_table


def zmat_to_pvec(zmat):
    cc_cart = zmat.get_cartesian()

    # sort by original atom index, to undo the atom shuffling
    # that occurred during z matrix construction
    cc_cart.sort_index(inplace=True)
    pvec = cartesian_to_pvec(cc_cart)
    return pvec


def zmat_interpolate(start_pvec,
                     stop_pvec,
                     labels,
                     interpolations,
                     use_2nd_constr_table=False):
    """Do an interpolation in z-matrix coordinates using the ChemChord module.
    Expects:
    - start_pvec: coordinates of start structure
    - end_pvec
    - labels: A list of the atom types
    - interpolations: list of numbers between 0 and 1 for the new interpolations
    - use_2nd_constr_table: True/False to choose the other construction table for a possibly better fit
    """
    if use_2nd_constr_table:
        zmat2, ctable = pvec_to_zmat(labels, stop_pvec)
        zmat1, ctable = pvec_to_zmat(labels, start_pvec, constr_table=ctable)
    else:
        # use 1st construction table instead
        zmat1, ctable = pvec_to_zmat(labels, start_pvec)
        zmat2, ctable = pvec_to_zmat(labels, stop_pvec, constr_table=ctable)

    # perform the actual interpolation
    with cc.TestOperators(False):
        interp_zmats = [zmat1 + (zmat2 - zmat1).minimize_dihedrals() * fac
                        for fac in interpolations]

    # convert to pvecs and return
    interp_pvecs = [zmat_to_pvec(zmat) for zmat in interp_zmats]

    return np.array(interp_pvecs), ctable