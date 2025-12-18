import os
import shutil
import numpy as np
import logging

from pathlib import Path
from neb_exceptions import NEBError

logger = logging.getLogger(__name__)


def qwrite(filepath, text):
    file = open(filepath, 'w')
    file.write(text)
    file.close()


def qread(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()
    return lines

def qappend(filepath, lines):
    file = open(filepath, 'a')
    file.writelines(lines)
    file.close()


def get_abs_path(rawpath):
    return Path(os.path.abspath(rawpath))


def move_dir(dir_to_move, target_dir):
    shutil.move(str(dir_to_move), str(target_dir))


def check_if_directory(dir_path, check_rwacc=True, print_msgs=False):
    """Check whether dir_path:
    - exists
    - is a directory
    - has reading access
    - has writing access

    The last two are only checked when check_rwacc is True. A message is only printed if print_msgs is True.
    """
    if not os.path.exists(dir_path):
        msg = 'Error in file_sys_io, safe_delete_dir: ' +\
              'could not find directory named ' + str(dir_path)
        if print_msgs:
            logger.error('\n %s \n', msg)
        return False

    if not os.path.isdir(dir_path):
        msg = 'Error in file_sys_io, safe_delete_dir: ' +\
              str(dir_path) + ' is a file'
        if print_msgs:
            logger.error('\n %s \n', msg)
        return False

    if not check_rwacc:
        # passing the first two checks is enough
        return True

    if not os.access(dir_path, os.R_OK):
        msg = 'Error in file_sys_io, safe_delete_dir: ' +\
              'no reading access in ' + str(dir_path)
        if print_msgs:
            logger.error('\n %s \n', msg)
        return False

    if not os.access(dir_path, os.W_OK):
        msg = 'Error in file_sys_io, safe_delete_dir: ' +\
              'no writing access in ' + str(dir_path)
        if print_msgs:
            logger.error('\n %s \n', msg)
        return False

    # the directory has passed all the checks
    return True


def safe_delete_dir(dir_path, does_not_exist_ok=False):
    """Delete a directory and check whether it is gone."""
    if not check_if_directory(dir_path):
        if does_not_exist_ok is True:
            # directory does not exist, we're done.
            return
        else:
            # it was supposed to exist, something is wrong
            raise NEBError('Error in file_sys_io, safe_delete_dir: ' +
                           str(dir_path) + ' is not a valid directory.')

    shutil.rmtree(str(dir_path))

    # dir_path should no longer be an existing directory. let's check
    if check_if_directory(dir_path,
                          check_rwacc=False,
                          print_msgs=False):
        # apparently, it somehow still exists. this is an error
        msg = '\nError in file_sys_io, safe_delete_dir: ' +\
              'unable to delete ' + str(dir_path) + '\n'
        raise NEBError(msg)


def safe_create_dir(new_dir_path, clear_if_existing=False, check_rwacc=True):
    """Create a new directory. Option clear_if_existing deletes the directory if it exists."""
    if clear_if_existing:
        # check if new_dir_path already exists
        if check_if_directory(new_dir_path,
                              check_rwacc=False,
                              print_msgs=False):
            # it does. delete in order to make a new one that is empty
            safe_delete_dir(new_dir_path)

    # (re)create new_dir_path. exist_ok=True in case clear_if_existing not set
    os.makedirs(new_dir_path, exist_ok=True)

    # check if directory now exists as desired
    if not check_if_directory(new_dir_path,
                              check_rwacc=check_rwacc):
        # it does not. this is an error
        msg = '\nError in file_sys_io, safe_create_dir: ' +\
              'unable to create or access' + str(new_dir_path) + '\n'
              
        raise NEBError(msg)


def safe_delete_file(filepath, does_not_exist_ok=False):
    """Delete a file and check whether it is gone."""
    if os.path.isfile(filepath):
        os.remove(filepath)

    elif not does_not_exist_ok:
        # 'filepath' did not exist, and the function wasn't told
        # that it not existing was not ok
        raise NEBError('Error in file_sys_io, safe_delete_file: ' +
                       str(filepath) + ' is not a valid filepath.')

    # confirm delete
    if os.path.isfile(filepath):
        raise NEBError('Error in file_sys_io, safe_delete_file: ' +
                       str(filepath) + ' could not be removed.')


#functions for xyz file and xyz trajectory file IO
#--------------------------------------------------


def conv_xyzlines(lines):
    """Expects a list of strings of the lines in a xyz file.\\
    Returns  the labels and the position vector.
    """
    atom_labels = []
    coords = []

    for line in lines[2:]:
        tokens = line.split()
        if len(tokens) == 0:
            continue
        atom_labels.append(tokens[0])
        for token in tokens[1:]:
            coords.append(float(token))
    return atom_labels, np.array(coords)


def read_xyz_file(filepath):
    """Read an xyz file and return atom labels and position vector."""
    lines = qread(filepath)
    labels, pvec = conv_xyzlines(lines)
    return labels, pvec


def write_xyz_file(atom_labels, coord_vec, filepath, mode='w', energy=None, spacer='    '):
    """Expects the atom_labels as a list and the position vector as a 1D list/array. 
    Writes an xyz file in the specified location with a specified spacer.
    """
    file = open(filepath, mode)
    coords = coord_vec.reshape(len(atom_labels), 3)

    # Comment line includes the energy
    if energy is None:
        comment = spacer + 'Energy:       unknown'
    else:
        comment = spacer + 'Energy:       ' + str(energy)
    comment = '\n' + comment + '\n'
    file.write(str(len(atom_labels)) + comment)

    for label, coord in zip(atom_labels, coords):
        file.write(spacer + label)
        for number in coord:
            file.write(spacer + str(repr(number)))
        file.write('\n')
    file.close()


def write_xyz_traj(labels, coord_vecs, filepath, energies=None):
    """Write a xyz trajectory file. Expects the atom labels as a list and a list of all position vectors in the trajectory."""
    if energies is None:
        energies = [None for vec in coord_vecs]
    #write first structure to file, while also clearing
    #it in case it already existed
    write_xyz_file(labels,
                   coord_vecs[0],
                   filepath,
                   mode='w',
                   energy=energies[0])

    #append the rest of the strucures
    for coord_vec, i_energy in zip(coord_vecs[1:], energies[1:]):
        write_xyz_file(labels,
                       coord_vec,
                       filepath,
                       mode='a',
                       energy=i_energy)


def read_xyz_traj(filepath):
    """Read xyz trajectory file from path."""
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    header_line = lines[0]
    headerl_inds = [0]

    for i in range(1, len(lines)):
        if lines[i] == header_line:
            headerl_inds.append(i)
    structs_lines = []
    for i in range(len(headerl_inds)):
        if i == len(headerl_inds)-1:
            startind = headerl_inds[i]
            structs_lines.append(lines[startind:])
        else:
            startind = headerl_inds[i]
            stopind = headerl_inds[i+1]
            structs_lines.append(lines[startind:stopind])
    atom_labels = None
    posvecs = []

    for struct in structs_lines:
        newlabels, newpvec = conv_xyzlines(struct)
        if atom_labels is None:
            atom_labels = newlabels
        elif not atom_labels == newlabels:
            raise NEBError('Invalid xyz trajectory file: ' + str(filepath))
        if len(newpvec) != len(atom_labels) * 3:
            raise NEBError('Invalid xyz trajectory file: ' + str(filepath))
        posvecs.append(newpvec)

    return atom_labels, posvecs