import os
import subprocess
import numpy as np
import file_sys_io as io
import logging

from helper import bettersplit

logger = logging.getLogger(__name__)


bohr = 0.529177211  # angs per bohr


# The following is the implementation of the MOLPRO interface
# ---------------------------------------------------------


def molpro_path_engrads(image_pvecs,
                        labels,
                        series_name,
                        workingdir,
                        molpro,
                        method_keywords,
                        charge,
                        spin,
                        nprocs,
                        mem,
                        errordir=None):
    """
    Calculate the complete energys and gradients using molpro.
    """
    image_data = [[labels, pvec] for pvec in image_pvecs]
    image_names = [series_name + str(i+1) for i in range(len(image_pvecs))]
    result = [calculate_molpro_engrad(image,
                                    name,
                                    workingdir,
                                    molpro,
                                    method_keywords,
                                    charge,
                                    spin,
                                    nprocs,
                                    mem,
                                    errordir)
            for image, name in zip(image_data, image_names)]                 
    energies = []
    engrads = []
    # reorganize results
    for item in result:
        if item is None:
            # the calculation failed to converge
            energies.append(None)
            engrads.append(np.zeros_like(image_pvecs[0]))
        else:
            [energy, engrad] = item
            energies.append(energy)
            engrads.append(engrad)
    return np.array(energies), np.array(engrads)

def calculate_molpro_engrad(struct_data,
                            inpfile_name,
                            workingdir,
                            molpro,
                            method_keywords,
                            charge='0',
                            spin='1',
                            nprocs=1,
                            mem=1000,
                            errordir=None):
    """
    Calculate the molpro energy and gradient. This function does:
    1. Delete previous outputs
    2. Write input file
    3. Run Molpro
    4. Read Output file
    """
    io.safe_create_dir(workingdir)
    input_filename = workingdir / (inpfile_name + '.com')
    output_filename = workingdir / (inpfile_name + '.out')
    # make sure no output file is left over. All output files for an
    # image will have the same name, and if one stays around, it might
    # be mistaken for the result of a future calculation that actually failed.
    io.safe_delete_file(output_filename, does_not_exist_ok=True)
    logger.info('Now calculating: ' + inpfile_name)
    write_molpro_inpfile(input_filename, struct_data,
                         method_keywords, charge, spin)
    if errordir is not None:
        esdir = errordir / (inpfile_name + '.com')
        write_molpro_inpfile(esdir, struct_data, method_keywords, charge, spin)
    # molpro needs to be called from within the same directory that
    # the output file is in, so we must go to that directory first
    olddir = os.getcwd()
    os.chdir(workingdir)
    cmd = [str(molpro), '--no-xml-output', '--nobackup',
                     '-n', str(nprocs), '-s', input_filename.name,
                     '-I', '.', '-W', '.', '-o', output_filename.name]
    subprocess.run(cmd)
    # now that molpro is finished, we change back to whatever
    # directory we were in before
    os.chdir(olddir)
    try:
        Energy, Gradient = read_molpro_forcefile(output_filename)
    except:
        return None  # engrad calculation didnt converge
    else:
        return [Energy, Gradient]


# Some helper functions


def write_xyz_filestring(atom_labels, coord_vec, comment=''):
    """
    Write an xyz file string for the molpro input.
    """
    coords = coord_vec.reshape(len(atom_labels), 3)
    if comment == '':
        comment = 'coordinates from NEB optimization module'
    comment = '\n\t' + comment + '\n'
    text = '\t' + str(len(atom_labels)) + comment
    for label, coord in zip(atom_labels, coords):
        text += label
        for number in coord:
            text += '\t' + str(repr(number))
        text += '\n'
    return text


def write_molpro_inpfile(filename, struct_data, keywords,
                         charge, spin, title='Title', mem=1000):
    """
    Write a molpro input file.
    """
    text = '***,' + title 
    text += '\nmemory,' + str(mem) + ',m\n'
    text += '\ngeometry={\n'
    text += write_xyz_filestring(struct_data[0], struct_data[1])
    text += '}\n'
    text += 'SET,CHARGE=' + str(charge) + '\n'
    text += 'SET,SPIN=' + str(spin - 1) + '\n'
    keytokens = bettersplit(keywords, ' \n\t')
    for token in keytokens:
        text += token + '\n'
    text += 'force'
    io.qwrite(filename, text)


def read_molpro_forcefile(resultfile):
    """
    Read the Molpro force file and return the energy and the gradient.
    """
    lines = io.qread(resultfile)
    if lines[-1] != ' Molpro calculation terminated\n':
        raise ValueError('Invalid molpro engrad file.')
    # find energy
    Energy = None
    for line in reversed(lines):
        if 'energy=' in line:
            tokens = bettersplit(line, ' \n\t')
            Energy = float(tokens[-1])
            break
    if Energy is None:
        raise ValueError('Invalid molpro engrad file.')

    # find forces
    startind = lines.index(' Atom          dE/dx               dE/dy'
                           + '               dE/dz\n') + 2
    stopind = lines.index('\n', startind)
    force_vals = []
    for i in range(startind, stopind):
        tokens = bettersplit(lines[i], ' \n\t')
        if len(tokens) != 4:
            raise ValueError('Invalid molpro engrad file.')
        for token in tokens[1:]:
            force_vals.append(float(token))
    force_vec = np.array(force_vals)
    return Energy, force_vec / bohr  # convert Eh/bohr to Eh/ang



