# the following section contains the helper 
# function to gather the engrad function and its arguments,
# depending on which interface was chosen by the user.
# --------------------------------------------------------

import os
import logging
from pathlib import Path
import codecs

import neb_exceptions as nex
from . import orca_interface as orc
from . import gaussian_interface as gss
from . import molpro_interface as mol
from . import dummy_interface as dmm

logger = logging.getLogger(__name__)

def gather_engrfunc(atomic_labels, conf_dict, temp_workdir):
    """Set up the appropriate energy function from the selected 
    interface (orca, gaussian, molpro or dummy interface).\\
    Returns:
    - energy function
    - energy keywords to be used with the function
    """
    interface = conf_dict['interface']


    if interface is None:
        # user forgot to set this
        raise nex.NEBError('Error: no interface for engrad calculation ' +\
                           'selected. Make sure to select an interface ' +\
                           'option in the input file.')

    elif interface == 'orca':
        engrfunc, engrfunc_kwargs = setup_orca(atomic_labels,
                                               conf_dict,
                                               temp_workdir)

    elif interface == 'gaussian':
        engrfunc, engrfunc_kwargs = setup_gaussian(atomic_labels,
                                                   conf_dict,
                                                   temp_workdir)

    elif interface == 'molpro':
        engrfunc, engrfunc_kwargs = setup_molpro(atomic_labels,
                                                 conf_dict,
                                                 temp_workdir)

    elif interface == 'dummy':
        number_of_images = conf_dict['n_images'] + 2
        engrfunc, engrfunc_kwargs = setup_dummy(number_of_images)

    else:
        raise nex.NEBError('Error: invalid interface selected in input' +\
                           ' file: ' + str(interface))

    return engrfunc, engrfunc_kwargs


# helper functions for setting up ORCA interface
def setup_orca(atomic_labels, conf_dict, temp_workdir):
    """Set up the orca energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Orca interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))

    engrfunc = orc.orca_path_engrads
    orca_keywords = conf_dict['orca_keywords']

    if orca_keywords is None:
        raise nex.NEBError('Error: ORCA interface selected, but no ' +\
                           'ORCA calculation keywords specified. Make ' +\
                           'sure to specify them in the .ini file under ' +\
                           'the "orca_keywords" keyword.')

    if conf_dict['orca_keywords2'] is None:
        ml2 = None
    else:
        ml2 = codecs.decode(conf_dict['orca_keywords2'], 'unicode_escape') 

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'npal' : conf_dict['n_threads'],
                       'workingdir' : temp_workdir,
                       'orca' : find_orcapath(conf_dict),
                       'method_keywords' : orca_keywords,
                       'charge' : conf_dict['charge'],
                       'spin' : conf_dict['spin'],
                       'series_name' : 'image',
                       'method_line2' : ml2}

    return engrfunc, engrfunc_kwargs


def find_orcapath(conf_dict):
    """Get the ORCA Path. Either from the ini file or from the environment variable ORCA_EXE."""
    logger = logging.getLogger(__name__)
    if conf_dict['orca_path'] is not None:
        logger.info('Using orca path from .ini file.')
        orcapath = conf_dict['orca_path']

    else:
        orcapath = os.getenv('ORCA_EXE')

        if orcapath is None:
            raise nex.NEBError('ORCA_EXE environment variable not set. ' +\
                               'Set this variable before starting this ' +\
                               'program, or specify the "orca_path" ' +\
                               'keyword in the ini file.')

    return orcapath


# helper functions for setting up the Gaussian interface
def setup_gaussian(atomic_labels, conf_dict, temp_workdir):
    """Set up the gaussian energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Gaussian interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))
    logger.info('Memory allocated: ' + str(conf_dict['memory']))

    engrfunc = gss.gaussian_path_engrads
    gauss_keywords = conf_dict['gaussian_keywords']

    if gauss_keywords is None:
        raise nex.NEBError('Error: gaussian interface selected, but no ' +\
                           'gaussian calculation keywords specified. Make' +\
                           ' sure to specify them in the .ini file under ' +\
                           'the "gaussian_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'workingdir' : temp_workdir,
                       'series_name' : 'image',
                       'gaussian' : find_gausspath(conf_dict),
                       'method_keywords' : gauss_keywords,
                       'charge' : conf_dict['charge'],
                       'spin' : conf_dict['spin'],
                       'nprocs' : conf_dict['n_threads'],
                       'mem' : conf_dict['memory'],
                       'title' : 'Title Card Required'}

    return engrfunc, engrfunc_kwargs


def find_gausspath(conf_dict):
    """Get the gaussian Path. Either from the ini file or from the environment variable GAUSS_EXE."""
    logger = logging.getLogger(__name__)
    if conf_dict['gaussian_path'] is not None:
        logger.info('Using gaussian path from .ini file.')
        gaussianpath = conf_dict['gaussian_path']

    else:
        gaussianpath = os.getenv('GAUSS_EXE')

        if gaussianpath is None:
            raise nex.NEBError('GAUSS_EXE environment variable not set. ' +\
                               'Set this variable before starting this ' +\
                               'program, or specify the "gaussian_path" ' +\
                               'keyword in the ini file.')

    return gaussianpath


# helper functions for setting up MolPro interface 

def setup_molpro(atomic_labels, conf_dict, temp_workdir):
    """Set up the Molpro energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    os.environ['TMPDIR'] = str(temp_workdir)
    os.environ['TMPDIR4'] = str(temp_workdir)

    logger.info('Molpro interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))
    logger.info('Memory allocated: ' + str(conf_dict['memory']))

    engrfunc = mol.molpro_path_engrads
    molpro_keywords = conf_dict['molpro_keywords']

    if molpro_keywords is None:
        raise nex.NEBError('Error: molpro interface selected, but no ' +\
                           'Molpro calculation keywords specified. Make ' +\
                           'sure to specify them in the .ini file under ' +\
                           'the "molpro_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'workingdir' : temp_workdir,
                       'series_name' : 'image',
                       'molpro' : find_molpro_path(conf_dict),
                       'method_keywords' : molpro_keywords,
                       'charge' : conf_dict['charge'],
                       'spin' : conf_dict['spin'],
                       'nprocs' : conf_dict['n_threads'],
                       'mem' : conf_dict['memory']}
                       #'errordir' : find_workdir()}

    return engrfunc, engrfunc_kwargs


def find_molpro_path(conf_dict):
    """Get the Molpro Path. Either from the ini file or from the environment variable MOL_EXE."""
    logger = logging.getLogger(__name__)
    if conf_dict['molpro_path'] is not None:
        logger.info('Using molpro path from .ini file.')
        molpath = Path(conf_dict['molpro_path'])

    else:
        molpath = os.getenv('MOL_EXE')
        if molpath is None:
            raise nex.NEBError('MOL_EXE environment variable not set. Set ' +\
                               'this variable before starting this program' +\
                               ', or specify the "molpro_path" keyword in ' +\
                               'the ini file.')

        molpath = Path(molpath)
    return molpath


# helper function for setting up the dummy interface
def setup_dummy(number_of_images):
    engrfunc = dmm.dummy_path_engrads
    engrfunc_kwargs = {'iteration' : dmm.IterationState(), 
                       'number_of_images' : number_of_images}
    return engrfunc, engrfunc_kwargs