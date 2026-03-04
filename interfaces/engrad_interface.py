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
from . import tblite_interface as tbl
from . import dummy_interface as dmm

logger = logging.getLogger(__name__)

def gather_engrfunc(atomic_labels, settings, temp_workdir):
    """Set up the appropriate energy function from the selected 
    interface (orca, gaussian, molpro or dummy interface).\\
    Returns:
    - energy function
    - energy keywords to be used with the function
    """
    interface = settings.interface


    if interface is None:
        # user forgot to set this
        raise nex.NEBError('Error: no interface for engrad calculation ' +\
                           'selected. Make sure to select an interface ' +\
                           'option in the input file.')

    elif interface == 'orca':
        engrfunc, engrfunc_kwargs = setup_orca(atomic_labels,
                                               settings,
                                               temp_workdir)

    elif interface == 'gaussian':
        engrfunc, engrfunc_kwargs = setup_gaussian(atomic_labels,
                                                   settings,
                                                   temp_workdir)

    elif interface == 'molpro':
        engrfunc, engrfunc_kwargs = setup_molpro(atomic_labels,
                                                 settings,
                                                 temp_workdir)
        
    elif interface == 'tblite':
        engrfunc, engrfunc_kwargs = setup_tblite(atomic_labels,
                                                 settings)

    elif interface == 'dummy':
        number_of_images = settings.n_images + 2
        engrfunc, engrfunc_kwargs = setup_dummy(number_of_images)

    else:
        raise nex.NEBError('Error: invalid interface selected in input' +\
                           ' file: ' + str(interface))

    return engrfunc, engrfunc_kwargs


# helper functions for setting up ORCA interface
def setup_orca(atomic_labels, settings, temp_workdir):
    """Set up the orca energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Orca interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))

    engrfunc = orc.orca_path_engrads
    orca_keywords = settings.orca_keywords

    if orca_keywords is None:
        raise nex.NEBError('Error: ORCA interface selected, but no ' +\
                           'ORCA calculation keywords specified. Make ' +\
                           'sure to specify them in the .ini file under ' +\
                           'the "orca_keywords" keyword.')

    if settings.orca_keywords2 is None:
        ml2 = None
    else:
        ml2 = codecs.decode(settings.orca_keywords2, 'unicode_escape') 

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'npal' : settings.n_threads,
                       'workingdir' : temp_workdir,
                       'orca' : find_orcapath(settings),
                       'method_keywords' : orca_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'series_name' : 'image',
                       'method_line2' : ml2}

    return engrfunc, engrfunc_kwargs


def find_orcapath(settings):
    """Get the ORCA Path. Either from the ini file or from the environment variable ORCA_EXE."""
    logger = logging.getLogger(__name__)
    if settings.orca_path is not None:
        logger.info('Using orca path from .ini file.')
        orcapath = settings.orca_path

    else:
        orcapath = os.getenv('ORCA_EXE')

        if orcapath is None:
            raise nex.NEBError('ORCA_EXE environment variable not set. ' +\
                               'Set this variable before starting this ' +\
                               'program, or specify the "orca_path" ' +\
                               'keyword in the ini file.')

    return orcapath


# helper functions for setting up the Gaussian interface
def setup_gaussian(atomic_labels, settings, temp_workdir):
    """Set up the gaussian energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Gaussian interface for EnGrad calculation selected.')
    logger.info('Temporary files stored at: \n' + str(temp_workdir))
    logger.info('Memory allocated: ' + str(settings.memory))

    engrfunc = gss.gaussian_path_engrads
    gauss_keywords = settings.gaussian_keywords

    if gauss_keywords is None:
        raise nex.NEBError('Error: gaussian interface selected, but no ' +\
                           'gaussian calculation keywords specified. Make' +\
                           ' sure to specify them in the .ini file under ' +\
                           'the "gaussian_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'workingdir' : temp_workdir,
                       'series_name' : 'image',
                       'gaussian' : find_gausspath(settings),
                       'method_keywords' : gauss_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'nprocs' : settings.n_threads,
                       'mem' : settings.memory,
                       'title' : 'Title Card Required'}

    return engrfunc, engrfunc_kwargs


def find_gausspath(settings):
    """Get the gaussian Path. Either from the ini file or from the environment variable GAUSS_EXE."""
    logger = logging.getLogger(__name__)
    if settings.gaussian_path is not None:
        logger.info('Using gaussian path from .ini file.')
        gaussianpath = settings.gaussian_path

    else:
        gaussianpath = os.getenv('GAUSS_EXE')

        if gaussianpath is None:
            raise nex.NEBError('GAUSS_EXE environment variable not set. ' +\
                               'Set this variable before starting this ' +\
                               'program, or specify the "gaussian_path" ' +\
                               'keyword in the ini file.')

    return gaussianpath


# helper functions for setting up MolPro interface 

def setup_molpro(atomic_labels, settings, temp_workdir):
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
    logger.info('Memory allocated: ' + str(settings.memory))

    engrfunc = mol.molpro_path_engrads
    molpro_keywords = settings.molpro_keywords

    if molpro_keywords is None:
        raise nex.NEBError('Error: molpro interface selected, but no ' +\
                           'Molpro calculation keywords specified. Make ' +\
                           'sure to specify them in the .ini file under ' +\
                           'the "molpro_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'workingdir' : temp_workdir,
                       'series_name' : 'image',
                       'molpro' : find_molpro_path(settings),
                       'method_keywords' : molpro_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'nprocs' : settings.n_threads,
                       'mem' : settings.memory}
                       #'errordir' : find_workdir()}

    return engrfunc, engrfunc_kwargs


def find_molpro_path(settings):
    """Get the Molpro Path. Either from the ini file or from the environment variable MOL_EXE."""
    logger = logging.getLogger(__name__)
    if settings.molpro_path is not None:
        logger.info('Using molpro path from .ini file.')
        molpath = Path(settings.molpro_path)

    else:
        molpath = os.getenv('MOL_EXE')
        if molpath is None:
            raise nex.NEBError('MOL_EXE environment variable not set. Set ' +\
                               'this variable before starting this program' +\
                               ', or specify the "molpro_path" keyword in ' +\
                               'the ini file.')

        molpath = Path(molpath)
    return molpath


# helper functions for setting up tblite interface
def setup_tblite(atomic_labels, settings):
    """Set up the tblite energy and gradient calculation function with the specified user input.\\
    Returns: 
    - energy function
    - energy keywords to be used with the function
    """
    logger = logging.getLogger(__name__)
    logger.info('Tblite interface for EnGrad calculation selected.')

    npal = settings.n_threads
    os.environ["OMP_NUM_THREADS"] = str(npal)

    engrfunc = tbl.tblite_path_engrads
    tblite_keywords = settings.tblite_keywords

    if tblite_keywords is None:
        raise nex.NEBError('Error: tblite interface selected, but no ' +\
                           'tblite calculation keywords specified. Make ' +\
                           'sure to specify them in the .ini file under ' +\
                           'the "tblite_keywords" keyword.')

    engrfunc_kwargs = {'labels' : atomic_labels,
                       'method_keywords' : tblite_keywords,
                       'charge' : settings.charge,
                       'spin' : settings.spin,
                       'series_name' : 'image'}

    return engrfunc, engrfunc_kwargs


# helper function for setting up the dummy interface
def setup_dummy(number_of_images):
    engrfunc = dmm.dummy_path_engrads
    engrfunc_kwargs = {'iteration' : dmm.IterationState(), 
                       'number_of_images' : number_of_images}
    return engrfunc, engrfunc_kwargs