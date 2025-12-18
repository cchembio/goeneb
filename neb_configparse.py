import logging

import neb_exceptions as nex
from file_sys_io import qread

logger = logging.getLogger(__name__)


factory_settings = {'starttraj' : None,
                    'TS_guess' : None,
                    'start_structure' : None,
                    'end_structure' : None,
                    'interface' : None,
                    'maxiter' : 500,
                    'climbing_image' : False,
                    'frozen_atom_indices' : None,
                    'relaxed_neb' : False,
                    'tempdir' : None,
                    'n_images' : 11,
                    'interp_mode' : 'internal',
                    'rot_align_mode' : 'pairwise',
                    'IDPP' : True,
                    'SIDPP' : False,
                    'IDPP_maxiter' : 1000,
                    'IDPP_max_RMSF' : 0.00945,
                    'IDPP_max_AbsF' : 0.0189,
                    'trajtest' : False,
                    'use_vark' : False,
                    'k_const' : 0.003,
                    'tangents' : 'henkjon',
                    'spring_gradient' : 'difference',
                    'vark_min_fac' : 0.1,
                    'step_pred_method' : 'AMGD',
                    'stepsize_fac' : 0.2,
                    'harmonic_stepsize_fac' : 0.01,
                    'harmonic_conv_fac' : 0.7,
                    'AMGD_max_gamma' : 0.9,
                    'NR_start' : 10,
                    'BFGS_start' : 5,
                    'initial_hessian' : 'diagonal',
                    'max_step' : 0.05,
                    'remove_gradtrans' : True,
                    'remove_gradrot' : False,
                    'use_analytical_springpos' : False,
                    'Max_RMSF_tol' : 0.000945, # neb tolerances like orca 6
                    'Max_AbsF_tol' : 0.00189,
                    'CI_RMSF_tol' : 0.000473,
                    'CI_AbsF_tol' : 0.000945,
                    'Relaxed_Max_RMSF_tol' : 0.00945,
                    'Relaxed_Max_AbsF_tol' : 0.0189,
                    'failed_img_tol_percent' : 1.0,
                    'molpro_path' : None,
                    'gaussian_path' : None,
                    'orca_path' : None,
                    'orca_keywords' : None,
                    'orca_keywords2' : None,
                    'gaussian_keywords' : None,
                    'molpro_keywords' : None,
                    'n_threads' : 1,
                    'charge' : 0,
                    'spin' : 1,
                    'memory' : 10000,
                    'verbose' : 'info'}


  
def clean_eqsign(text):
    """Remove blank spaces in the beginning, end and around the '='."""
    if '=' not in text:
        logger.warning("No '=' found in the input text.")
        return ""
    eqsign = text.index('=')
    left = text[:eqsign].strip()
    right = text[eqsign+1:].strip()
    return left + '=' + right


def convert_entry(entry):
    """function for checking if a config entry
    is a special kind of type (e.g. None, bool, int,...)
    and convert them if they are.
    """
    if entry == 'None':
        return None
    elif entry == 'True':
        return True
    elif entry == 'False':
        return False
    else:
        if '.' not in entry:
            # see if it might be an int
            try:
                result = int(entry)
            except:
                pass
            else:
                return result
        # otherwise, see if it might be a float
        try:
            result = float(entry)
        except:
            pass
        else:
            return result
        # apparently, it really is a string, return as-is
        return entry


def parse_inpfile(filepath, input_dict):
    """Read the input file, remove any blank spaces, convert the 
    entrys to their respective types and add to the dictionary.
    """
    lines = qread(filepath)
    for line in lines:
        # ignore lines with no '=' in them
        if '=' not in line:
            continue
        try:
            clean_line = clean_eqsign(line)
            [newkey, newval] = clean_line.split('=', 1)
            input_dict[newkey] = convert_entry(newval)
        except (ValueError, IndexError):
            raise nex.NEBError("Error in neb_configparse: couldn't parse" +
                               "input file line: " + str(line))
    return input_dict


def load_inputfile(filepath):
    """Read the input file and change all the default values to the specified 
    values. returns a dictionary with the current settings."""
    global factory_settings
    current_settings = parse_inpfile(filepath,
                                     factory_settings.copy())
    return current_settings
