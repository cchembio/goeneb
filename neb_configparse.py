import logging

import neb_exceptions as nex
from file_sys_io import qread

logger = logging.getLogger(__name__)

class Settings:
    def __init__(self, neb_ini: str = None):
        # First all defaults
        # structures and files
        self.starttraj = None
        self.TS_guess = None
        self.start_structure = None
        self.end_structure = None
        self.tempdir = None

        # general options
        self.maxiter = 500
        self.climbing_image = False
        self.frozen_atom_indices = None
        self.relaxed_neb = False
        self.trajtest = False
        self.n_images = 11

        # springs and tangents
        self.use_vark = False
        self.k_const = 0.003
        self.tangents = 'henkjon'
        self.spring_gradient = 'difference'
        self.vark_min_fac = 0.1
        self.use_analytical_springpos = False

        # step prediction
        self.step_pred_method = 'AMGD'
        self.stepsize_fac = 0.2
        self.harmonic_stepsize_fac = 0.01
        self.harmonic_conv_fac = 0.7
        self.AMGD_max_gamma = 0.9
        self.NR_start = 10
        self.BFGS_start = 5
        self.initial_hessian = 'diagonal'
        self.max_step = 0.05

        # rotations and translations
        self.rot_align_mode = 'pairwise'
        self.remove_gradtrans = True
        self.remove_gradrot = False

        # IDPP and interpolation
        self.interp_mode = None
        self.IDPP = None
        self.SIDPP = None
        self.IDPP_maxiter = None
        self.IDPP_max_RMSF = None
        self.IDPP_max_AbsF = None

        # convergence
        self.Max_RMSF_tol = 0.000945
        self.Max_AbsF_tol = 0.00189
        self.CI_RMSF_tol = 0.000473
        self.CI_AbsF_tol = 0.000945
        self.Relaxed_Max_RMSF_tol = 0.00945
        self.Relaxed_Max_AbsF_tol = 0.0189
        self.failed_img_tol_percent = 1.0

        # interfaces
        self.interface = None
        self.molpro_path = None
        self.gaussian_path = None
        self.orca_path = None
        self.orca_keywords = None
        self.orca_keywords2 = None
        self.gaussian_keywords = None
        self.molpro_keywords = None

        # cluster
        self.n_threads = 1
        self.memory = 10000
        self.verbose = 'info'

        # system
        self.charge = 0
        self.spin = 1

        # for case-insensitive checking of user settings
        self.allowed_keywords = set(self.__dict__.keys())
        self.allowed_keywords_map = {k.lower(): k for k in self.allowed_keywords}

        if neb_ini is not None:
            user_settings = self.parse_inpfile(neb_ini)
            self.check_values(user_settings)
            self.set_values(user_settings)

    def parse_inpfile(self, filepath:str):
        """
        Read the input file, remove any blank spaces, convert the 
        entrys to their respective types and return a dictionary.
        """
        user_settings = {}
        lines = qread(filepath)
        for line in lines:
            # ignore lines with no '=' in them
            if '=' not in line:
                continue
            try:
                clean_line = self.clean_eqsign(line)
                [newkey, newval] = clean_line.split('=', 1)
                user_settings[newkey] = self.convert_entry(newval)
            except (ValueError, IndexError):
                raise nex.NEBError("Error in neb_configparse: couldn't parse" +
                                   "input file line: " + str(line))
        return user_settings
    
    def check_values(self, user_settings:dict):
        for key in user_settings:
            if key.lower() not in self.allowed_keywords_map:
                raise ValueError(f"Invalid keyword: {key}")
            
    def set_values(self, user_settings:dict):
        for key, value in user_settings.items():
            correct_key = self.allowed_keywords_map[key.lower()]
            setattr(self, correct_key, value)

    @staticmethod
    def clean_eqsign(text):
        """Remove blank spaces in the beginning, end and around the '='."""
        if '=' not in text:
            logger.warning("No '=' found in the input text.")
            return ""
        eqsign = text.index('=')
        left = text[:eqsign].strip()
        right = text[eqsign+1:].strip()
        return left + '=' + right

    @staticmethod
    def convert_entry(entry):
        """
        function for checking if a config entry
        is a special kind of type (e.g. None, bool, int,...)
        and convert them if they are.
        """
        entry = entry.strip()
        if entry.lower() == 'none':
            return None
        elif entry.lower() == 'true':
            return True
        elif entry.lower() == 'false':
            return False
        else:
            # check for integer
            try:
                parsed_entry = int(entry)
            except ValueError:
                pass
            else:
                return parsed_entry
            # check for float
            try:
                parsed_entry = float(entry)
            except ValueError:
                pass
            else:
                return parsed_entry
            # it is a string
            return entry

