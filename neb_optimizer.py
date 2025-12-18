import numpy as np
from datetime import timedelta
import time
import logging

import springforce_module as sfm
import step_pred_module as spm
import nebgrad_module as ngm
import tangent_module as tgm
import path_interpolator_module as pim
import struct_aligner_module as sam
import failed_img_recalculator as fir
import convsig_module as csm
import file_sys_io as io
from neb_exceptions import NEBError
from neb_path import NEBPath
import hessian_module as hm
import neb_exceptions as nex
import logging_module as lgm
from basic_neb import BasicNEB
from helper import parse_index_list

logger = logging.getLogger(__name__)

Hartree_in_kJmol = 2625.49963948

# the following section contains the NEB Optimizer
# --------------------------------------------------------------------

class NEB_Optimizer(BasicNEB):
    def __init__(self, NEBPath:NEBPath, conf_dict, log=True):
        # Parent class
        super().__init__(NEBPath, conf_dict)

        # Begin with relaxed NEB
        self.relaxed = True
        self.ci_user_setting = self.climbing_image
        self.climbing_image = False
        self.max_rmsf = self.Relaxed_Max_RMSF_tol
        self.max_absf = self.Relaxed_Max_AbsF_tol

        # Set ups
        # Step predictor
        self.hessian = None
        self.hessians = [None] * self.images

        if self.step_pred_method == 'AMGD':
            self.predictor = [spm.AMGD(self.stepsize_fac, self.AMGD_max_gamma) for _ in range(self.images)]

        elif self.step_pred_method == 'SD':
            self.predictor = [spm.SD(self.stepsize_fac) for _ in range(self.images)]

        elif self.step_pred_method in ('NR', 'RFO'):
            if self.step_pred_method == 'NR':
                self.predictor = spm.NewtonRaphson(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            elif self.step_pred_method == 'RFO':
                self.predictor = spm.RationalFunction(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            # Also set up global hessian
            self.hessian = hm.hessian(mode=self.initial_hessian, start=self.BFGS_start, labels=self.labels) 

        elif self.step_pred_method in ('SCT', 'L-RFO', 'L-NR'):
            if self.step_pred_method == 'SCT':
                self.predictor = spm.self_consistent_tangents(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            elif self.step_pred_method == 'L-RFO':
                self.predictor = spm.LocalRF(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            elif self.step_pred_method == 'L-NR':
                self.predictor = spm.LocalNR(self.images, self.BFGS_start, self.NR_start, self.stepsize_fac, self.AMGD_max_gamma)
            # Also set up hessians
            self.hessians = [hm.hessian(mode=self.initial_hessian, start = self.BFGS_start, labels=self.labels) for _ in range(self.images)]
        else:
            raise nex.NEBError('Error in with step prediction method. %s is not a valid step predictor mode.',
                               str(conf_dict['step_pred_method']))

        # Logging
        if log:
            self.logger = lgm.NEBLogger(conf_dict['logfile_path'])
        else:
            self.logger = None

    def calc_springgrads(self):
        """
        Callback function used to calculate springforce gradients
        of the images, as well as to recalculate the variable k
        constants if variable k and CI are active.
        """
        if self.use_vark:
            self.recalculate_varks()

        else:
            # if not, set all ks to be the value set in the conf_dict
            self.path.set_img_k_const(self.k_const)

        # calculate regular springforce gradients. even if analytical positions scheme is active,
        # they are still needed for calculating convergence thresholds.
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        img_pair_ks = self.path.get_img_pair_ks()
        tanvecs = self.path.get_tanvecs()

        if self.spring_gradient == 'difference':
            springs = sfm.delta_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [spring * tanvec 
                           for spring, tanvec in zip(springs, tanvecs)]

        elif self.spring_gradient == 'projected':
            full_springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [ngm.project(full_springgrad, tanvec) 
                           for full_springgrad, tanvec in zip(full_springgrads, tanvecs)]

        elif self.spring_gradient == 'raw':
            springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)

        else:
            raise NEBError('Error with springforce definition. %s is not a valid springforce mode.',
                           str(self.spring_gradient))

        # if ci is active, the springforces acting on the ci are zero
        ci_index = self.current_CI_index
        if ci_index is not None:
            springgrads[ci_index][:] = 0.0

        return springgrads

    def recalculate_varks(self):
        """
        Helper function for the spring gradient function.
        Sets the spring constants for all image pairs according to
        the improved variable k scheme.
        """
        full_path_energies = self.path.get_energies(include_ends=True)
        maxk = self.k_const
        mink = self.k_const * self.vark_min_fac

        pairwise_ks = sfm.compute_pairwise_ks(full_path_energies,
                                              maxk,
                                              mink)
        self.path.set_img_pair_ks(pairwise_ks)

    def calc_nebgrads(self):
        """
        Callback function for calculating neb gradients,
        after engrads, springgrads, tanvecs have all been
        calculated and saved in the NEBPath object.
        """
        ci_index = self.current_CI_index
        engrads = self.path.get_engrads()
        img_pvecs = self.path.get_img_pvecs(include_ends=False)

        raw_nebgrads = ngm.calculate_nebgrads(engrads,
                                              self.path.get_springgrads(),
                                              self.path.get_tanvecs(),
                                              ci_index)

        # project out translation and/or rotation from neb gradients
        # (experimental feature), if selected by user
        nebgrads = ngm.sanitize_stepvecs(raw_nebgrads,
                                         img_pvecs,
                                         self.remove_gradtrans,
                                         self.remove_gradrot)

        # zero the gradients corresponding to frozen atoms,
        # so they don't skew the signals of convergence thresholds
        frozen_atom_indices = self.frozen_atom_indices

        if frozen_atom_indices is not None:
            # convert config entry into an actual python list,
            # then apply atom freezing for each step vector
            frozen_index_list = parse_index_list(frozen_atom_indices)

            for i in range(len(nebgrads)):
                nebgrads[i] = ngm.freeze_atom_indices(nebgrads[i], frozen_index_list)

        # calculate the orthogonal gradients
        # project out the spring contribution, which is parallel to the tangent
        tanvecs = self.path.get_tanvecs()
        orth_grads = np.zeros_like(nebgrads)
        for i in range(len(nebgrads)):
            # except if there is a climbing image, which has a special NEB gradient.
            if i != ci_index:
                orth_grads[i] = ngm.reject(nebgrads[i], tanvecs[i])

        return nebgrads, orth_grads

    def calc_tanvecs(self):
        """
        Callback function for computing image tangent vectors.
        we want it to be able to both return normal tangents,
        or apply smoothing for the tangent vectors, depending on
        what the user chose.
        """
        # first, compute unsmoothed tans
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        full_path_energies = self.path.get_energies(include_ends=True)

        # tangent definitions
        if self.tangents == 'henkjon':
            raw_tans = tgm.henkjon_tans(full_path_pvecs,
                                        full_path_energies)
        elif self.tangents == 'simple':
            raw_tans = tgm.simple_tans(full_path_pvecs)
        else:
            raise NEBError('Error with tangent definition. %s is not a valid tangent mode.',
                           str(self.tangents))
        return raw_tans


    # -----------------------------------------------------------------------------------------
    # Step Prediction

    def predict_steps(self):
        """
        Do the step prediction with the saved step predictors
        """
        # first gather a bunch of data from the NEBPath
        full_energies   = self.path.get_energies(include_ends=True) 
        energies        = full_energies[1:-1] 

        nebgrads        = self.path.get_nebgrads()
        engrads         = self.path.get_engrads()
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        img_pvecs       = self.path.get_img_pvecs(include_ends=False)
        ci_index        = self.current_CI_index

        # remove spring contribution from neb gradients
        # if analytical position scheme is active
        if self.use_analytical_springpos:
            nebgrads = self.path.get_orth_grads()

        # Perform the step prediction
        if self.step_pred_method == 'SCT':
            # Update the hessian objects by providing cartesian coordinates
            self.predictor.update(img_pvecs, engrads, energies, self.hessians)
            steps = self.predictor.predict(nebgrads, 
                                           self.hessians,
                                           full_energies,
                                           self.dict,
                                           self.path)

        elif self.step_pred_method in ['AMGD', 'SD']:
            for object, pvec in zip(self.predictor, img_pvecs):
                object.update(pvec)
            steps = [object.predict(nebgrad) 
                     for object, nebgrad in zip(self.predictor, nebgrads)]

        elif self.step_pred_method in ['NR','RFO']:
            # no checking for failed calculations, is that good?
            self.predictor.update(img_pvecs, nebgrads, energies, self.hessian)
            steps = self.predictor.predict(img_pvecs, nebgrads, energies, self.hessian)

        elif self.step_pred_method in ['L-NR','L-RFO']:
            # no checking for failed calculations, is that good?
            self.predictor.update(img_pvecs, nebgrads, energies, self.hessians)
            steps = self.predictor.predict(img_pvecs, nebgrads, energies, self.hessians)

        steps = np.array(steps)

        # zero step and reset for energy None
        steps[energies==None] = np.zeros(int(self.atoms * 3))

        # Enforce maxstep
        if self.max_step is not None:
            steps = spm.enforce_maxsteps(steps, self.max_step)

        # If the user chose analytical position scheme,
        # we now add in the spring steps from the analytical
        # position scheme 
        if self.use_analytical_springpos:
            if self.step_pred_method == 'SCT':
                logger.warning('Analytical spring position scheme cant be used with'
                                + ' self consisten tangents. The NEB will do the SCT'
                                + ' and ignore the use_analytical_springpos')
            else:
                img_pair_ks     = self.path.get_img_pair_ks()
                tanvecs         = self.path.get_tanvecs()
                ap_spring_steps = sfm.calc_analytic_springsteps(full_path_pvecs,
                                                                img_pair_ks,
                                                                tanvecs,
                                                                ci_index)

                # make sure none of the spring steps exceed max stepsize
                ap_spring_steps = spm.enforce_maxsteps(ap_spring_steps, self.max_step)

                # again, we will not compute steps for failed images, so we zero the spring steps in question
                ap_spring_steps[energies==None][:] = 0.0

                # add spring steps to our optimization steps
                steps += ap_spring_steps

        # Apply atom freezing if selected by user
        if self.frozen_atom_indices is not None:
            # their nebgrads should already be zeroed at this point. to ensure 
            # these atoms are not moved, we zero their step vecs as well.
            # convert config entry into an actual python list, then apply atom freezing to all step vectors
            frozen_index_list = parse_index_list(self.frozen_atom_indices)
            steps = np.array([ngm.freeze_atom_indices(step, frozen_index_list) for step in steps])

        return steps

    def giveup_signal_func(self):
        """
        Callback function that checks if NEB has reached an unrecoverable
        state and should be aborted. The only giveup state is if there are too 
        many failed images. We check for that here.
        """
        img_energies = self.path.get_energies()
        failed_img_count = 0
        for energy in img_energies:
            if energy is None:
                failed_img_count += 1
        failure_percentage = failed_img_count / len(img_energies)

        if failure_percentage > self.failed_img_tol_percent:
            # signal that the NEB should be aborted
            logger.error('Too many images failed to converge (%f %% failed). The NEB will be aborted. \nNOTE: the NEB ' +
                         'did not converge! Be careful with the results!', failure_percentage * 100.0)
            return True
        else:
            # Signal that the NEB should not give up
            return False

    def failed_image_replacer_func(self):
        """
        Callback function for reinterpolating failed images
        """
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        full_path_energies = self.path.get_energies(include_ends=True)

        # our interpolation function is the same we used for generating
        # the initial path
        interp_func =  pim.interpolate_path

        # set up additional arguments needed by that function
        interp_func_kwargs = {'labels' : self.labels,
                              'interp_mode' : self.interp_mode,
                              'rot_align_mode' : self.rot_align_mode,
                              'IDPP' : self.IDPP,
                              'SIDPP' : False,      # probably not reasonable to do SIDPP here..
                              'IDPP_maxiter' : self.IDPP_maxiter,
                              'IDPP_max_RMSF' : self.IDPP_max_RMSF,
                              'IDPP_max_AbsF' : self.IDPP_max_AbsF,
                              'max_step' : self.max_step}

        # perform the reinterpolation of failed images
        new_full_path_pvecs = fir.replace_failed_images(full_path_pvecs,
                                                        full_path_energies,
                                                        interp_func,
                                                        interp_func_kwargs)

        # in addition to the image replacing, the images need to
        # be recentered and rotationally realigned every iteration.
        new_full_path_pvecs = sam.align_path(new_full_path_pvecs,
                                                self.rot_align_mode)

        # replace NEBPath images with the now patched path
        patched_images = new_full_path_pvecs[1:-1]
        self.path.set_img_pvecs(patched_images)
        return self.path

    # ----------------------------------------------------------------------------------
    # the following section contains the callback functions for checking
    # NEB convergence. There are several stages of optimization (relaxed NEB,
    # NEB-CI, regular NEB) which have different convergence thresholds.
    # This is implemented by having three slightly different checker functions.

    def conv_checker_func(self, steps):
        if self.relaxed or not self.climbing_image:
            return self.conv_checker_func_noci(steps)
        else:
            return self.conv_checker_func_ci(steps)

    def conv_checker_func_noci(self, steps):
        energies = self.path.get_energies()

        # now perform the check for the signals of convergence
        nebgrads_o = self.path.get_orth_grads()
        nebgrads = self.path.get_nebgrads()

        max_rmsf_o = csm.NEB_RMSF(nebgrads_o)
        max_absf_o = csm.NEB_ABSF(nebgrads_o)
        max_rmsf = csm.NEB_RMSF(nebgrads)
        max_absf = csm.NEB_ABSF(nebgrads)

        # compile rmsf, absf etc. and put them into the log
        values_dict = {'RMSF' : max_rmsf,
                       'AbsF' : max_absf,
                       'RMSF_o': max_rmsf_o,
                       'AbsF_o': max_absf_o}

        if self.logger is not None:
            self.logger.write_to_log(self.path, values_dict)

        # print other information about the current NEB path
        self.do_iter_printout()

        if None in energies:
            logger.warning('There are still failed images left in the path.' +
                           ' There is no point in checking for signals of ' +
                           'convergence yet.')
            return False

        # now check for signals of convergence. get the values from conf dict
        Max_RMSF_tol = self.max_rmsf
        Max_AbsF_tol = self.max_absf

        logger.info('%-30s %f Tol. %f, %s', 'Current max. RMSF:', max_rmsf, Max_RMSF_tol, csm.yesno(max_rmsf<Max_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current max. AbsF:', max_absf, Max_AbsF_tol, csm.yesno(max_absf<Max_AbsF_tol))

        logger.info('%-30s %f Tol. %f', 'Current max. orthogonal RMSF:', max_rmsf_o, Max_RMSF_tol)
        logger.info('%-30s %f Tol. %f', 'Current max. orthogonal AbsF:', max_absf_o, Max_AbsF_tol)

        # return the signal if converged
        if max_rmsf <= Max_RMSF_tol and max_absf <= Max_AbsF_tol:
            # The relaxed NEB did converge succesfully! However, if the NEB calculation continues after this
            # we still need to apply the optimization steps anyways, so this step doesn't get forgotten
            if self.relaxed and not self.relaxed_neb:
                self.relaxed = False
                self.max_rmsf = self.Max_RMSF_tol
                self.max_absf = self.Max_AbsF_tol
                self.climbing_image = self.ci_user_setting
                logger.info('Applying optimization steps.\n')
                self.path.set_img_pvecs(self.path.get_img_pvecs() + steps)
            return True

        else:
            return False

    def conv_checker_func_ci(self, steps):
        energies = self.path.get_energies()

        # now check for signals of convergence. get the values from conf dict
        Max_RMSF_tol = self.Max_RMSF_tol
        Max_AbsF_tol = self.Max_AbsF_tol
        CI_RMSF_tol = self.CI_RMSF_tol
        CI_AbsF_tol = self.CI_AbsF_tol

        # then, separate the nebgrad of the CI from the rest
        nebgrads = self.path.get_nebgrads()
        nebgrads_o = self.path.get_orth_grads()
        ci_index = self.current_CI_index

        ci_nebgrad = nebgrads[ci_index]
        path_nebgrads = np.concatenate([nebgrads[:ci_index],
                                        nebgrads[ci_index+1:]])
        path_nebgrads_o = np.concatenate([nebgrads_o[:ci_index],
                                          nebgrads_o[ci_index+1:]])

        # now perform the check for the signals of convergence
        max_rmsf_o = csm.NEB_RMSF(path_nebgrads_o)
        max_absf_o = csm.NEB_ABSF(path_nebgrads_o)
        max_rmsf = csm.NEB_RMSF(path_nebgrads)
        max_absf = csm.NEB_ABSF(path_nebgrads)
        ci_rmsf = csm.RMS(ci_nebgrad)
        ci_absf = csm.MaxAbs(ci_nebgrad)

        # print other information about the current NEB path
        self.do_iter_printout()

        # compile rmsf, absf etc. and put them into the log
        values_dict = {'RMSF' : max_rmsf,
                       'AbsF' : max_absf,
                       'RMSF_o': max_rmsf_o,
                       'AbsF_o': max_absf_o,
                       'RMSF_CI' : ci_rmsf,
                       'AbsF_CI' : ci_absf}

        if self.logger is not None:
            self.logger.write_to_log(self.path, values_dict)
        if None in energies:
            logger.warning('There are still failed images left in the path.' +
                           ' There is no point in checking for signals of ' +
                           'convergence yet.')
            return False

        logger.info('%-30s %f Tol. %f, %s', 'Current max. RMSF:', max_rmsf, Max_RMSF_tol, csm.yesno(max_rmsf<Max_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current max. AbsF:', max_absf, Max_AbsF_tol, csm.yesno(max_absf<Max_AbsF_tol))

        logger.info('%-30s %f Tol. %f, %s', 'Current CI RMSF:', ci_rmsf, CI_RMSF_tol, csm.yesno(ci_rmsf<CI_RMSF_tol))
        logger.info('%-30s %f Tol. %f, %s', 'Current CI AbsF:', ci_absf, CI_AbsF_tol, csm.yesno(ci_absf<CI_AbsF_tol))

        # return the signal if converged
        if (max_rmsf <= Max_RMSF_tol and
            max_absf <= Max_AbsF_tol and
            ci_rmsf <= CI_RMSF_tol and
            ci_absf <= CI_AbsF_tol):
            return True

        else:
            return False

    def do_iter_printout(self):
        """
        Helper function for printing some NEB stats to the console
        """
        path_pvecs = self.path.get_img_pvecs(include_ends=True)
        energies = self.path.get_energies(include_ends=True)
        labels = self.labels
        io.write_xyz_traj(labels,
                        path_pvecs,
                        self.workdir / 'currenttraj.xyz',
                        energies=energies)
        energies_np = np.array(energies, dtype=float)  # forces None -> np.nan

        # HEI
        hei_index = np.nanargmax(energies_np)
        max_energy = energies_np[hei_index]
        io.write_xyz_file(labels, 
                          path_pvecs[hei_index], 
                          self.workdir / 'HEI_trj.xyz',
                          mode='a',
                          energy=max_energy)

        # TS_guess
        if not self.ci_user_setting:
            ts_coords = pim.interpolate_TS(path_pvecs,
                                        energies_np,
                                        labels,
                                        self.interp_mode)
            io.write_xyz_file(labels, 
                            ts_coords, 
                            self.workdir / 'TS_trj.xyz',
                            mode='a')

        left_barrier_kJmol = (max_energy - energies[0]) * Hartree_in_kJmol
        right_barrier_kJmol = (max_energy - energies[-1]) * Hartree_in_kJmol
        time_elapsed = time.time() - self.start_time

        logger.info('Wall time elapsed (h:m:s): %s', str(timedelta(seconds=time_elapsed)))
        logger.info('There are %d failed images in the path.',  self.path.n_failed_images())

        ci_index = self.current_CI_index
        if ci_index is not None:
            logger.info('Image at index %d is now the climbing image.', ci_index)

        logger.info('Approx. Barrier with respect to left end: %f kJ/mol', left_barrier_kJmol)
        logger.info('Approx. Barrier with respect to right end: %f kJ/mol', right_barrier_kJmol)


class SCT_Optimizer(BasicNEB):
    def __init__(self, NEBPath:NEBPath, dict={}):
        super().__init__(NEBPath, dict)

        # Step predictor is AMGD
        self.predictor = [spm.AMGD(self.harmonic_stepsize_fac, self.AMGD_max_gamma) for _ in range(self.images)]
        self.maxiter = 400

    def predict_steps(self):
        """
        Do the step prediction with the saved step predictors
        """
        # first gather a bunch of data from the NEBPath
        nebgrads        = self.path.get_nebgrads()
        img_pvecs       = self.path.get_img_pvecs(include_ends=False)

        # Update the AMGD objects
        for object, pvec in zip(self.predictor, img_pvecs):
            object.update(pvec)

        # Perform the step prediction
        steps = [object.predict(nebgrad) 
                 for object, nebgrad in zip(self.predictor, nebgrads)]

        # Enforce maxstep
        if self.max_step is not None:
            steps = spm.enforce_maxsteps(steps, self.max_step)
        return steps

    def calc_springgrads(self):
        """
        Callback function used to calculate springforce gradients
        of the images, as well as to recalculate the variable k
        constants if variable k and CI are active.
        """
        if self.use_vark:
            self.recalculate_varks()

        else:
            # if not, set all ks to be the value set in the conf_dict, to make sure
            self.path.set_img_k_const(self.k_const)

        # calculate regular springforce gradients.
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        img_pair_ks = self.path.get_img_pair_ks()
        tanvecs = self.path.get_tanvecs()

        if self.spring_gradient == 'difference':
            springs = sfm.delta_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [spring * tanvec 
                           for spring, tanvec in zip(springs, tanvecs)]

        elif self.spring_gradient == 'projected':
            full_springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)
            springgrads = [ngm.project(full_springgrad, tanvec) 
                           for full_springgrad, tanvec in zip(full_springgrads, tanvecs)]

        elif self.spring_gradient == 'raw':
            springgrads = sfm.full_springgrads(full_path_pvecs, img_pair_ks)

        else:
            raise NEBError('Error with springforce definition. %s is not a valid springforce mode.',
                           str(self.spring_gradient))

        # if ci is active, the springforces acting on the ci are zero
        ci_index = self.current_CI_index
        if ci_index is not None:
            springgrads[ci_index][:] = 0.0

        return springgrads

    def calc_tanvecs(self):
        """
        Callback function for computing image tangent vectors.
        we want it to be able to both return normal tangents,
        or apply smoothing for the tangent vectors, depending on
        what the user chose.
        """
        full_path_pvecs = self.path.get_img_pvecs(include_ends=True)
        full_path_energies = self.path.get_energies(include_ends=True)
        
        # use tangent definition
        if self.tangents == 'henkjon':
            raw_tans = tgm.henkjon_tans(full_path_pvecs,
                                        full_path_energies)
        elif self.tangents == 'simple':
            raw_tans = tgm.simple_tans(full_path_pvecs)
        else:
            raise NEBError('Error with tangent definition. %s is not a valid tangent mode.',
                           str(self.tangents))
        return raw_tans

    def calc_nebgrads(self):
        """
        Callback function for calculating neb gradients,
        after engrads, springgrads, tanvecs have all been
        calculated and saved in the NEBPath object.
        """
        ci_index = self.current_CI_index
        engrads = self.path.get_engrads()
        img_pvecs = self.path.get_img_pvecs(include_ends=False)

        raw_nebgrads = ngm.calculate_nebgrads(engrads,
                                              self.path.get_springgrads(),
                                              self.path.get_tanvecs(),
                                              ci_index)

        # project out translation and/or rotation from neb gradients
        # (experimental feature), if selected by user
        nebgrads = ngm.sanitize_stepvecs(raw_nebgrads,
                                         img_pvecs,
                                         self.remove_gradtrans,
                                         self.remove_gradrot)

        # zero the gradients corresponding to frozen atoms,
        # so they don't skew the signals of convergence thresholds
        frozen_atom_indices = self.frozen_atom_indices

        if frozen_atom_indices is not None:
            # convert config entry into an actual python list,
            # then apply atom freezing for each step vector
            frozen_index_list = parse_index_list(frozen_atom_indices)

            for i in range(len(nebgrads)):
                nebgrads[i] = ngm.freeze_atom_indices(nebgrads[i], frozen_index_list)

        # calculate the orthogonal gradients
        # project out the spring contribution, which is parallel to the tangent
        tanvecs = self.path.get_tanvecs()
        orth_grads = np.zeros_like(nebgrads)
        for i in range(len(nebgrads)):
            # except if there is a climbing image, which has a special NEB gradient.
            if i != ci_index:
                orth_grads[i] = ngm.reject(nebgrads[i], tanvecs[i])
        return nebgrads, orth_grads
    
    def recalculate_varks(self):
        """
        Helper function for the spring gradient function.
        Sets the spring constants for all image pairs according to
        the improved variable k scheme.
        """
        full_path_energies = self.path.get_energies(include_ends=True)
        maxk = self.k_const
        mink = self.k_const * self.vark_min_fac

        pairwise_ks = sfm.compute_pairwise_ks(full_path_energies,
                                              maxk,
                                              mink)
        self.path.set_img_pair_ks(pairwise_ks)

    def conv_checker_func(self, steps):
        nebgrads = self.path.get_nebgrads()

        rmsf = csm.NEB_RMSF(nebgrads)
        absf = csm.NEB_ABSF(nebgrads)

        max_rmsf = self.harmonic_conv_fac * self.Max_RMSF_tol
        max_absf = self.harmonic_conv_fac * self.Max_AbsF_tol

        logger.info('Harmonic Max. RMSF: %f Tol.: %f', rmsf, max_rmsf)
        logger.info('Harmonic Max. Abs. F: %f Tol.: %f', absf, max_absf)

        if rmsf < max_rmsf and absf < max_absf:
            return True
        else:
            return False
