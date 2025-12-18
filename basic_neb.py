import logging

logger = logging.getLogger(__name__)

class BasicNEB:
    def __init__(self, NEBPath, dict={}):
        self.path = NEBPath
        self.iteration = 0
        self.state = 'FAILED'
        self.labels = NEBPath.get_labels()
        self.atoms = len(self.labels)
        self.climbing_image = False
        self.maxiter = 50

        # save all information from config
        self.dict = dict
        for key, value in dict.items():
            setattr(self, key, value)

    @property
    def images(self):
        return self.path.n_images()

    @property
    def current_CI_index(self):
        """
        This function returns the NEBPath_obj's current
        CI index if CI is currently active, or None
        if CI is inactive
        """
        if self.climbing_image:
            # update CI to be the highest energy image
            self.path.update_ci_index()
            ci_index = self.path.get_ci_index()

        else:
            ci_index = None

        return ci_index

    def do_opt_loop(self,
                engrad_calc_func,
                engrad_calc_kwargs={},
                silent_mode=False):

        # In silent mode the logging is reduced
        if silent_mode:
            root_logger = logging.getLogger()
            old_level = root_logger.getEffectiveLevel()
            root_logger.setLevel(logging.WARNING)

        try:
            logger.info('Beginning NEB optimization loop.\n')

            # Set state to failed in beginning
            self.state = 'FAILED'

            while self.iteration < self.maxiter:
                self.iteration += 1

                logger.info('Iteration no. %d', self.iteration)
                logger.info('Updating energies and gradients.')

                energies, engrads = engrad_calc_func(self.path.get_img_pvecs(),
                                                    **engrad_calc_kwargs)

                self.path.set_energies(energies)
                self.path.set_engrads(engrads)

                logger.info('Updating tangent vectors.')

                tanvecs = self.calc_tanvecs()
                self.path.set_tanvecs(tanvecs)
                logger.debug('Tangents have shape %s', len(tanvecs[0]))

                logger.info('Updating springforce gradients.')

                springgrads = self.calc_springgrads()
                self.path.set_springgrads(springgrads)
                logger.debug('Springgrads have shape %s', len(springgrads[0]))

                logger.info('Calculating NEB gradients.')

                # this function also needs to take care of gradient freezing
                nebgrads, orth_grads = self.calc_nebgrads()
                self.path.set_nebgrads(nebgrads)
                self.path.set_orth_grads(orth_grads)
                logger.debug('NEB-grads have shape %s', len(nebgrads[0]))

                logger.info('Calculating optimization steps.')

                # this function also needs to take care of step freezing
                # and of doing analytical position steps for spring forces
                steps = self.predict_steps()

                logger.info('Checking for signals of convergence.')

                # this function also needs to take care of logging
                if self.conv_checker_func(steps):
                    self.state = 'SUCCESS'
                    break

                # this function should be used to abort the NEB if it has
                # reached an unrecoverable state. It should also print
                # a message as to why the NEB was aborted.
                if self.giveup_signal_func():
                    break

                logger.info('Applying optimization steps.')
                self.path.set_img_pvecs([pvec + step for pvec, step in zip(self.path.get_img_pvecs(), steps)])

                # This function should try to replace structures
                # if their current energy is None, indicating they didn't converge,
                # indicating that their structures are nonsensical
                logger.info('Checking for failed images.\n')
                self.path = self.failed_img_repl_func()

        finally:
            if silent_mode:
                root_logger.setLevel(old_level)
        return self.path, self.state, self.iteration

    def giveup_signal_func(self):
        """
        Placeholder Function
        """
        return False

    def failed_img_repl_func(self):
        """
        Placeholder Function
        """
        return self.path

    def calc_springgrads(self):
        """
        Placeholder Function
        """
        return None

    def calc_nebgrads(self):
        """
        Placeholder Function
        """
        return None

    def calc_tanvecs(self):
        """
        Placeholder Function
        """
        return None

    def predict_steps(self):
        """
        Placeholder Function
        """
        return None

    def conv_checker_func(self):
        """
        Placeholder Function
        """
        return False