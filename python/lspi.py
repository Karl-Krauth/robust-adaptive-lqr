"""lspi.py

"""

import numpy as np
import utils
import logging
import cvxpy as cvx
import scipy.linalg
import math

from adaptive import AdaptiveMethod


class RankDegeneracyException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


def phi(x, u):
    z = np.hstack((x, u))
    return utils.svec(np.outer(z, z))

class LSPIStrategy(AdaptiveMethod):

    def __init__(self,
                 Q,
                 R,
                 A_star,
                 B_star,
                 sigma_w,
                 sigma_explore,
                 epoch_multiplier,
                 num_PI_iters,
                 K_init):
        super().__init__(Q, R, A_star, B_star, sigma_w, None)
        self._sigma_explore = sigma_explore
        self._epoch_multiplier = epoch_multiplier

        self._mu = min(utils.min_eigvalsh(Q), utils.min_eigvalsh(R))
        self._L = np.inf
        self._num_PI_iters = num_PI_iters
        self._Kt = K_init

        self._logger = logging.getLogger(__name__)

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        _, n = states.shape
        logger = self._get_logger()
        logger.info("_design_controller(epoch={}): n_transitions={}".format(self._epoch_idx + 1, states.shape[0]))

        for i in range(self._num_PI_iters):
            #print(utils.spectral_radius(self._A_star + self._B_star @ self._Kt))
            Qt = self._lstdq(states, inputs, transitions, self._Kt,
                             self._mu, self._L)
            Ktp1 = -scipy.linalg.solve(Qt[n:, n:], Qt[:n, n:].T, sym_pos=True)
            #if utils.spectral_radius(self._A_star + self._B_star @ Ktp1) >= 1:
            #    print(i)
            #    break
            self._Kt = Ktp1
        #print(self._Kt)

        rho_true = utils.spectral_radius(self._A_star + self._B_star @ self._Kt)
        logger.info("_design_controller(epoch={}): rho(A_* + B_* K)={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            rho_true))
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._Kt, self._Q, self._R, self._sigma_w)
        return (self._A_star, self._B_star, Jnom)

    def _lstdq(self, states, inputs, transitions, Keval, mu, L):
        T, n = states.shape
        _, d = inputs.shape
        lifted_dim = (n + d) * (n + d + 1) // 2
        #costs = np.sum(states * (states @ self._Q), axis=1)
        costs = (np.diag((states @ self._Q) @ states.T) +
                 np.diag((inputs @ self._R) @ inputs.T))

        I_K = np.vstack((np.eye(n), Keval))
        f = (self._sigma_w ** 2) * utils.svec(I_K @ I_K.T)
        Phis = np.zeros((T, lifted_dim))
        diffs = np.zeros_like(Phis)
        for t in range(T):
            Phis[t] = phi(states[t], inputs[t])
            diffs[t] = Phis[t] - phi(transitions[t], Keval @ transitions[t]) + f
        Amat = Phis.T @ diffs
        bmat = Phis.T @ costs
        svals = scipy.linalg.svdvals(Amat)
        if min(svals) <= 1e-8:
            raise RankDegeneracyException(
                "Amat is degenerate: s_min(Amat)={}".format(min(svals)))
        qhat = np.linalg.lstsq(Amat, bmat, rcond=None)[0]
        Qhat = utils.psd_project(utils.smat(qhat), mu, L)
        return Qhat

    def _epoch_length(self):
        return self._epoch_multiplier * (self._epoch_idx + 1)

    def _explore_stddev(self):
        sigma_explore_decay = 1/math.pow(self._epoch_idx + 1, 1/3)
        return self._sigma_explore * sigma_explore_decay

    def _should_terminate_epoch(self):
        if self._iteration_within_epoch_idx >= self._epoch_length():
            return True
        else:
            return False

    def _get_input(self, state, rng):
        rng = self._get_rng(rng)
        ctrl_input = self._Kt @ state
        explore_input = self._explore_stddev() * rng.normal(size=(self._Kt.shape[0],))
        return ctrl_input + explore_input
