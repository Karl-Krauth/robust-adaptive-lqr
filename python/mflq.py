"""mflq.py

"""

import numpy as np
import utils
import logging
import cvxpy as cvx
import scipy.linalg
import math

from adaptive import AdaptiveMethod
from numba import jit


class RankDegeneracyException(Exception):
    def __init__(self, msg=None):
        super().__init__(msg)


@jit(nopython=True)
def phi(x):
    return utils.svec(np.outer(x, x))

@jit(nopython=True)
def psi(x, u):
    z = np.hstack((x, u))
    return utils.svec(np.outer(z, z))

class MFLQStrategy(AdaptiveMethod):

    def __init__(self,
                 Q,
                 R,
                 A_star,
                 B_star,
                 sigma_w,
                 sigma_explore,
                 epoch_length,
                 exploration_period,
                 K_init):
        super().__init__(Q, R, A_star, B_star, sigma_w, None)
        self._sigma_explore = sigma_explore
        self._epoch_length = epoch_length
        self._exploration_period = exploration_period

        self._Kt = K_init

        self._logger = logging.getLogger(__name__)

        self._Phis = None
        self._Phis_plus = None
        self._Psis = None
        self._costs = None
        self._G_sum = None

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        T, n = states.shape
        _, d = inputs.shape

        phi_dim = n * (n + 1) // 2
        psi_dim = (n + d) * (n + d + 1) // 2

        logger = self._get_logger()
        logger.info("_design_controller(epoch={}): n_transitions={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            states.shape[0]))

        if self._Phis is None:
            assert self._Phis_plus is None
            assert self._costs is None
            assert self._Psis is None
            assert self._G_sum is None
            self._Phis = np.zeros((states.shape[0], phi_dim))
            self._Phis_plus = np.zeros((states.shape[0], phi_dim))
            self._Psis = np.zeros((states.shape[0], psi_dim))
            self._G_sum = np.zeros((n + d, n + d))
            for i in range(states.shape[0]):
                self._Phis[i] = phi(states[i])
                self._Phis_plus[i] = phi(transitions[i])
                self._Psis[i] = psi(states[i], inputs[i])
            self._costs = (np.diag((states @ self._Q) @ states.T) +
                           np.diag((inputs @ self._R) @ inputs.T))
        else:
            assert self._Phis_plus is not None
            assert self._costs is not None
            assert self._Psis is not None
            assert self._Phis.shape[0] == self._Psis.shape[0]
            assert self._Phis.shape[0] == self._Phis_plus.shape[0]
            base_idx = self._Phis.shape[0]
            newPhis = np.zeros((states.shape[0] - base_idx, phi_dim))
            newPhis_plus = np.zeros((states.shape[0] - base_idx, phi_dim))
            newPsis = np.zeros((states.shape[0] - base_idx, psi_dim))
            for i in range(newPhis.shape[0]):
                newPhis[i] = phi(states[base_idx + i])
                newPhis_plus[i] = phi(transitions[base_idx + i])
                newPsis[i] = psi(states[base_idx + i], inputs[base_idx + i])
            newCosts = (np.diag((states[base_idx:] @ self._Q) @ states[base_idx:].T) +
                        np.diag((inputs[base_idx:] @ self._R) @ inputs[base_idx:].T))
            self._Phis = np.vstack((self._Phis, newPhis))
            self._Phis_plus = np.vstack((self._Phis_plus, newPhis_plus))
            self._Psis = np.vstack((self._Psis, newPsis))
            self._costs = np.hstack((self._costs, newCosts))

        Gt = self._estimate_G(self._Phis, self._Phis_plus, self._Psis, self._costs,
                              self._sigma_w, n)
        self._G_sum += Gt
        self._Kt = -np.linalg.solve(self._G_sum[d:, d:], self._G_sum[d:, :n])

        rho_true = utils.spectral_radius(self._A_star + self._B_star @ self._Kt)
        logger.info("_design_controller(epoch={}): rho(A_* + B_* K)={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            rho_true))
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._Kt, self._Q, self._R, self._sigma_w)
        return (self._A_star, self._B_star, Jnom)

    def _estimate_G(self, Phis, Phis_plus, Psis, costs, sigma_w, n):
        W = utils.svec(np.eye(n) * sigma_w ** 2)
        Amat = Phis.T @ (Phis - Phis_plus + W)
        bmat = Phis.T @ costs
        hhat = np.linalg.lstsq(Amat, bmat)[0]
        Hhat_proj = utils.psd_project(utils.smat(hhat) - self._Q, 0, np.inf) + self._Q
        hhat_proj = utils.svec(Hhat_proj)

        ghat = scipy.linalg.solve(Psis.T @ Psis, Psis.T @ (costs + (Phis_plus - W) @ hhat_proj), sym_pos=True)
        cost_block = np.block([[self._Q, np.zeros([self._Q.shape[0], self._R.shape[1]])],
                               [np.zeros([self._Q.shape[1], self._R.shape[0]]), self._R]])
        Ghat = utils.psd_project(utils.smat(ghat) - cost_block, 0, np.inf) + cost_block
        return Ghat

    def _should_terminate_epoch(self):
        if self._iteration_within_epoch_idx >= self._epoch_length:
            return True
        else:
            return False

    def _get_input(self, state, rng):
        if (self._iteration_within_epoch_idx + 1) % self._exploration_period == 0:
            rng = self._get_rng(rng)
            explore_input = self._sigma_explore * rng.normal(size=(self._Kt.shape[0],))
            return explore_input
        else:
            ctrl_input = self._Kt @ state
            return ctrl_input
