"""lspi.py

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
def phi(x, u):
    z = np.hstack((x, u))
    return utils.svec(np.outer(z, z))


@jit(nopython=True)
def _fill_Psis(Phis, Psis, next_states, next_inputs, f):
    T, _ = Phis.shape
    for t in range(T):
        Psis[t] = Phis[t] - phi(next_states[t], next_inputs[t]) + f


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

        self._Phis = None
        self._costs = None

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        T, n = states.shape
        _, d = inputs.shape

        lifted_dim = (n + d) * (n + d + 1) // 2

        logger = self._get_logger()
        logger.info("_design_controller(epoch={}): n_transitions={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            states.shape[0]))

        if self._Phis is None:
            assert self._costs is None
            self._Phis = np.zeros((states.shape[0], lifted_dim))
            for i in range(states.shape[0]):
                self._Phis[i] = phi(states[i], inputs[i])
            self._costs = (np.diag((states @ self._Q) @ states.T) +
                           np.diag((inputs @ self._R) @ inputs.T))
        else:
            assert self._costs is not None
            base_idx = self._Phis.shape[0]
            newPhis = np.zeros((states.shape[0] - base_idx, lifted_dim))
            for i in range(newPhis.shape[0]):
                newPhis[i] = phi(states[base_idx + i], inputs[base_idx + i])
            newCosts = (np.diag((states[base_idx:] @ self._Q) @ states[base_idx:].T) +
                        np.diag((inputs[base_idx:] @ self._R) @ inputs[base_idx:].T))
            self._Phis = np.vstack((self._Phis, newPhis))
            self._costs = np.hstack((self._costs, newCosts))

        # TODO(stephentu):
        # this is a hack
        if T <= 2000:
            num_iters = self._num_PI_iters
        elif T <= 4000:
            num_iters = self._num_PI_iters + 1
        elif T <= 6000:
            num_iters = self._num_PI_iters + 2
        else:
            num_iters = self._num_PI_iters + 3

        logger.info("num_iters={}".format(num_iters))
        for i in range(num_iters):
            Qt = self._lstdq(self._Phis, transitions, self._costs, self._Kt,
                             self._sigma_w, self._mu, self._L)
            Ktp1 = -scipy.linalg.solve(Qt[n:, n:], Qt[:n, n:].T, sym_pos=True)
            self._Kt = Ktp1

        rho_true = utils.spectral_radius(self._A_star + self._B_star @ self._Kt)
        logger.info("_design_controller(epoch={}): rho(A_* + B_* K)={}".format(
            self._epoch_idx + 1 if self._has_primed else 0,
            rho_true))
        Jnom = utils.LQR_cost(self._A_star, self._B_star, self._Kt, self._Q, self._R, self._sigma_w)
        return (self._A_star, self._B_star, Jnom)

    def _lstdq(self, Phis, next_states, costs, Keval, sigma_w, mu, L):
        _, n = next_states.shape

        I_K = np.vstack((np.eye(n), Keval))
        f = (sigma_w ** 2) * utils.svec(I_K.dot(I_K.T))

        Psis = np.zeros_like(Phis)
        next_inputs = next_states.dot(Keval.T)

        _fill_Psis(Phis, Psis, next_states, next_inputs, f)

        Amat = Phis.T.dot(Psis)
        bmat = Phis.T.dot(costs)

        svals = scipy.linalg.svdvals(Amat)
        if min(svals) <= 1e-8:
            raise RankDegeneracyException(
                "Amat is degenerate: s_min(Amat)={}".format(min(svals)))
        qhat = np.linalg.lstsq(Amat, bmat)[0]
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
