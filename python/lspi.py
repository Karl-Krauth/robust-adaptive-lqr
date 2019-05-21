"""lspi.py

"""

import numpy as np
import utils
import logging
import cvxpy as cvx
import scipy.linalg

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
                 sigma_eta,
                 epoch_mult,
                 num_PI_iters,
                 K_init):
        super().__init__(Q, R, A_star, B_star, sigma_w, None)
        self._logger = logging.getLogger(__name__)
        self._epoch_len = epoch_mult
        self._mu = max(utils.min_eigvalsh(Q), utils.min_eigvalsh(R))
        self._sigma_eta = sigma_eta
        self._L = np.inf
        self._epoch_mult = epoch_mult
        self._num_PI_iters = num_PI_iters
        self._Kt = K_init

    def _get_logger(self):
        return self._logger

    def _design_controller(self, states, inputs, transitions, rng):
        _, n = states.shape
        for i in range(self._num_PI_iters):
            print(utils.spectral_radius(self._A_star + self._B_star @ self._Kt))
            Qt = self._lstdq(states, inputs, transitions, self._Kt,
                             self._mu, self._L)
            Ktp1 = -scipy.linalg.solve(Qt[n:, n:], Qt[:n, n:].T, sym_pos=True)
            if utils.spectral_radius(self._A_star + self._B_star @ Ktp1) >= 1:
                print(i)
                break
            self._Kt = Ktp1
        print(self._Kt)
        return (self._A_star, self._B_star, self._sigma_w ** 2)

    def _lstdq(self, states, inputs, transitions, Keval, mu, L):
        T, n = states.shape
        _, d = inputs.shape
        lifted_dim = (n + d) * (n + d + 1) // 2
        costs = np.sum(states * (states @ self._Q), axis=1)

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

    def _should_terminate_epoch(self):
        if self._iteration_within_epoch_idx >= self._epoch_len:
            self._epoch_len *= 2
            self._sigma_eta /= 2 ** (1.0 / 6.0)
            return True
        else:
            return False

    def _get_input(self, state, rng):
        return self._Kt @ state + rng.normal(scale=self._sigma_eta, size=self._Kt.shape[0])
