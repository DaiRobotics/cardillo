import numpy as np
from jax import numpy as jnp

from cardillo.rods import DiscreteRod
from cardillo.contributions import HContribution, TauContribution


class Force_line_distributed(HContribution):

    def __init__(self, force, rod):
        r"""Line distributed dead load for rods

        Parameters
        ----------
        force : np.ndarray (3,)
            Force w.r.t. inertial I-basis as a callable function in time t and
            rod position xi.
        rod : CosseratRod
            Cosserat rod from Cardillo.

        """
        HContribution.__init__(self)
        self.force = (lambda t, xi: force) if not callable(force) else force
        self.rod = rod
        try:
            self._jaxed = rod._jaxed
        except AttributeError:
            self._jaxed = False
        self._is_discrete_rod = isinstance(rod, DiscreteRod)
        if self._is_discrete_rod:
            self._h_weights = (
                np.pad(rod.L_els, (1, 0)) + np.pad(rod.L_els, (0, 1))
            ) / 2
            self._h_weights = np.repeat(self._h_weights, 3)

    def assembler_callback(self):
        if self._is_discrete_rod:
            self.qDOF = self.rod.qDOF
            self.uDOF = self.rod.uDOF.reshape((-1, 6))[:, :3].ravel()
        else:
            self.qDOF = self.rod.qDOF
            self.uDOF = self.rod.uDOF

    ##################
    # potential energy
    ##################
    def E_pot(self, t, q):
        raise NotImplementedError
        E_pot = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            E_pot += self.E_pot_el(t, qe, el)
        return E_pot

    def E_pot_el(self, t, qe, el):
        raise NotImplementedError
        # TODO: nullify with initial configuration q0
        E_pot_el = 0.0

        for i in range(self.rod.nquadrature):
            # extract reference state variables
            qpi = self.rod.qp[el, i]
            qwi = self.rod.qw[el, i]
            Ji = self.rod.J[el, i]

            # interpolate centerline position
            r_OC = np.zeros(3, dtype=float)
            for node in range(self.nnodes_element_r):
                r_OC += self.N_r[el, i, node] * qe[self.nodalDOF_element_r[node]]

            E_pot_el -= (r_OC @ self.force(t, qpi)) * Ji * qwi

        return E_pot_el

    #####################
    # equations of motion
    #####################
    def h_jax(self, t, q, u):
        if self._is_discrete_rod:
            return (
                jnp.concatenate([self.force(t, xi) for xi in self.rod.xi_node])
                * self._h_weights
            )
        else:
            h = jnp.zeros(self.rod.nu, dtype=np.common_type(q, u))
            for el in range(self.rod.nelement):
                h = h.at[self.rod.elDOF_u[el]].add(self.h_el(t, el))
            return h

    def h(self, t, q, u):
        if self._is_discrete_rod:
            return (
                np.concatenate([self.force(t, xi) for xi in self.rod.xi_node])
                * self._h_weights
            )
        else:
            h = np.zeros(self.rod.nu, dtype=np.common_type(q, u))
            for el in range(self.rod.nelement):
                h[self.rod.elDOF_u[el]] += self.h_el(t, el)
            return h

    def h_el(self, t, el):
        he = np.zeros(self.rod.nu_element, dtype=float)

        for i in range(self.rod.nquadrature):
            # extract reference state variables
            qpi = self.rod.qp[el, i]
            qwi = self.rod.qw[el, i]
            Ji = self.rod.J[el, i]

            # compute local force vector
            he_qp = self.force(t, qpi) * Ji * qwi

            # multiply local force vector with variation of centerline
            for node in range(self.rod.nnodes_element_r):
                he[self.rod.nodalDOF_element_r[node]] += (
                    self.rod.N_r[el, i, node] * he_qp
                )

        return he


class RodGravity(TauContribution):

    def __init__(self, rod, density, g_accel=9.81, g_direction=np.array([0, 0, -1])):
        TauContribution.__init__(self, nla_tau=1, ntau=1)

        assert isinstance(rod, DiscreteRod), "RodGravity only supports DiscreteRod."

        self.rod = rod
        self.density = density
        self.g_accel = g_accel
        self.direction = g_direction
        self._jaxed = True

        l_nodes = (np.pad(rod.L_els, (1, 0)) + np.pad(rod.L_els, (0, 1))) / 2
        l_nodes = np.repeat(l_nodes, 3)

        self._W_tau = l_nodes * np.concatenate(
            [
                density * rod.cross_section.area(xi) * g_accel * g_direction
                for xi in rod.xi_node
            ]
        )

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF.reshape((-1, 6))[:, :3].ravel()

        self.W_tau_jax_row = np.arange(len(self.uDOF), dtype=int)
        self.W_tau_jax_col = np.zeros_like(self.W_tau_jax_row)

    def W_tau_jax(self, t, q):
        return self._W_tau

    def W_tau(self, t, q):
        return self._W_tau

    def la_tau_jax(self, t, q, u, tau):
        return tau

    def la_tau(self, t, q, u, tau):
        return tau

    def tau(self, t):
        return 0.0
