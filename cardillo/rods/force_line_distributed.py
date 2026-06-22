import numpy as np
from jax import jit, vmap
from jax import numpy as jnp

from cardillo.rods import DiscreteRod


class Force_line_distributed:
    @staticmethod
    def _make_h_nodes(force):
        def h_node(t, xi, weight):
            return jnp.pad(force(t, xi), (0, 3)) * weight

        return jit(vmap(h_node, in_axes=(None, 0, 0)))

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
        if not callable(force):
            _force = lambda t, xi: force
        else:
            _force = force
        self.rod = rod
        self._is_discrete_rod = isinstance(rod, DiscreteRod)
        if self._is_discrete_rod:
            self._h_weights = (
                np.pad(rod.L_els, (1, 0)) + np.pad(rod.L_els, (0, 1))
            ) / 2
            self._h_nodes = Force_line_distributed._make_h_nodes(_force)
        else:
            self.force = _force

    def assembler_callback(self):
        self.qDOF = self.rod.qDOF
        self.uDOF = self.rod.uDOF

    ##################
    # potential energy
    ##################
    def E_pot(self, t, q):
        E_pot = 0
        for el in range(self.nelement):
            qe = q[self.elDOF[el]]
            E_pot += self.E_pot_el(t, qe, el)
        return E_pot

    def E_pot_el(self, t, qe, el):
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
    def h(self, t, q, u):
        if self._is_discrete_rod:
            return (
                (self._h_nodes(t, self.rod.xi_node, self._h_weights))
                .__array__()
                .ravel()
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
