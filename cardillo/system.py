import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu

from .frame import Frame
from .utility.coo_matrix import CooMatrix
from .visualization import Export
from .solver import SolverOptions

properties = []

properties.extend(["M", "Mu_q"])

properties.extend(["h", "h_q", "h_u"])

properties.extend(["q_dot", "q_dot_q", "q_dot_u"])

properties.extend(["g"])

properties.extend(["c", "c_q", "c_u"])

properties.extend(["g_S"])

properties.extend(["assembler_callback", "step_callback"])


IS_CLOSE_ATOL = 1e-8


def consistent_initial_conditions(
    system,
    options=SolverOptions(),
):
    """Checks consistency of initial conditions with constraints on position and velocity level and finds initial accelerations and constraint/contact forces.

    Parameters
    ----------
    system : cardillo.System
        System for which the consistent initial conditions are computed.
    options : cardillo.solver.SolverOptions
        Solver options for the computations of the constraint/contact forces.
    """
    t0 = system.t0
    q0 = system.q0
    u0 = system.u0

    # normalize quaternions etc.
    q0, u0 = system.step_callback(t0, q0, u0)

    q_dot0 = system.q_dot(t0, q0, u0)

    if (
        not options.compute_consistent_initial_conditions or system.nu == 0
    ):  # second case can happen during debugging, when only frames are added to the system
        return (
            t0,
            q0,
            u0,
            q_dot0,
            np.zeros(system.nu),
            np.zeros(system.nla_g),
            np.zeros(system.nla_c),
        )

    # evaluate constant quantities
    M = system.M(t0, q0)
    h = system.h(t0, q0, u0)

    W_g = system.W_g(t0, q0)
    g_dot_u = system.g_dot_u(t0, q0)
    zeta_g = system.zeta_g(t0, q0, u0)

    W_c = system.W_c(t0, q0)
    la_c0 = system.la_c(t0, q0, u0)

    split_x = np.cumsum(
        [
            system.nu,
            system.nla_g,
        ]
    )[:-1]

    # fmt: off
    A = bmat(
        [
            [      M, -W_g],
            [g_dot_u, None],
        ],
        format="csc",
    )
    # fmt: on

    lu = splu(A)

    b0 = np.concatenate(
        [
            h + W_c @ la_c0,
            -zeta_g,
        ]
    )

    x0 = np.zeros(system.nu + system.nla_g)

    # compute accelerations and constraints without contacts
    x0 = lu.solve(b0)

    u_dot0, la_g0 = np.array_split(x0, split_x)

    # check if initial conditions satisfy constraints on position, velocity
    # and acceleration level
    g0 = system.g(t0, q0)
    g_dot0 = system.g_dot(t0, q0, u0)
    g_ddot0 = system.g_ddot(t0, q0, u0, u_dot0)
    g_S0 = system.g_S(t0, q0)

    assert np.allclose(
        g0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g0!"
    assert np.allclose(
        g_dot0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_dot0!"
    assert np.allclose(
        g_ddot0, np.zeros(system.nla_g), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_ddot0!"
    assert np.allclose(
        g_S0, np.zeros(system.nla_S), atol=IS_CLOSE_ATOL
    ), "Initial conditions do not fulfill g_S0!"

    return t0, q0, u0, q_dot0, u_dot0, la_g0, la_c0


class System:
    """Sparse model implementation which assembles all global objects without
    copying on body and element level.

    Parameters
    ----------
    t0 : float
        Initial time of the initial state of the system.

    Notes
    -----

    All model functions which return matrices have :py:class:`scipy.sparse.coo_array`
    as default scipy sparse matrix type (:py:class:`scipy.sparse.spmatrix`).
    This is due to the fact that the assembling of global iteration matrices
    is done using :py:func:`scipy.sparse.bmat` which in a first step transforms
    all matrices to :py:class:`scipy.sparse.coo_array`. A :py:class:`scipy.sparse.coo_array`,
    inherits form :py:class:`scipy.sparse._data_matrix`
    `[1] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/data.py#L21-L126>`_,
    have limited support for arithmetic operations, only a few operations as
    :py:func:`__neg__`, :py:func:`__imul__`, :py:func:`__itruediv__` are implemented.
    For all other operations the matrix is first transformed to a :py:class:`scipy.sparse.csr_array`
    `[2] <https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/sparse/base.py#L330-L335>`_.
    Slicing is also not supported for matrices of type :py:class:`scipy.sparse.coo_array`,
    we have to use other formats as :py:class:`scipy.sparse.csr_array` or
    :py:class:`scipy.sparse.csc_array` for that.

    """

    def __init__(self, t0=0.0):
        self.t0 = t0
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_c = 0
        self.nla_S = 0

        self.contributions = []
        self.contributions_map = {}
        self.ncontr = 0

        self.origin = Frame()

        self.origin.name = "cardillo_origin"
        self.add(self.origin)

    def add(self, *contrs):
        """Adds contributions to the system.

        Parameters
        ----------
        contrs : object or list
            Single object or list of objects to add to the system.
        """
        for contr in contrs:
            if not contr in self.contributions:
                self.contributions.append(contr)
                if not hasattr(contr, "name"):
                    contr.name = "contr" + str(self.ncontr)

                if contr.name in self.contributions_map:
                    new_name = contr.name + "_contr" + str(self.ncontr)
                    print(
                        f"There is another contribution named '{contr.name}' which is already part of the system. Changed the name to '{new_name}' and added it to the system."
                    )
                    contr.name = new_name
                self.contributions_map[contr.name] = contr
                self.ncontr += 1
            else:
                raise ValueError(f"contribution {str(contr)} already added")

    def remove(self, *contrs):
        for contr in contrs:
            if contr in self.contributions:
                self.contributions.remove(contr)
            else:
                raise ValueError(f"no contribution {str(contr)} to remove")

    def pop(self, index):
        self.contributions.pop(index)

    def set_new_initial_state(self, q0, u0, t0=None, **assemble_kwargs):
        """
        Sets the initial state of the system.

        Parameters:
        -----------
        q0 : np.ndarray
            initial position coordinates
        u0 : np.ndarray
            initial velocity coordinates
        t0 : float
            initial time

        """
        self.t0 = t0 if t0 is not None else self.t0

        # extract final generalized coordiantes and distribute to subsystems
        for contr in self.contributions:
            if hasattr(contr, "nq"):
                contr.q0 = q0[contr.my_qDOF]

        # optionally distribute all other solution fields
        for contr in self.contributions:
            if hasattr(contr, "nu"):
                contr.u0 = u0[contr.my_uDOF]

        self.assemble(**assemble_kwargs)

    def export(self, path, folder_name, solution, overwrite=True, fps=50):
        e = Export(path, folder_name, overwrite, fps, solution)
        for contr in self.contributions:
            if hasattr(contr, "export"):
                e.export_contr(contr, file_name=contr.name)
        return e

    def assemble(self, *args, **kwargs):
        """Assembles the system, i.e., counts degrees of freedom, sets connectivities and assembles global initial state.

        Parameters
        ----------
        slice_active_contacts : bool
            When computing consistent initial conditions, slice friction forces to contemplate only those corresponding to active normal contact.
        options : cardillo.solver.SolverOptions
            Solver options for the computation of the constraint/contact forces.
        """
        self.nq = 0
        self.nu = 0
        self.nla_g = 0
        self.nla_c = 0
        self.nla_S = 0

        q0 = []
        u0 = []
        self.constant_force_reservoir = False

        for p in properties:
            setattr(self, f"_{self.__class__.__name__}__{p}_contr", [])

        for contr in self.contributions:
            contr.t0 = self.t0
            for p in properties:
                # if property is implemented as class function append to property contribution
                # - p in contr.__class__.__dict__: has global class attribute p
                # - callable(getattr(contr, p, None)): p is callable
                if hasattr(contr, p) and callable(getattr(contr, p)):
                    getattr(self, f"_{self.__class__.__name__}__{p}_contr").append(
                        contr
                    )

            # if contribution has position degrees of freedom address position coordinates
            if hasattr(contr, "nq"):
                contr.my_qDOF = np.arange(0, contr.nq) + self.nq
                contr.qDOF = contr.my_qDOF.copy()
                self.nq += contr.nq
                q0.extend(contr.q0.tolist())

            # if contribution has velocity degrees of freedom address velocity coordinates
            if hasattr(contr, "nu"):
                contr.my_uDOF = np.arange(0, contr.nu) + self.nu
                contr.uDOF = contr.my_uDOF.copy()
                self.nu += contr.nu
                u0.extend(contr.u0.tolist())

            # if contribution has compliance contribution
            if hasattr(contr, "nla_c"):
                contr.la_cDOF = np.arange(0, contr.nla_c) + self.nla_c
                self.nla_c += contr.nla_c

            # if contribution has constraints on position level address constraint coordinates
            if hasattr(contr, "nla_g"):
                contr.la_gDOF = np.arange(0, contr.nla_g) + self.nla_g
                self.nla_g += contr.nla_g

            # if contribution has stabilization conditions for the kinematic equation
            if hasattr(contr, "nla_S"):
                contr.la_SDOF = np.arange(0, contr.nla_S) + self.nla_S
                self.nla_S += contr.nla_S

        # call assembler callback: call methods that require first an assembly of the system
        self.assembler_callback()

        # compute consisten initial conditions
        self.q0 = np.array(q0)
        self.u0 = np.array(u0)

        # compute constant system parts
        # - parts of the mass matrix
        coo = CooMatrix((self.nu, self.nu))
        if self.__M_contr:
            I_constant_mass_matrix = np.array(
                [
                    (
                        contr.constant_mass_matrix
                        if hasattr(contr, "constant_mass_matrix")
                        else False
                    )
                    for contr in self.__M_contr
                ]
            )
            self.I_M = ~I_constant_mass_matrix
            self.__M_contr = np.array(self.__M_contr)
            for contr in self.__M_contr[I_constant_mass_matrix]:
                coo[contr.uDOF, contr.uDOF] = contr.M(self.t0, self.q0[contr.qDOF])
        self.constant_mass_matrix = np.all(I_constant_mass_matrix)
        self._M0 = coo

        # - compliance matrix
        coo = CooMatrix((self.nla_c, self.nla_c))
        for contr in self.__c_contr:
            coo[contr.la_cDOF, contr.la_cDOF] = contr.c_la_c()
        self._c_la_c0 = coo.tocoo()

        # compute consistent initial conditions
        (
            self.t0,
            self.q0,
            self.u0,
            self.q_dot0,
            self.u_dot0,
            self.la_g0,
            self.la_c0,
        ) = consistent_initial_conditions(self, *args, **kwargs)

    def assembler_callback(self):
        for contr in self.__assembler_callback_contr:
            contr.assembler_callback()

    #####################
    # kinematic equations
    #####################
    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq, dtype=float)
        for contr in self.__q_dot_contr:
            q_dot[contr.my_qDOF] = contr.q_dot(t, q[contr.qDOF], u[contr.uDOF])
        return q_dot

    def q_dot_q(self, t, q, u, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nq, self.nq))
        for i, contr in enumerate(self.__q_dot_q_contr):
            coo[i, contr.my_qDOF, contr.qDOF] = contr.q_dot_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def q_dot_u(self, t, q, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nq, self.nu))
        for i, contr in enumerate(self.__q_dot_u_contr):
            coo[i, contr.my_qDOF, contr.uDOF] = contr.q_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def step_callback(self, t, q, u):
        for contr in self.__step_callback_contr:
            q[contr.qDOF], u[contr.uDOF] = contr.step_callback(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return q, u

    #####################
    # equations of motion
    #####################
    def M(self, t, q, format="coo", coo=None):
        if self.constant_mass_matrix:
            if coo is None:
                coo = CooMatrix((self.nu, self.nu))
                coo[:, :] = self._M0
            for i, contr in enumerate(
                self.__M_contr[self.I_M]
            ):  # only loop over variable mass parts
                coo[i, contr.uDOF, contr.uDOF] = contr.M(t, q[contr.qDOF])
            return coo.asformat(format)
        else:
            return self._M0.asformat(format)

    def Mu_q(self, t, q, u, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nq))
        for i, contr in enumerate(self.__Mu_q_contr):
            coo[i, contr.uDOF, contr.qDOF] = contr.Mu_q(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    def h(self, t, q, u):
        h = np.zeros(self.nu, dtype=float)
        for contr in self.__h_contr:
            np.add.at(h, contr.uDOF, contr.h(t, q[contr.qDOF], u[contr.uDOF]))
            # maybe faster to sum up contributions for the same uDOF first and then add to h
            # uDOF, inv = np.unique(contr.uDOF, return_inverse=True)
            # sums = np.bincount(inv, weights=contr.h(t, q[contr.qDOF], u[contr.uDOF]))
            # h[uDOF] += sums
        return h

    def h_q(self, t, q, u, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nq))
        for i, contr in enumerate(self.__h_q_contr):
            coo[i, contr.uDOF, contr.qDOF] = contr.h_q(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    def h_u(self, t, q, u, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nu))
        for i, contr in enumerate(self.__h_u_contr):
            coo[i, contr.uDOF, contr.uDOF] = contr.h_u(t, q[contr.qDOF], u[contr.uDOF])
        return coo.asformat(format)

    ############
    # compliance
    ############
    def la_c(self, t, q, u):
        la_c = np.zeros(self.nla_c, dtype=float)
        for contr in self.__c_contr:
            la_c[contr.la_cDOF] = contr.la_c(t, q[contr.qDOF], u[contr.uDOF])
        return la_c

    def c(self, t, q, u, la_c):
        c = np.zeros(self.nla_c, dtype=float)
        for contr in self.__c_contr:
            c[contr.la_cDOF] = contr.c(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return c

    def c_q(self, t, q, u, la_c, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nla_c, self.nq))
        for i, contr in enumerate(self.__c_q_contr):
            coo[i, contr.la_cDOF, contr.qDOF] = contr.c_q(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    def c_u(self, t, q, u, la_c, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nla_c, self.nu))
        for i, contr in enumerate(self.__c_u_contr):
            coo[i, contr.la_cDOF, contr.uDOF] = contr.c_u(
                t, q[contr.qDOF], u[contr.uDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    def c_la_c(self, format="coo"):
        return self._c_la_c0.asformat(format)

    def W_c(self, t, q, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nla_c))
        for i, contr in enumerate(self.__c_contr):
            coo[i, contr.uDOF, contr.la_cDOF] = contr.W_c(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_c_q(self, t, q, la_c, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nq))
        for i, contr in enumerate(self.__c_q_contr):
            coo[i, contr.uDOF, contr.qDOF] = contr.Wla_c_q(
                t, q[contr.qDOF], la_c[contr.la_cDOF]
            )
        return coo.asformat(format)

    #########################################
    # bilateral constraints on position level
    #########################################
    def g(self, t, q):
        g = np.zeros(self.nla_g, dtype=q.dtype)
        for contr in self.__g_contr:
            g[contr.la_gDOF] = contr.g(t, q[contr.qDOF])
        return g

    def g_q(self, t, q, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nla_g, self.nq))
        for i, contr in enumerate(self.__g_contr):
            coo[i, contr.la_gDOF, contr.qDOF] = contr.g_q(t, q[contr.qDOF])
        return coo.asformat(format)

    def W_g(self, t, q, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nla_g))
        for i, contr in enumerate(self.__g_contr):
            coo[i, contr.uDOF, contr.la_gDOF] = contr.W_g(t, q[contr.qDOF])
        return coo.asformat(format)

    def Wla_g_q(self, t, q, la_g, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nu, self.nq))
        for i, contr in enumerate(self.__g_contr):
            coo[i, contr.uDOF, contr.qDOF] = contr.Wla_g_q(
                t, q[contr.qDOF], la_g[contr.la_gDOF]
            )
        return coo.asformat(format)

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g, dtype=float)
        for contr in self.__g_contr:
            g_dot[contr.la_gDOF] = contr.g_dot(t, q[contr.qDOF], u[contr.uDOF])
        return g_dot

    def g_dot_u(self, t, q, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nla_g, self.nu))
        for i, contr in enumerate(self.__g_contr):
            coo[i, contr.la_gDOF, contr.uDOF] = contr.g_dot_u(t, q[contr.qDOF])
        return coo.asformat(format)

    def g_dot_q(self, t, q, u, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nla_g, self.nq))
        for i, contr in enumerate(self.__g_contr):
            coo[i, contr.la_gDOF, contr.qDOF] = contr.g_dot_q(
                t, q[contr.qDOF], u[contr.uDOF]
            )
        return coo.asformat(format)

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g, dtype=float)
        for contr in self.__g_contr:
            g_ddot[contr.la_gDOF] = contr.g_ddot(
                t, q[contr.qDOF], u[contr.uDOF], u_dot[contr.uDOF]
            )
        return g_ddot

    # TODO: Assemble zeta_g for efficency
    def zeta_g(self, t, q, u):
        return self.g_ddot(t, q, u, np.zeros(self.nu))

    #####################################################
    # stabilization conditions for the kinematic equation
    #####################################################
    def g_S(self, t, q):
        g_S = np.zeros(self.nla_S, dtype=q.dtype)
        for contr in self.__g_S_contr:
            g_S[contr.la_SDOF] = contr.g_S(t, q[contr.qDOF])
        return g_S

    def g_S_q(self, t, q, format="coo", coo=None):
        if coo is None:
            coo = CooMatrix((self.nla_S, self.nq))
        for i, contr in enumerate(self.__g_S_contr):
            coo[i, contr.la_SDOF, contr.qDOF] = contr.g_S_q(t, q[contr.qDOF])
        return coo.asformat(format)
