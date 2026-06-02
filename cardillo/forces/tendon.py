from cardillo.interactions import nPointInteraction


class TendonForce(nPointInteraction):
    def __init__(
        self,
        subsystem_list,
        connectivity,
        xi_list=None,
        B_r_CP_list=None,
        name="tendon",
    ) -> None:
        self.name = name
        super().__init__(subsystem_list, connectivity, xi_list, B_r_CP_list)

    def h(self, t, q, u):
        return -self.la(t) * self.W_l(t, q)

    def h_q(self, t, q, u):
        return -self.la(t) * self.W_l_q(t, q)

    def la(self, t):
        return 0.0

    def set_force(self, force):
        self.la = force if callable(force) else lambda t: force
