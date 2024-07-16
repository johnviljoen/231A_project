"""
Minimal MPC example on the quad, I have removed:

- variable prediction timesteps
- fancy prediction timestep RK4 integrator
- unecessary methods
"""

import casadi as ca
import numpy as np
from typing import Dict

class MPC:
    def __init__(
            self,
            N: int,         # horizon length
            Ts: float,      # timestep
            f: callable,    # xdot = f(x,u)
            p: Dict,        # quad parameters dictionary
        ) -> None:

        n = 17
        m = 4
        
        # create optimizer container and define its optimization variables
        self.opti = ca.Opti()
        # opti variables are optimized variables, we can only change their initial pre-optimized value at runtime
        self.X = self.opti.variable(self.n, N) # first state plays no role
        self.U = self.opti.variable(self.m, N) # final input plays no role

        # define cost function weights
        #           {x,y,z,q0,q1,q2,q3,xd,yd,zd,p,q,r,wM1,wM2,wM3,wM4}
        Q = np.diag([1,1,1,0, 0, 0, 0, 1, 1, 1, 1,1,1,0,  0,  0,  0  ])
        #           {wM1d, wM2d, wM3d, wM4d}
        R = np.diag([1,    1,    1,    1])

        # apply initial condition constraints
        state0 = p["default_init_state_np"]
        self.init = self.opti.parameter(n,1) # opti parameters are non-optimized variables that we can change at runtime
        self.opti.set_value(self.init, state0)
        self.opti.subject_to(self.X[:,0] == self.init)

        # apply dynamics constraints with euler integrator
        for k in range(N):
            self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + f(self.X[:,k], self.U[:,k], p) * Ts)

        # apply state constraints
        for k in range(self.N):
            self.opti.subject_to(self.X[:,k] < p["state_ub"])
            self.opti.subject_to(self.X[:,k] > p["state_lb"])

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))
