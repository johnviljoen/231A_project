"""
Minimal MPC example on the quad, I have removed:

- variable prediction timesteps
- fancy prediction timestep RK4 integrator
- unecessary methods
"""

import casadi as ca
import numpy as np
from typing import Dict
from tqdm import tqdm

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
        self.X = self.opti.variable(n, N+1) # first state plays no role
        self.U = self.opti.variable(m, N+1) # final input plays no role

        # define cost function weights
        #           {x,y,z,q0,q1,q2,q3,xd,yd,zd,p,q,r,wM1,wM2,wM3,wM4}
        Q = np.diag([1,1,1,0, 0, 0, 0, 1, 1, 1, 1,1,1,0,  0,  0,  0  ])
        #           {wM1d, wM2d, wM3d, wM4d}
        R = np.diag([1,    1,    1,    1])

        # apply initial condition constraints
        self.init = self.opti.parameter(n,1) # opti parameters are non-optimized variables that we can change at runtime
        self.opti.subject_to(self.X[:,0] == self.init)

        # apply dynamics constraints with euler integrator
        for k in range(N):
            self.opti.subject_to(self.X[:,k+1] == self.X[:,k] + f(self.X[:,k], self.U[:,k], p) * Ts)

        # apply state constraints
        for k in range(N+1):
            self.opti.subject_to(self.X[:,k] < p["state_ub"])
            self.opti.subject_to(self.X[:,k] > p["state_lb"])

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))

        # apply cylinder constraint - can you turn the cylinder parameterization into a opti.parameter to enable runtime changes to it?
        r = 0.5 # radius
        xc = 1. # cylinder x center position
        yc = 1. # cylinder y center position
        is_in_cylinder = lambda X, Y, m: r ** 2 * m <= (X - xc)**2 + (Y - yc)**2
        for k in range(N):
            current_time = k*Ts
            multiplier = 1 + current_time * 0.1
            current_x, current_y = self.X[0,k+1], self.X[1,k+1]
            self.opti.subject_to(is_in_cylinder(current_x, current_y, multiplier))

        # define the cost function - can you parameterize the reference to be a ca.MX so that we can change our desired destination at runtime?
        def J(state, reference, input):
            state_error = reference - state
            cost = ca.MX(0)
            # lets get cost per timestep:
            for k in range(N+1):
                timestep_input = input[:,k]
                timestep_state_error = state_error[:,k]
                cost += (timestep_state_error.T @ Q @ timestep_state_error + timestep_input.T @ R @ timestep_input)
            return cost
        
        # apply the static reference cost to the opti container
        ref = np.array([2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.opti.minimize(J(self.X, ref, self.U))

        # tell the opti container we want to use IPOPT to optimize, and define settings for the solver
        opts = {
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.tol': 1e-6,
        } # silence!
        self.opti.solver('ipopt', opts)

        # lets do a solve in the __init__ function to test functionality - the only thing that we have parameterized to change at runtime is the initial condition
        state0 = p["default_init_state_np"]
        self.opti.set_value(self.init, state0)

        # perform the solve
        sol = self.opti.solve()

        # extract the answer and save to an attribute we will later use to warm start the optimization variables for efficiency
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)

    def __call__(self, x):

        # assign the new initial condition to the runtime changeable parameter
        self.opti.set_value(self.init, x)

        # warm starting based off of previous solution
        old_x_sol = self.x_sol[:,2:] # ignore old start and first step (this step start)
        x_warm_start = np.hstack([old_x_sol, old_x_sol[:,-1:]]) # stack final solution onto the end again for next warm start
        old_u_sol = self.u_sol[:,1:] # ignore previous solution
        u_warm_start = np.hstack([old_u_sol, old_u_sol[:,-1:]]) # stack final u solution onto the end again for next warm start

        self.opti.set_initial(self.X[:,1:], x_warm_start)
        self.opti.set_initial(self.U[:,:], u_warm_start) 

        # perform the solve
        sol = self.opti.solve()

        # extract the answer
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)

        # return first input to be used
        return self.u_sol[:,0]
    
    def get_predictions(self):
        return self.opti.value(self.X), self.opti.value(self.U)
    
if __name__ == "__main__":

    from dynamics import get_quad_params, state_dot
    from utils.quad import Animator

    quad_params = get_quad_params()

    x = quad_params["default_init_state_np"]
    Ti, Tf, Ts = 0.0, 5.0, 0.1
    N = 30

    mpc = MPC(N=N,Ts=Ts,f=state_dot.casadi,p=quad_params)
    test = mpc(x)

    ctrl_pred_x = []
    memory = {'state': [x], 'cmd': [np.zeros(4)]}
    true_times = np.arange(Ti, Tf, Ts)
    for t in tqdm(true_times):

        u = mpc(x)
        x += state_dot.numpy(x, u, quad_params) * Ts

        print(f'u: {u}')
        print(f'x: {x}')

        ctrl_predictions = mpc.get_predictions()
        ctrl_pred_x.append(ctrl_predictions[0])

        memory['state'].append(np.copy(x))
        memory['cmd'].append(u)

    memory['state'] = np.vstack(memory['state'])
    memory['cmd'] = np.vstack(memory['cmd'])

    ctrl_pred_x = np.stack(ctrl_pred_x)

    animator = Animator(memory['state'], true_times, memory['state'], max_frames=500, save_path='data', state_prediction=ctrl_pred_x, drawCylinder=True)
    animator.animate()

    print('fin')

