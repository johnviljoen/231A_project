import casadi as ca
import numpy as np

def state_dot_np(state, cmd, params):
    q0 =    state[3]
    q1 =    state[4]
    q2 =    state[5]
    q3 =    state[6]
    xdot =  state[7]
    ydot =  state[8]
    zdot =  state[9]
    p =     state[10]
    q =     state[11]
    r =     state[12]
    wM1 =   state[13]
    wM2 =   state[14]
    wM3 =   state[15]
    wM4 =   state[16]

    wMotor = np.stack([wM1, wM2, wM3, wM4])
    wMotor = np.clip(wMotor, params["minWmotor"], params["maxWmotor"])
    thrust = params["kTh"] * wMotor ** 2
    torque = params["kTo"] * wMotor ** 2

    # Wind Model
    # ---------------------------
    velW, qW1, qW2 = [0]*3

    # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
    # ---------------------------
    DynamicsDot = np.stack(
        [
            xdot,
            ydot,
            zdot,
            -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
            0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
            0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
            -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
            (
                params["Cd"]
                * np.sign(velW * np.cos(qW1) * np.cos(qW2) - xdot)
                * (velW * np.cos(qW1) * np.cos(qW2) - xdot) ** 2
                - 2 * (q0 * q2 + q1 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
            )
            / params["mB"],
            (
                params["Cd"]
                * np.sign(velW * np.sin(qW1) * np.cos(qW2) - ydot)
                * (velW * np.sin(qW1) * np.cos(qW2) - ydot) ** 2
                + 2 * (q0 * q1 - q2 * q3) * (thrust[0] + thrust[1] + thrust[2] + thrust[3])
            )
            / params["mB"],
            (
                -params["Cd"] * np.sign(velW * np.sin(qW2) + zdot) * (velW * np.sin(qW2) + zdot) ** 2
                - (thrust[0] + thrust[1] + thrust[2] + thrust[3])
                * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                + params["g"] * params["mB"]
            )
            / params["mB"],
            (
                (params["IB"][1,1] - params["IB"][2,2]) * q * r
                - params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                + (thrust[0] - thrust[1] - thrust[2] + thrust[3]) * params["dym"]
            )
            / params["IB"][0,0],  # uP activates or deactivates the use of gyroscopic precession.
            (
                (params["IB"][2,2] - params["IB"][0,0]) * p * r
                + params["usePrecession"] * params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                + (thrust[0] + thrust[1] - thrust[2] - thrust[3]) * params["dxm"]
            )
            / params["IB"][1,1],  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
            ((params["IB"][0,0] - params["IB"][1,1]) * p * q - torque[0] + torque[1] - torque[2] + torque[3]) / params["IB"][2,2],
        ]
    )

    # we must limit the actuator rotational rate
    omega_check_upper = state[13:] > params["maxWmotor"]
    omega_check_lower = state[13:] < params["minWmotor"]
    ActuatorsDot = cmd/params["IRzz"]
    ActuatorsDot[(omega_check_upper) | (omega_check_lower)] = 0

    # State Derivative Vector
    # ---------------------------
    if state.shape[0] == 1:
        return np.hstack([DynamicsDot.squeeze(), ActuatorsDot.squeeze()])
    else:
        return np.hstack([DynamicsDot.T, ActuatorsDot])

# casadi quadcopter for MPC
class QuadcopterCA:
    def __init__(
            self,
            params
        ) -> None:
        
        self.params = params

    def state_dot_1d(self, state: ca.MX, cmd: ca.MX):

        # Import params to numpy for CasADI
        # ---------------------------
        IB = self.params["IB"]
        IBxx = ptu.to_numpy(IB[0, 0])
        IByy = ptu.to_numpy(IB[1, 1])
        IBzz = ptu.to_numpy(IB[2, 2])

        # Unpack state tensor for readability
        # ---------------------------
        q0 =    state[3]
        q1 =    state[4]
        q2 =    state[5]
        q3 =    state[6]
        xdot =  state[7]
        ydot =  state[8]
        zdot =  state[9]
        p =     state[10]
        q =     state[11]
        r =     state[12]
        wM1 =   state[13]
        wM2 =   state[14]
        wM3 =   state[15]
        wM4 =   state[16]

        # a tiny bit more readable
        ThrM1 = self.params["kTh"] * wM1 ** 2
        ThrM2 = self.params["kTh"] * wM2 ** 2
        ThrM3 = self.params["kTh"] * wM3 ** 2
        ThrM4 = self.params["kTh"] * wM4 ** 2
        TorM1 = self.params["kTo"] * wM1 ** 2
        TorM2 = self.params["kTo"] * wM2 ** 2
        TorM3 = self.params["kTo"] * wM3 ** 2
        TorM4 = self.params["kTo"] * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW, qW1, qW2 = 0, 0, 0

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = ca.vertcat(
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    self.params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / self.params["mB"],
                (
                    self.params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / self.params["mB"],
                (
                    -self.params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + self.params["g"] * self.params["mB"]
                )
                / self.params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * self.params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * self.params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0]/self.params["IRzz"], cmd[1]/self.params["IRzz"], cmd[2]/self.params["IRzz"], cmd[3]/self.params["IRzz"]
        )

        if DynamicsDot.shape[1] == 17:
            print('fin')

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot

    def state_dot(self, state: ca.MX, cmd: ca.MX):

        # Import params to numpy for CasADI
        # ---------------------------
        IB = self.params["IB"]
        IBxx = ptu.to_numpy(IB[0, 0])
        IByy = ptu.to_numpy(IB[1, 1])
        IBzz = ptu.to_numpy(IB[2, 2])

        # Unpack state tensor for readability
        # ---------------------------
        q0 =    state[3,:]
        q1 =    state[4,:]
        q2 =    state[5,:]
        q3 =    state[6,:]
        xdot =  state[7,:]
        ydot =  state[8,:]
        zdot =  state[9,:]
        p =     state[10,:]
        q =     state[11,:]
        r =     state[12,:]
        wM1 =   state[13,:]
        wM2 =   state[14,:]
        wM3 =   state[15,:]
        wM4 =   state[16,:]

        # a tiny bit more readable
        ThrM1 = self.params["kTh"] * wM1 ** 2
        ThrM2 = self.params["kTh"] * wM2 ** 2
        ThrM3 = self.params["kTh"] * wM3 ** 2
        ThrM4 = self.params["kTh"] * wM4 ** 2
        TorM1 = self.params["kTo"] * wM1 ** 2
        TorM2 = self.params["kTo"] * wM2 ** 2
        TorM3 = self.params["kTo"] * wM3 ** 2
        TorM4 = self.params["kTo"] * wM4 ** 2

        # Wind Model (zero in expectation)
        # ---------------------------
        velW, qW1, qW2 = 0, 0, 0

        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = ca.vertcat(
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (
                    self.params["Cd"]
                    * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                    * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                    - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / self.params["mB"],
                (
                    self.params["Cd"]
                    * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                    * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                    + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                )
                / self.params["mB"],
                (
                    -self.params["Cd"] * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                    - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                    * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                    + self.params["g"] * self.params["mB"]
                )
                / self.params["mB"],
                (
                    (IByy - IBzz) * q * r
                    - self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * q
                    + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * self.params["dym"]
                )
                / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                (
                    (IBzz - IBxx) * p * r
                    + self.params["usePrecession"] * self.params["IRzz"] * (wM1 - wM2 + wM3 - wM4) * p
                    + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * self.params["dxm"]
                )
                / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                cmd[0,:]/self.params["IRzz"], cmd[1,:]/self.params["IRzz"], cmd[2,:]/self.params["IRzz"], cmd[3,:]/self.params["IRzz"]
        )

        # State Derivative Vector
        # ---------------------------
        return DynamicsDot
    
