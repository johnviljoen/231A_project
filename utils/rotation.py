import torch
import numpy as np
import casadi as ca

class euler_to_quaternion:

    @staticmethod
    def pytorch_batched(euler_angles: torch.Tensor):
        # euler_angles is a tensor of shape (batch_size, 3)
        # where each row is (roll, pitch, yaw)
        rolls = euler_angles[:, 0]
        pitches = euler_angles[:, 1]
        yaws = euler_angles[:, 2]

        cys = torch.cos(yaws * 0.5)
        sys = torch.sin(yaws * 0.5)
        cps = torch.cos(pitches * 0.5)
        sps = torch.sin(pitches * 0.5)
        crs = torch.cos(rolls * 0.5)
        srs = torch.sin(rolls * 0.5)

        q0s = crs * cps * cys + srs * sps * sys
        q1s = srs * cps * cys - crs * sps * sys
        q2s = crs * sps * cys + srs * cps * sys
        q3s = crs * cps * sys - srs * sps * cys

        quaternions = torch.stack((q0s, q1s, q2s, q3s), dim=1)
        # Normalize each quaternion
        norms = torch.norm(quaternions, p=2, dim=1, keepdim=True)
        quaternions_normalized = quaternions / norms
        
        return quaternions_normalized
    
    @staticmethod
    def pytorch(euler_angles: np.ndarray):
        raise NotImplementedError
    
    @staticmethod
    def casadi(euler_angles: ca.MX):
        raise NotImplementedError

    @staticmethod
    def numpy_batched(euler_angles: np.ndarray):
        raise NotImplementedError
    
    @staticmethod
    def numpy(euler_angles: np.ndarray):
        raise NotImplementedError

class quaternion_to_euler:

    @staticmethod
    def pytorch_batched(quaternions: torch.Tensor):
        # quaternions is a tensor of shape (batch_size, 4)
        # where each row is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        
        # Extracting the values from each quaternion
        q0 = quaternions[:, 0]
        q1 = quaternions[:, 1]
        q2 = quaternions[:, 2]
        q3 = quaternions[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1.pow(2) + q2.pow(2))
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2.pow(2) + q3.pow(2))
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.vstack([roll, pitch, yaw]).T
    
    @staticmethod
    def pytorch(quaternion: np.ndarray):
        # quaternion is a tensor of shape (4,)
        # where the array is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternion = quaternion / quaternion.norm()
        
        # Extracting the values from the quaternion
        q0, q1, q2, q3 = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = torch.asin(sinp) if torch.abs(sinp) < 1 else torch.sign(sinp) * torch.pi / 2

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.tensor([roll, pitch, yaw])
    
    @staticmethod
    def casadi(quaternion: None):
        raise NotImplementedError

    @staticmethod
    def numpy_batched(quaternions: np.ndarray):
        # quaternions is a numpy array of shape (batch_size, 4)
        # where each row is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
        
        # Extracting the values from each quaternion
        q0 = quaternions[:, 0]
        q1 = quaternions[:, 1]
        q2 = quaternions[:, 2]
        q3 = quaternions[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = np.where(
            np.abs(sinp) >= 1,
            np.sign(sinp) * np.pi / 2,
            np.arcsin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.vstack([roll, pitch, yaw]).T
    
    @staticmethod
    def numpy(quaternion: np.ndarray):
        # quaternion is a numpy array of shape (4,)
        # where the array is [q0, q1, q2, q3] with q0 being the scalar part

        # Normalize the quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)
        
        # Extracting the values from the quaternion
        q0, q1, q2, q3 = quaternion

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        pitch = np.arcsin(sinp) if np.abs(sinp) < 1 else np.sign(sinp) * np.pi / 2

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

class euler_to_rot_matrix:

    @staticmethod
    def pytorch_batched(roll, pitch, yaw):
        # Assumes the angles are in radians and the order is 'XYZ'
        cos = torch.cos
        sin = torch.sin

        # Pre-calculate cosines and sines
        cos_roll = cos(roll)
        sin_roll = sin(roll)
        cos_pitch = cos(pitch)
        sin_pitch = sin(pitch)
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)

        # Calculate rotation matrix components
        R_x = torch.stack([
            torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll),
            torch.zeros_like(roll), cos_roll, -sin_roll,
            torch.zeros_like(roll), sin_roll, cos_roll
        ], dim=1).reshape(-1, 3, 3)

        R_y = torch.stack([
            cos_pitch, torch.zeros_like(pitch), sin_pitch,
            torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
            -sin_pitch, torch.zeros_like(pitch), cos_pitch
        ], dim=1).reshape(-1, 3, 3)

        R_z = torch.stack([
            cos_yaw, -sin_yaw, torch.zeros_like(yaw),
            sin_yaw, cos_yaw, torch.zeros_like(yaw),
            torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
        ], dim=1).reshape(-1, 3, 3)

        # The final rotation matrix combines rotations around all axes
        R = torch.bmm(torch.bmm(R_z, R_y), R_x)

        return R
    
    @staticmethod
    def pytorch(roll, pitch, yaw):
        raise NotImplementedError
    
    @staticmethod
    def casadi(roll, pitch, yaw):
        raise NotImplementedError    
    
    @staticmethod
    def numpy_batched(roll, pitch, yaw):
        raise NotImplementedError    
    
    @staticmethod
    def numpy(roll, pitch, yaw):
        # Assumes the angles are in radians and the order is 'XYZ'
        cos = np.cos
        sin = np.sin

        # Pre-calculate cosines and sines
        cos_roll = cos(roll)
        sin_roll = sin(roll)
        cos_pitch = cos(pitch)
        sin_pitch = sin(pitch)
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)

        # Calculate rotation matrix components
        R_x = np.array([
            [1, 0, 0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll, cos_roll]
        ])

        R_y = np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])

        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # The final rotation matrix combines rotations around all axes
        R = R_z @ R_y @ R_x

        return R

class quaternion_derivative:
    # Functions to calculate the quaternion derivative given quaternion and angular velocity

    @staticmethod
    def pytorch_batched(q: torch.Tensor, omega: torch.Tensor):
        # Assuming q is of shape [batch, 4] with q = [q0, q1, q2, q3]
        # omega is of shape [batch, 3] with omega = [p, q, r]
        
        # Quaternion multiplication matrix
        Q_mat = torch.cat([
            -q[:, 1:2], -q[:, 2:3], -q[:, 3:4],
            q[:, 0:1], -q[:, 3:4],  q[:, 2:3],
            q[:, 3:4],  q[:, 0:1], -q[:, 1:2],
            -q[:, 2:3],  q[:, 1:2],  q[:, 0:1]
        ], dim=1).view(-1, 4, 3)

        # Multiply by the angular velocity
        q_dot = 0.5 * torch.bmm(Q_mat, omega.unsqueeze(-1)).squeeze(-1)
        return q_dot
    
    @staticmethod
    def pytorch(q, omega):
        raise NotImplementedError
    
    @staticmethod
    def casadi(q, omega):
        # Assuming q is a column vector [4, 1] with q = [q0, q1, q2, q3]
        # omega is a column vector [3, 1] with omega = [p, q, r]
        
        # Quaternion multiplication matrix
        Q_mat = ca.vertcat(
            ca.horzcat(-q[1], -q[2], -q[3]),
            ca.horzcat( q[0], -q[3],  q[2]),
            ca.horzcat( q[3],  q[0], -q[1]),
            ca.horzcat(-q[2],  q[1],  q[0])
        )

        # Multiply by the angular velocity to get the quaternion derivative
        q_dot = 0.5 * ca.mtimes(Q_mat, omega)

        return q_dot 
    
    @staticmethod
    def numpy_batched(q, omega):
        raise NotImplementedError  

class quaternion_conjugate:

    @staticmethod
    def casadi(q):
        return ca.vertcat(q[0,:], -q[1,:], -q[2,:], -q[3,:])

class quaternion_multiply:

    @staticmethod
    def casadi(q1, q2):
        # Quaternion multiplication (Hamilton product)
        w1, x1, y1, z1 = q1[0,:], q1[1,:], q1[2,:], q1[3,:]
        w2, x2, y2, z2 = q2[0,:], q2[1,:], q2[2,:], q2[3,:]
        return ca.vertcat(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

class quaternion_error:

    @staticmethod
    def casadi(q, q_desired):
        # Calculate the quaternion error
        q_desired_conjugate = quaternion_conjugate.casadi(q_desired)
        return quaternion_multiply.casadi(q, q_desired_conjugate)

class quaternion_to_dcm:

    @staticmethod
    def numpy(q):
        dcm = np.zeros([3,3])

        dcm[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
        dcm[0,1] = 2.0*(q[1]*q[2] - q[0]*q[3])
        dcm[0,2] = 2.0*(q[1]*q[3] + q[0]*q[2])
        dcm[1,0] = 2.0*(q[1]*q[2] + q[0]*q[3])
        dcm[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
        dcm[1,2] = 2.0*(q[2]*q[3] - q[0]*q[1])
        dcm[2,0] = 2.0*(q[1]*q[3] - q[0]*q[2])
        dcm[2,1] = 2.0*(q[2]*q[3] + q[0]*q[1])
        dcm[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

        return dcm


if __name__ == "__main__":
    
    import utils.pytorch as ptu

    # Example usage:
    # Define a batch of Euler angles (roll, pitch, yaw) in radians
    euler_angles_batch = ptu.tensor([
        [30, 45, 60],
        [10, 20, 30],
        # ... more angles
    ]) * torch.pi/180  # Convert degrees to radians

    # Convert to quaternions
    quaternions_batch = euler_to_quaternion.pytorch_batched(euler_angles_batch)

    # Print the quaternions
    print("Quaternions:\n", quaternions_batch)

    # Convert to Euler angles
    eul_batch = quaternion_to_euler.pytorch_batched(quaternions_batch)

    # Print the Euler angles in radians
    print(f"eul_batch: {eul_batch * 180/torch.pi}")