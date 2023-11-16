import os
from datetime import datetime
import torch
import numpy as np
import neuromancer as nm
from neuromancer.dynamics import ode, integrators

import utils.pytorch as ptu
import utils.callback



class DatasetGenerator:
    def __init__(
            self,
            task,           # 'wp_p2p', 'wp_traj', 'fig8'
            batch_size,     # 5000
            minibatch_size, # 10
            nstep,          # 100
            Ts,             # 0.001
        ):

        self.task = task
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.nstep = nstep
        self.Ts = Ts

        # parameters which will not change:
        self.radius = 0.5
        self.shuffle_dataloaders = False
        self.x_range = 3.0
        self.r_range = 3.0
        self.cyl_range = 3.0
        self.nx = 6

    # Obstacle Avoidance Methods:
    # ---------------------------

    def is_inside_cylinder(self, x, y, cx, cy):
        """
        Check if a point (x,y) is inside a cylinder with center (cx, cy) and given radius.
        """
        distance = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance < self.radius
    
    def get_wp_p2p_dataset(self):
        """
        here we generate random samples in a cylinder around the desired end point, instead
        of the line in dataset2
        """
        X = []
        R = []
        Cyl = []

        n = self.batch_size  # Replace with your actual number of points

        R = torch.cat([torch.cat([torch.tensor([[[2., 0, 2, 0, 1, 0]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Cyl = torch.cat([torch.cat([torch.tensor([[[1.,1]]])]*(self.nstep+1), dim=1)]*n, dim=0)
        Idx = torch.cat([torch.vstack([torch.tensor([0.0])]).unsqueeze(1)]*n, dim=0)
        M = torch.cat([torch.ones([1, 1, 1])]*n, dim=0)

        cylinder_center = torch.tensor([2.0, 2.0])  # Center of the cylinder
        cylinder_radius = torch.sqrt(torch.tensor(2**2 + 2**2))  # Radius of the cylinder to pass through the origin
        z_lower = -1  # Lower bound for Z values
        z_upper = 1  # Upper bound for Z values
        
        # std was 1 previously
        angles = torch.normal(mean=torch.tensor(1.25*torch.pi), std=torch.tensor(2), size=(n,))# (2 * torch.pi * torch.randn(n)) + 0.75 * torch.pi

        # Convert polar to Cartesian coordinates
        x = cylinder_radius * torch.cos(angles) + cylinder_center[0]
        y = cylinder_radius * torch.sin(angles) + cylinder_center[1]

        # Generate uniformly distributed Z values
        z = z_lower + (z_upper - z_lower) * torch.rand(n)

        # Stack x, y, and z to get the final coordinates
        xdot, ydot, zdot = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        X = torch.stack((x, xdot, y, ydot, z, zdot), dim=1).unsqueeze(1)
        
        return {
            'X': X,
            'R': R,
            'Cyl': Cyl,
            'Idx': Idx,
            'M': M
        }

    def validate_dataset(self, dataset):
        X = dataset['X']
        R = dataset['R']
        Cyl = dataset['Cyl']
        
        batch_size = X.shape[0]
        nstep = R.shape[1] - 1

        print("validating dataset...")
        for i in range(batch_size):
            for t in range(nstep + 1):
                # Check initial state.
                if self.is_inside_cylinder(X[i, 0, 0], X[i, 0, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                    return False, f"Initial state at index {i} lies inside the cylinder."
                # Check each reference point.
                if self.is_inside_cylinder(R[i, t, 0], R[i, t, 2], Cyl[i, t, 0], Cyl[i, t, 1]):
                    return False, f"Reference at time {t} for batch index {i} lies inside the cylinder."

        return True, "All points are outside the cylinder."
    
    # Trajectory Reference Methods:
    # -------------------------------

    def fig8(self, t, A=4, B=4, C=4, Z=-5, average_vel=1.0):

        # accelerate or decelerate time based on velocity desired
        t *= average_vel

        # Position
        x = A * np.cos(t)
        y = B * np.sin(2*t) / 2
        z = C * np.sin(t) + Z  # z oscillates around the plane Z

        # Velocities
        xdot = -A * np.sin(t)
        ydot = B * np.cos(2*t)
        zdot = C * np.cos(t)

        return ptu.tensor([x, y, z]).squeeze(), ptu.tensor([xdot, ydot, zdot]).squeeze()
    
    def generate_reference(self, mode='linear', average_velocity=0.5):
        """
        Generate a reference dataset.
        Parameters:
        - nstep: Number of steps
        - nx: Dimensionality of the reference
        - Ts: Time step
        - r_range: Range for random sampling
        - mode: 'linear' for straight line references, 'sinusoidal' for sinusoidal references
        """
        if mode == 'linear':
            start_point = self.r_range * torch.randn(1, 1, self.nx)
            end_point = self.r_range * torch.randn(1, 1, self.nx)

            pos_sample = []
            for dim in range(3):  # Only interpolate the positions (x, y, z)
                interpolated_values = torch.linspace(start_point[0, 0, dim], end_point[0, 0, dim], steps=self.nstep+1)
                pos_sample.append(interpolated_values)

            pos_sample = torch.stack(pos_sample, dim=-1)
            # Calculate the CORRECT velocities for our timestep
            vel_sample = (pos_sample[1:, :] - pos_sample[:-1, :]) / self.Ts

            # For the last velocity, we duplicate the last calculated velocity
            vel_sample = torch.cat([vel_sample, vel_sample[-1:, :]], dim=0)

            return pos_sample, vel_sample


        elif mode == 'sinusoidal':
            # randomise the initial time so as to look across the trajectory without such a long 
            # nstep prediction length.
            t_start = np.random.rand(1) * 15
            times = np.linspace(t_start, t_start + self.nstep * self.Ts, self.nstep + 1)  # Generate time values
            A = (np.random.rand(1) - 0.5) * 2 + 4
            B = (np.random.rand(1) - 0.5) * 2 + 4
            C = (np.random.rand(1) - 0.5) * 2 + 4
            Z = (np.random.rand(1) - 0.5) * 2 - 5
            average_vel = average_velocity # np.random.rand(1) * 2

            pos_sample = []
            vel_sample = []
            paras_sample = []
            T_total = self.Ts * (self.nstep)
            for t in times:
                pos, vel = self.fig8(t=t, A=A, B=B, C=C, Z=Z, average_vel=average_vel)
                paras = ptu.from_numpy(np.hstack([A,B,C,Z]))
                # pos[2] *= -1 # NED
                pos_sample.append(pos)
                paras_sample.append(paras)
                
                # vel_sample.append(vel)

            pos_sample = torch.stack(pos_sample)
            paras_sample = torch.stack(paras_sample)

            vel_sample = (pos_sample[1:, :] - pos_sample[:-1, :]) / self.Ts

            # For the last velocity, we duplicate the last calculated velocity
            vel_sample = torch.cat([vel_sample, vel_sample[-1:, :]], dim=0)

            return pos_sample, vel_sample, paras_sample
            # 
            # vel_sample = torch.stack(vel_sample)


            # print('fin')

    def get_linear_wp_traj_dataset(self):
        X = []
        R = []
        
        # Loop until the desired batch size is reached.
        print(f"generating dataset of batchsize: {self.batch_size}")
        while len(X) < self.batch_size:
            x_sample = self.x_range * torch.randn(1, 1, self.nx)
            x_sample[:,:,0] *= 2.5
            # x_sample[:,:,0] += 2

            pos_sample, vel_sample = self.generate_reference(mode='linear')
            pos_sample *= 2.5
            pos_sample -= 2

            # Rearrange to the desired order {x, xdot, y, ydot, z, zdot}
            r_sample = torch.zeros(1, self.nstep+1, self.nx)
            r_sample[0, :, 0] = pos_sample[:, 0]
            r_sample[0, :, 1] = vel_sample[:, 0]
            r_sample[0, :, 2] = pos_sample[:, 1]
            r_sample[0, :, 3] = vel_sample[:, 1]
            r_sample[0, :, 4] = pos_sample[:, 2]
            r_sample[0, :, 5] = vel_sample[:, 2]
            
            X.append(x_sample)
            R.append(r_sample)
        
        # Convert lists to tensors.
        X = torch.cat(X, dim=0)
        R = torch.cat(R, dim=0)
        
        return {
            'X': X,
            'R': R,
        }
    
    def get_sinusoidal_traj_dataset(self, average_velocity):
        X = []
        R = []
        P = []

        # Loop until the desired batch size is reached.
        print(f"generating dataset of batchsize: {self.batch_size}")
        while len(X) < self.batch_size:
            
            x_sample = (torch.rand(1, 1, self.nx) - 0.5) * 8
            # set velocities to zero
            x_sample[:,:,1::2] *= 0
            # offset Z to be closer to the trajectories
            x_sample[:,:,4] -= 5

            pos_sample, vel_sample, paras_sample = self.generate_reference(mode='sinusoidal', average_velocity=average_velocity)

            # Rearrange to the desired order {x, xdot, y, ydot, z, zdot}
            r_sample = torch.zeros(1, self.nstep+1, self.nx)
            r_sample[0, :, 0] = pos_sample[:, 0]
            r_sample[0, :, 1] = vel_sample[:, 0]
            r_sample[0, :, 2] = pos_sample[:, 1]
            r_sample[0, :, 3] = vel_sample[:, 1]
            r_sample[0, :, 4] = pos_sample[:, 2]
            r_sample[0, :, 5] = vel_sample[:, 2]
            
            X.append(x_sample)
            R.append(r_sample)
            P.append(paras_sample.unsqueeze(0))
        
        # Convert lists to tensors.
        X = torch.cat(X, dim=0)
        R = torch.cat(R, dim=0)
        P = torch.cat(P, dim=0)
        
        return {
            'X': X,
            'R': R,
            'P': P,
        }
    
    # Shared Methods
    # --------------

    # this is unused currently
    def get_random_state(self):
        if self.sample_type == 'normal':
            x_sample = self.x_range * torch.randn(1, 1, self.nx)
            r_sample = torch.cat([self.r_range * torch.randn(1, 1, self.nx)] * (self.nstep + 1), dim=1)
            cyl_sample = torch.cat([self.cyl_range * torch.randn(1, 1, 2)] * (self.nstep + 1), dim=1)
        elif self.sample_type == 'uniform':
            x_sample = 2 * self.x_range * (torch.rand(1, 1, self.nx) - 0.5)
            r_sample = torch.cat([2 * self.r_range * (torch.rand(1, 1, self.nx) - 0.5)] * (self.nstep + 1), dim=1)
            cyl_sample = torch.cat([2 * self.cyl_range * (torch.rand(1, 1, 2) - 0.5)] * (self.nstep + 1), dim=1)
        else:
            raise ValueError(f"invalid sample type passed: {self.sample_type}")
        # reference velocities should be zero here 
        r_sample[:,:,1::2] = 0.
        return x_sample, r_sample, cyl_sample

    def get_dictdatasets(self):
        
        if self.task == 'wp_p2p':
            train_data = nm.dataset.DictDataset(self.get_wp_p2p_dataset(), name='train')
            dev_data = nm.dataset.DictDataset(self.get_wp_p2p_dataset(), name='dev')      
                    
        elif self.task == 'wp_traj':
            train_data = nm.dataset.DictDataset(self.get_linear_wp_traj_dataset(), name='train')
            dev_data = nm.dataset.DictDataset(self.get_linear_wp_traj_dataset(), name='dev')

        elif self.task == 'fig8':
            train_data = nm.dataset.DictDataset(self.get_sinusoidal_traj_dataset(average_velocity=1.0), name='train')
            dev_data = nm.dataset.DictDataset(self.get_sinusoidal_traj_dataset(average_velocity=1.0), name='dev')  
        else:

            raise Exception

        return train_data, dev_data

    def get_loaders(self):

        train_data, dev_data = self.get_dictdatasets()

        if self.task == 'wp_p2p':
            self.validate_dataset(train_data.datadict)
            self.validate_dataset(dev_data.datadict)        

        # put datasets on correct device
        train_data.datadict = {key: value.to(ptu.device) for key, value in train_data.datadict.items()}
        dev_data.datadict = {key: value.to(ptu.device) for key, value in dev_data.datadict.items()}

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.minibatch_size,
                                                collate_fn=train_data.collate_fn, shuffle=self.shuffle_dataloaders)
        dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=self.minibatch_size,
                                                collate_fn=dev_data.collate_fn, shuffle=self.shuffle_dataloaders)
        
        return train_loader, dev_loader
    
class Dynamics(ode.ODESystem):
    def __init__(self, insize, outsize, x_std=0.0) -> None:
        super().__init__(insize=insize, outsize=outsize)

        A = ptu.tensor([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

        B = ptu.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        self.f = lambda x, u: x @ A.T + u @ B.T
        self.in_features = insize
        self.out_features = outsize
        
        # noise definitions
        self.x_std = x_std

    def ode_equations(self, x, u):
        # x = xu[:,0:6]
        # u = xu[:,6:9]
        # add noise if required
        x = x + torch.randn(x.shape, device=ptu.device) * self.x_std
        return self.f(x,u)

def posVel2cyl(state, cyl, radius):
    x = state[:, 0:1]
    y = state[:, 2:3]
    xc = cyl[:, 0:1]
    yc = cyl[:, 1:2]

    dx = x - xc
    dy = y - yc

    # Calculate the Euclidean distance from each point to the center of the cylinder
    distance_to_center = (dx**2 + dy**2) ** 0.5
    
    # Subtract the radius to get the distance to the cylinder surface
    distance_to_cylinder = distance_to_center - radius

    xdot = state[:, 1:2]
    ydot = state[:, 3:4]

    # Normalize the direction vector (from the point to the center of the cylinder)
    dx_normalized = dx / (distance_to_center + 1e-10)  # Adding a small number to prevent division by zero
    dy_normalized = dy / (distance_to_center + 1e-10)

    # Compute the dot product of the normalized direction vector with the velocity vector
    velocity_to_cylinder = dx_normalized * xdot + dy_normalized * ydot

    return distance_to_cylinder, velocity_to_cylinder

def train_wp_p2p(    # recommendations:
    iterations,      # 2
    epochs,          # 15
    batch_size,      # 5000
    minibatch_size,  # 10
    nstep,           # 100
    lr,              # 0.05
    Ts,              # 0.1
    policy_save_path = 'data/',
    media_save_path = 'data/training/',
    ):

    # unchanging parameters:
    radius = 0.5
    Q_con = 1_000_000
    R = 0.1
    Qpos = 5.00
    Qvel = 5.00
    barrier_type = 'softexp'
    barrier_alpha = 0.05
    lr_multiplier = 0.5

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images_path = media_save_path + current_datetime + '/'

    # Check if directory exists, and if not, create it
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # NeuroMANCER System Definition
    # -----------------------------
    nx = 6 # state size
    nu = 3 # input size
    nc = 2 # cylinder distance and velocity

    # Variables:
    r = nm.constraint.variable('R')           # the reference
    u = nm.constraint.variable('U')           # the input
    x = nm.constraint.variable('X')           # the state
    cyl = nm.constraint.variable('Cyl')       # the cylinder center coordinates

    node_list = []

    process_policy_input = lambda x, r, c: torch.hstack([r - x, *posVel2cyl(x, c, radius=radius)])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'Cyl'], ['Obs'], name='preprocess')
    node_list.append(process_policy_input_node)

    policy = nm.modules.blocks.MLP(
        insize=nx + nc, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    ).to(ptu.device)
    policy_node = nm.system.Node(policy, ['Obs'], ['U'], name='policy')
    node_list.append(policy_node)

    dynamics = Dynamics(insize=9, outsize=6)
    integrator = nm.dynamics.integrators.Euler(dynamics, h=torch.tensor(Ts))
    dynamics_node = nm.system.Node(integrator, ['X', 'U'], ['X'], name='dynamics')
    node_list.append(dynamics_node)

    print(f'node list used in cl_system: {node_list}')
    cl_system = nm.system.System(node_list)

    # Dataset Generation Class
    # ------------------------
    dataset = DatasetGenerator(
        task = 'wp_p2p',
        batch_size = batch_size,
        minibatch_size = minibatch_size,
        nstep = nstep,
        Ts = Ts,
    )

    # Problem Setup:
    # --------------
    constraints = []
    cylinder_constraint = Q_con * ((radius**2 <= (x[:,:,0]-cyl[:,:,0])**2 + (x[:,:,2]-cyl[:,:,1])**2)) ^ 2
    constraints.append(cylinder_constraint)

    # Define Loss:
    objectives = []

    action_loss = R * (u == ptu.tensor(0.))^2  # control penalty
    action_loss.name = 'action_loss'
    objectives.append(action_loss)

    pos_loss = Qpos * (x[:,:,::2] == r[:,:,::2])^2
    pos_loss.name = 'pos_loss'
    objectives.append(pos_loss)

    vel_loss = Qvel * (x[:,:,1::2] == r[:,:,1::2])^2
    vel_loss.name = 'vel_loss'
    objectives.append(vel_loss)

    # objectives = [action_loss, pos_loss, vel_loss]
    loss = nm.loss.BarrierLoss(objectives, constraints, barrier=barrier_type, alpha=barrier_alpha)

    # Define the Problem and the Trainer:
    problem = nm.problem.Problem([cl_system], loss, grad_inference=True)
    optimizer = torch.optim.Adagrad(policy.parameters(), lr=lr)

    # Custom Callack Setup
    # --------------------
    callback = utils.callback.WP_Callback(save_dir=current_datetime, media_path=media_save_path, nstep=nstep, nx=nx)

        # Perform the Training
    # --------------------
    for i in range(iterations):
        print(f'training with prediction horizon: {nstep}, lr: {lr}')

        # Get First Datasets
        # ------------------
        dataset.nstep = nstep
        train_loader, dev_loader = dataset.get_loaders()

        trainer = nm.trainer.Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            callback=callback,
            epochs=epochs,
            patience=epochs,
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric='dev_loss',
            warmup=400,
            lr_scheduler=False,
            device=ptu.device
        )

        # Train Over Nsteps
        # -----------------
        cl_system.nsteps = nstep
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

        # Update Parameters for the Next Iteration
        # ----------------------------------------
        lr *= lr_multiplier # 0.2

        # update the prediction horizon
        cl_system.nsteps = nstep
        optimizer.param_groups[0]['lr'] = lr


    callback.animate()
    callback.delete_all_but_last_image()

    # Save the Policy
    # ---------------
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, policy_save_path + f"wp_p2p_policy.pth")

def train_wp_traj(    # recommendations:
    iterations,      # 2
    epochs,          # 15
    batch_size,      # 5000
    minibatch_size,  # 10
    nstep,           # 100
    lr,              # 0.05
    Ts,              # 0.1
    policy_save_path = 'data/',
    media_save_path = 'data/training/',
    ):

    # unchanging parameters:
    R = 0.1
    Qpos = 5.00
    Qvel = 5.00
    barrier_type = 'softexp'
    barrier_alpha = 0.05
    lr_multiplier = 0.5

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images_path = media_save_path + current_datetime + '/'

    # Check if directory exists, and if not, create it
    if not os.path.exists(images_path):
        os.makedirs(images_path)

        # NeuroMANCER System Definition
    # -----------------------------

    nx = 6 # state size
    nu = 3 # input size

    # Variables:
    r = nm.constraint.variable('R')           # the reference
    u = nm.constraint.variable('U')           # the input
    x = nm.constraint.variable('X')           # the state

    # Nodes:
    node_list = []
    
    process_policy_input = lambda x, r: r - x
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R'], ['Obs'], name='preprocess')
    node_list.append(process_policy_input_node)

    policy = nm.modules.blocks.MLP(
        insize=nx, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    ).to(ptu.device)
    policy_node = nm.system.Node(policy, ['Obs'], ['U'], name='policy')
    node_list.append(policy_node)

    dynamics = Dynamics(insize=9, outsize=6)
    integrator = integrators.Euler(dynamics, h=torch.tensor(Ts))
    dynamics_node = nm.system.Node(integrator, ['X', 'U'], ['X'], name='dynamics')
    node_list.append(dynamics_node)

    print(f'node list used in cl_system: {node_list}')
    cl_system = nm.system.System(node_list)

    # Dataset Generation Class
    # ------------------------
    dataset = DatasetGenerator(
        task = 'wp_traj',
        batch_size = batch_size,
        minibatch_size = minibatch_size,
        nstep = nstep,
        Ts = Ts,
    )

    # Define Constraints: - none for this situation
    constraints = []

    # Define Loss:
    objectives = []

    action_loss = R * (u == ptu.tensor(0.))^2  # control penalty
    action_loss.name = 'action_loss'
    objectives.append(action_loss)

    pos_loss = Qpos * (x[:,:,::2] == r[:,:,::2])^2
    pos_loss.name = 'pos_loss'
    objectives.append(pos_loss)

    vel_loss = Qvel * (x[:,:,1::2] == r[:,:,1::2])^2
    vel_loss.name = 'vel_loss'
    objectives.append(vel_loss)

    # objectives = [action_loss, pos_loss, vel_loss]
    loss = nm.loss.BarrierLoss(objectives, constraints, barrier=barrier_type, alpha=barrier_alpha)

    # Define the Problem and the Trainer:
    problem = nm.problem.Problem([cl_system], loss, grad_inference=True)
    optimizer = torch.optim.Adagrad(policy.parameters(), lr=lr)

    # Custom Callack Setup
    # --------------------
    callback = utils.callback.LinTrajCallback(save_dir=current_datetime, media_path=media_save_path, nstep=nstep, nx=nx, Ts=Ts)
    
    # Perform the Training
    # --------------------
    for i in range(iterations):
        print(f'training with prediction horizon: {nstep}, lr: {lr}')

        # Get First Datasets
        # ------------
        dataset.nstep = nstep
        train_loader, dev_loader = dataset.get_loaders()

        trainer = nm.trainer.Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            callback=callback,
            epochs=epochs,
            patience=epochs,
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric='dev_loss',
            warmup=400,
            lr_scheduler=False,
            device=ptu.device
        )

        # Train Over Nsteps
        # -----------------
        cl_system.nsteps = nstep
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

        # Update Parameters for the Next Iteration
        # ----------------------------------------
        lr *= lr_multiplier # 0.2

        # update the prediction horizon
        cl_system.nsteps = nstep

        optimizer.param_groups[0]['lr'] = lr

    callback.animate()
    callback.delete_all_but_last_image()

    # Save the Policy
    # ---------------
    # %%
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, policy_save_path + f"wp_traj_policy.pth")

def train_fig8(    # recommendations:
    iterations,      # 2
    epochs,          # 15
    batch_size,      # 5000
    minibatch_size,  # 10
    nstep,           # 100
    lr,              # 0.05
    Ts,              # 0.1
    policy_save_path = 'data/',
    media_save_path = 'data/training/',
    ):

    # unchanging parameters:
    R = 0.1
    Qpos = 5.00
    Qvel = 5.00
    barrier_type = 'softexp'
    barrier_alpha = 0.05
    lr_multiplier = 0.5

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images_path = media_save_path + current_datetime + '/'

    # Check if directory exists, and if not, create it
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # NeuroMANCER System Definition
    # -----------------------------

    nx = 6 # state size
    nr = 6 # reference size
    nu = 3 # input size
    np = 4 # number of params to describe the trajectory

    # Variables:
    r = nm.constraint.variable('R')           # the reference
    u = nm.constraint.variable('U')           # the input
    x = nm.constraint.variable('X')           # the state
    P = nm.constraint.variable('P')           # the equation parameters of the fig8

    # Nodes:
    node_list = []

    process_policy_input = lambda x, r, p: torch.hstack([r-x,p])
    process_policy_input_node = nm.system.Node(process_policy_input, ['X', 'R', 'P'], ['Obs'], name='preprocess')
    node_list.append(process_policy_input_node)

    policy = nm.modules.blocks.MLP(
        insize=nx + np, outsize=nu, bias=True,
        linear_map=torch.nn.Linear,
        nonlin=torch.nn.ReLU,
        hsizes=[20, 20, 20, 20]
    ).to(ptu.device)
    policy_node = nm.system.Node(policy, ['Obs'], ['U'], name='policy')
    node_list.append(policy_node)

    dynamics = Dynamics(insize=9, outsize=6)
    integrator = integrators.Euler(dynamics, h=torch.tensor(Ts))
    dynamics_node = nm.system.Node(integrator, ['X', 'U'], ['X'], name='dynamics')
    node_list.append(dynamics_node)

    print(f'node list used in cl_system: {node_list}')
    # node_list = [node.callable.to(ptu.device) for node in node_list]
    cl_system = nm.system.System(node_list)

    # Dataset Generation Class
    # ------------------------
    dataset = DatasetGenerator(
        task = 'fig8',
        batch_size = batch_size,
        minibatch_size = minibatch_size,
        nstep = nstep,
        Ts = Ts,
    )

    # Define Constraints:
    constraints = []

    # Define Loss:
    objectives = []

    action_loss = R * (u == ptu.tensor(0.))^2  # control penalty
    action_loss.name = 'action_loss'
    objectives.append(action_loss)

    pos_loss = Qpos * (x[:,:,::2] == r[:,:,::2])^2
    pos_loss.name = 'pos_loss'
    objectives.append(pos_loss)

    vel_loss = Qvel * (x[:,:,1::2] == r[:,:,1::2])^2
    vel_loss.name = 'vel_loss'
    objectives.append(vel_loss)

    # objectives = [action_loss, pos_loss, vel_loss]
    loss = nm.loss.BarrierLoss(objectives, constraints, barrier=barrier_type, alpha=barrier_alpha)
    optimizer = torch.optim.Adagrad(policy.parameters(), lr=lr)
    problem = nm.problem.Problem([cl_system], loss, grad_inference=True)

    # Custom Callack Setup
    # --------------------
    callback = utils.callback.SinTrajCallback(save_dir=current_datetime, media_path=media_save_path, nstep=nstep, nx=nx, Ts=Ts)


    # Perform the Training
    # --------------------
    for i in range(iterations):
        print(f'training with prediction horizon: {nstep}, lr: {lr}')

        # Get First Datasets
        # ------------
        dataset.nstep = nstep
        train_loader, dev_loader = dataset.get_loaders()

        trainer = nm.trainer.Trainer(
            problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer,
            callback=callback,
            epochs=epochs,
            patience=epochs,
            train_metric="train_loss",
            dev_metric="dev_loss",
            test_metric="test_loss",
            eval_metric='dev_loss',
            warmup=400,
            lr_scheduler=False,
            device=ptu.device
        )

        # Train Over Nsteps
        # -----------------
        cl_system.nsteps = nstep
        best_model = trainer.train()
        trainer.model.load_state_dict(best_model)

        # Update Parameters for the Next Iteration
        # ----------------------------------------
        lr *= lr_multiplier # 0.2

        # update the prediction horizon
        cl_system.nsteps = nstep

        optimizer.param_groups[0]['lr'] = lr


    callback.animate()
    callback.delete_all_but_last_image()

    # Save the Policy
    # ---------------

    # %%
    policy_state_dict = {}
    for key, value in best_model.items():
        if "callable." in key:
            new_key = key.split("nodes.0.nodes.1.")[-1]
            policy_state_dict[new_key] = value
    torch.save(policy_state_dict, policy_save_path + f"fig8_policy.pth")



if __name__ == "__main__":

    # best setup below for T1 desktop
    train = True
    run = True

    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    np.random.seed(0)
    ptu.init_gpu(use_gpu=False)

    import time
    train_fig8(iterations=1, epochs=5, batch_size=5000, minibatch_size=10, nstep=100, lr=0.05, Ts=0.1)
    train_wp_traj(iterations=1, epochs=10, batch_size=5000, minibatch_size=10, nstep=100, lr=0.05, Ts=0.1)
    start_time = time.time()
    train_wp_p2p(iterations=2, epochs=10, batch_size=5000, minibatch_size=10, nstep=100, lr=0.05, Ts=0.1)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Training elapsed time: {elapsed_time}")

    """ This worked rather well, slight z SSE remained
    torch.manual_seed(0)
    np.random.seed(0)
    ptu.init_gpu(use_gpu=False)

    import time

    start_time = time.time()
    train_wp_p2p(iterations=3, epochs=10, batch_size=5000)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Training elapsed time: {elapsed_time}")
    """
