""" This code describes a training environment for the NVIDIA Isaac Gym 
simulator. In order to run this environment this file should be put in the 
`tasks` folder, as well as its entry added to `config.py` and `parse_task.py`.

"""
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import numpy as np
import random
import torch
import cv2

from rlgpu.utils.torch_jit_utils import *
from rlgpu.utils.pygame_utils import *
from rlgpu.utils.logger import Logger

HEIGHT = 720
WIDTH = 1000

class GogoroSteeringCommand(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, graphics_device, device):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.graphics_device = graphics_device
        self.device = device

        # Number of envs can be used interchangeably with number of actors as 
        # we only create 1 actor per env
        self.num_envs = self.cfg["env"]["numEnvs"]
        # Max lean angle of bike
        self.tilt_limit = np.radians(self.cfg["env"]["tiltLimit"])
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.min_speed = self.cfg["env"]["speedRange"][0]
        self.max_speed = self.cfg["env"]["speedRange"][1]
        self.range_speed = self.max_speed - self.min_speed
        self.set_speed = self.cfg["env"]["setSpeed"]

        self.min_delta_yaw = np.radians(self.cfg["env"]["yawRange"][0])
        self.max_delta_yaw = np.radians(self.cfg["env"]["yawRange"][1])
        #self.delta_yaw_limit = self.cfg["env"]["yawDeltaLimit"]
        self.set_target_delta_yaw = self.cfg["env"]["setDeltaYaw"]
        if self.set_target_delta_yaw != None:
            self.set_target_delta_yaw = np.radians(self.set_target_delta_yaw)

        self.test_mode = self.cfg["env"]["test_mode"]
        self.curses_mode = self.cfg["env"]["curses_mode"]
        self.pygame_mode = self.cfg["env"]["pygame_mode"]
        self.noise_scale = self.cfg["env"]["noise_scale"]
        self.log_observations = self.cfg["env"]["log_observations"]

        # Set up curses
        if self.curses_mode:
            import curses
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)

            self.curses_yaw = 0.0

        if self.pygame_mode:
            import pygame
            self.joy_screen = JoystickScreen(600, 600)

        # Observations are [1x `roll`,
        #                   1x `pitch`,
        #                   1x `yaw`,
        #                   1x `curr delta yaw`,
        #                   1x `steering_angle`
        #                   1x `rear wheel velocity`, 
        #                   1x `target delta yaw`,
        #                   1x `curr delta roll`,
        #                   1x `curr delta pitch,
        #                   3x `linear velocities`
        #
        num_obs = 12
        # The action is setting the angle of the steering  joint
        num_acts = 1

        # Create logger object to record the observations at each time step
        fields = ["roll", "pitch", "yaw", "delta_yaw", "steering_angle",
                "rear_wheel_velocity", "target_delta_yaw", "curr_delta_roll", 
                "curr_delta_pitch", "linear_vel1", "linear_vel2", "linear_vel3"]
        self.logger = Logger("gogoro_steering_command_logtest.txt", fields)

        # Same as kafka arm env
        self.action_scale = 10.0
        self.dt = 1/60.
        self.root_velocity_scale = 0.25
        
        # Call `BaseTask` constructor. This will initialize the simulation,
        # importing the assets and etc. Calls `create_sim`
        super().__init__(
            num_obs=num_obs,
            num_acts=num_acts,
            num_envs=self.cfg["env"]["numEnvs"],
            graphics_device=graphics_device,
            device=device
        )

        # Acquiring reference to the tensors that contain the state of the 
        # actors in the environment. For now, the only observation we use for
        # the gogoro environment is the Roll angle of the bike's body.
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor) # PyTorch-wrapped tensor
        # Actors x Info
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_angular_vels = self.root_tensor[:, 10:13]

        self._state_dof = self.gym.acquire_dof_state_tensor(self.sim)
        self.state_dof = gymtorch.wrap_tensor(self._state_dof)
        self.dof_pos = self.state_dof.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.state_dof.view(self.num_envs, self.num_dof, 2)[..., 1]

        # Set initial state to DOF targets
        for j_name in self.cfg["joints_pos"]:
            self.dof_pos[:, self.dof_name_to_id[j_name]] = self.cfg["joints_pos"][j_name]

        # Mount the tensor used to reset the environments
        #self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_reset_tensor = self.root_tensor.clone() # Copy root tensor
        self.reset_state_dof = self.state_dof.clone()
        self.root_reset_tensor[:, 7:13] = 0 # Set all velocities to 0

        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.target_delta_yaw = torch.zeros_like(self.root_angular_vels[:, 2])

        # The noise scale added to the observations
        self.noise_scale_tensor = torch.ones(self.num_obs, device=self.device,
                dtype=torch.float32) * self.noise_scale

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(0, self.graphics_device, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.7
        plane_params.dynamic_friction = 0.7
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "../../assets"
        asset_file = "urdf/gogoro_description/urdf/gogoro_and_thormang3.urdf"
        # Load asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Get number of DOFs and DOF names
        self.num_dof = self.gym.get_asset_dof_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        # Helper dictionary to map joint names to tensor ID
        self.dof_name_to_id = {k: v for k, v in zip(self.dof_names, np.arange(self.num_dof))}

        # set up the env grid
        num_envs = num_envs
        envs_per_row = num_per_row
        env_spacing = spacing
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # Set the target position of the DOFs according to the cfg
        self.pos_targets = np.zeros(self.num_dof).astype('f')
        # Set DOF initial position according to CFG
        for j_name in self.cfg["joints_pos"]:
            self.pos_targets[self.dof_name_to_id[j_name]] = self.cfg["joints_pos"][j_name]

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 1.2)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.envs = []
        self.handles = []
        # create and populate the environments
        for i in range(num_envs):
            # Create environments
            ref_env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            self.envs.append(ref_env)

            actor_handle = self.gym.create_actor(ref_env, asset, start_pose, "Gogoro", i, 1)
            self.handles.append(actor_handle)

            # Set controller modes
            props = self.gym.get_actor_dof_properties(ref_env, actor_handle)
            # Everything set to pos so they all stay in-place
            props["driveMode"].fill(gymapi.DOF_MODE_POS)
            props["stiffness"].fill(1000.0)
            props["damping"].fill(200.0)

            # Set velocity mode on back wheel
            props["driveMode"][self.dof_name_to_id['rear_wheel_joint']] = gymapi.DOF_MODE_VEL
            props["damping"][self.dof_name_to_id['rear_wheel_joint']] = 15.0 

            # Front wheel mode set to None so it runs freely
            props["driveMode"][self.dof_name_to_id['front_wheel_joint']] = gymapi.DOF_MODE_NONE
            props["stiffness"][self.dof_name_to_id['front_wheel_joint']] = 0.0
            props["damping"][self.dof_name_to_id['front_wheel_joint']] = 0.0

            # Set position mode for steering joint
            props["driveMode"][self.dof_name_to_id['steering_joint']] = gymapi.DOF_MODE_POS
            props["stiffness"][self.dof_name_to_id['steering_joint']] = 1000.0
            props["damping"][self.dof_name_to_id['steering_joint']] = 200.0
            self.gym.set_actor_dof_properties(ref_env, actor_handle, props)

            # Set constant speed for back wheel
            speeds = np.full(self.num_dof, 0).astype('f')
            # Set speed, will only change rear wheel
            #self.gym.set_actor_dof_velocity_targets(ref_env, actor_handle, speeds)
            # Set pos targets for every other joint
            self.gym.set_actor_dof_position_targets(ref_env, actor_handle, self.pos_targets)

        self.steer_upper_limit = props["upper"][13]
        self.steer_lower_limit = props["lower"][13]
        self.steering_range = self.steer_upper_limit - self.steer_lower_limit


        # Position targets tensor that is modified every loop
        # Initialize with the position targets of the still joints
        p = np.tile(self.pos_targets, self.num_envs)
        self.positions_tensor = torch.tensor(p, dtype=torch.float32, device=self.device)


        if self.test_mode:
            # Used for testing, create a camera and attach it to the last created actor
            camera_props = gymapi.CameraProperties()
            camera_props.width = WIDTH
            camera_props.height = 720
            self.cam_handle = self.gym.create_camera_sensor(ref_env, camera_props)
            # Attach camera to actor body
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(3.0, 0.0, 0.75)
            local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(75.0))
            self.gym.attach_camera_to_body(self.cam_handle, ref_env, actor_handle, local_transform, 1)

            # Set up text objects for OpenCV
            self.font = cv2.FONT_HERSHEY_SIMPLEX 
            self.font_thickness = 2
            self.font_scale = 0.8
            self.font_pos = (10, 30)
            self.font_color = (0, 255, 0)

            self.video_writer = cv2.VideoWriter("output.avi", 
                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                    30, (WIDTH, HEIGHT))

    def reset(self, env_ids):
        # Reset environments indicated by `env_ids`
        #print("Reset called")

        # Set sim to reset tensor for envs in env_ids
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        _env_ids_int32 = gymtorch.unwrap_tensor(env_ids_int32) 

        for env_id in env_ids_int32.cpu().numpy():
            if self.set_speed == -1:
                speed = np.random.rand() * self.range_speed + self.min_speed
            else:
                speed = self.set_speed
            targets = np.full(self.num_dof, speed).astype('f')
            self.gym.set_actor_dof_velocity_targets(self.envs[env_id],
                    self.handles[env_id], targets)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                gymtorch.unwrap_tensor(self.reset_state_dof),
                _env_ids_int32, len(env_ids_int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                gymtorch.unwrap_tensor(self.root_reset_tensor),
                _env_ids_int32, len(env_ids_int32))

        # Reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def compute_observations(self):
        # Refresh the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Randomly generate offsets for the target delta yaw from a gaussian
        # distribution
        random_offsets = torch.rand_like(self.target_delta_yaw) * 0.2 - 0.1
        self.target_delta_yaw = torch.where(self.progress_buf % 150 == 0, 
                self.target_delta_yaw + random_offsets,
                self.target_delta_yaw)
        
        self.target_delta_yaw = torch.where(self.target_delta_yaw < self.min_delta_yaw,
                torch.ones_like(self.target_delta_yaw) * self.min_delta_yaw,
                self.target_delta_yaw)
        self.target_delta_yaw = torch.where(self.target_delta_yaw > self.max_delta_yaw,
                torch.ones_like(self.target_delta_yaw) * self.max_delta_yaw,
                self.target_delta_yaw)

        # Override target delta yaw tensor if value was set in config file
        if self.set_target_delta_yaw != None:
            self.target_delta_yaw = torch.ones_like(self.target_delta_yaw) * self.set_target_delta_yaw

        # Override target delta yaw tensor if curses mode is on
        if self.curses_mode:
            c = self.stdscr.getch()
            if c == ord('a'):
                self.curses_yaw += np.radians(1)
            elif c == ord('d'):
                self.curses_yaw -= np.radians(1)

            self.target_delta_yaw = torch.ones_like(self.target_delta_yaw) * self.curses_yaw

        # Override target delta yaw with joystick control
        if self.pygame_mode:
            # Update joystick visualization and get joystick inputs
            left_X, btns = self.joy_screen.update()
            
            if left_X >= 0:
                delta_yaw_cmd = left_X * self.min_delta_yaw
            else:
                delta_yaw_cmd = -left_X * self.max_delta_yaw

            # Set delta target
            self.target_delta_yaw = torch.ones_like(self.target_delta_yaw) * delta_yaw_cmd

        # Insert relevant info in the observation buffer
        self.obs_buf[:] = compute_gogoro_observations(self.root_tensor,
                                            self.dof_pos,
                                            self.dof_vel,
                                            self.target_delta_yaw,
                                            self.inv_start_rot,
                                            self.noise_scale_tensor,
                                            self.max_speed)

        # Save iteration yaw to compute delta yaw in next
        #self.last_yaw = self.obs_buf[:, 2].clone()

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_gogoro_reward(self.obs_buf,
                self.progress_buf,
                self.reset_buf,
                self.tilt_limit, 
                self.max_episode_length,
                self.root_velocity_scale)

        #input("freeze")

    def pre_physics_step(self, actions):
        # Modify position tensor for steering joint only (id:13)
        self.positions_tensor[13::self.num_dof] = torch.squeeze(actions) * self.steer_upper_limit * self.action_scale 
        # Clamp to limits
        self.positions_tensor[13::self.num_dof] = \
                torch.clamp(self.positions_tensor[13::self.num_dof], 
                        self.steer_lower_limit, self.steer_upper_limit)

        _positions_tensor = gymtorch.unwrap_tensor(self.positions_tensor)

        self.gym.set_dof_position_target_tensor(self.sim, _positions_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        
        # Check which environments should be reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        #print(f"Reset IDs length: {len(env_ids)}")
        if len(env_ids) > 0:
            self.reset(env_ids)

        # Compute observations
        self.compute_observations()

        # Compute rewards
        self.compute_reward()

        if self.test_mode:
            self.gym.render_all_camera_sensors(self.sim)
            color_image = self.gym.get_camera_image(self.sim, self.envs[-1], self.cam_handle, gymapi.IMAGE_COLOR)

            # Grab info that will put on screen
            curr_delta_yaw = np.degrees(self.obs_buf[0, 3].cpu().item())
            curr_steering_angle = np.degrees(self.obs_buf[0, 4].cpu().item())
            curr_target_delta_yaw = np.degrees(self.obs_buf[0, 6].cpu().item())
            speed = self.obs_buf[0, 5].cpu().item() * self.max_speed

            color_image = color_image.reshape(720, WIDTH, 4)

            pos = self.font_pos
            color_image = cv2.putText(color_image, f"Delta Yaw: {curr_delta_yaw:.2f}",
                    pos, self.font, self.font_scale, self.font_color,
                    self.font_thickness, cv2.LINE_AA)
            pos = (pos[0], pos[1] + 30)
            color_image = cv2.putText(color_image, f"Target Delta Yaw: {curr_target_delta_yaw:.2f}",
                    pos, self.font, self.font_scale, self.font_color,
                    self.font_thickness, cv2.LINE_AA)
            pos = (pos[0], pos[1] + 30)
            color_image = cv2.putText(color_image, f"Steering Angle: {curr_steering_angle:.2f}",
                    pos, self.font, self.font_scale, self.font_color,
                    self.font_thickness, cv2.LINE_AA)
            pos = (pos[0], pos[1] + 30)
            color_image = cv2.putText(color_image, f"Speed: {speed:.1f} rad/s | {speed * 0.5:.1f} km/h",
                    pos, self.font, self.font_scale, self.font_color,
                    self.font_thickness, cv2.LINE_AA)

            color_image = color_image[:, :, :3]

            #self.video_writer.write(color_image)

            cv2.imshow("Camera Image", color_image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('s'):
                print("Saving frame...")
                cv2.imwrite("screenshot.png", color_image)

            if self.log_observations:
                self.logger.record(self.obs_buf[0, :].cpu().numpy())

@torch.jit.script
def compute_gogoro_reward(obs_buf, progress_buf, reset_buf,
        tilt_limit, max_episode_length, delta_scale):
    # type: (Tensor, Tensor, Tensor, float, int, float) -> Tuple[Tensor, Tensor]

    # Reset if over iteration limit
    reset = torch.where(progress_buf > max_episode_length, 1, reset_buf)
    # Reset if over tilt limit
    reset = torch.where(torch.abs(obs_buf[:, 0]) > tilt_limit, 1, reset)

    curr_delta_yaw = obs_buf[:, 3] #* delta_scale
    desired_delta = obs_buf[:, 6]
    diff_delta = curr_delta_yaw - desired_delta

    reward = 1.0 - (torch.abs(diff_delta) ** 0.2)

    # Negative reward for falling over
    reward = torch.where(torch.abs(obs_buf[:, 0]) > tilt_limit, 
                         torch.ones_like(reward) * -2.0, reward)

    return reward, reset

@torch.jit.script
def compute_gogoro_observations(root_state, dof_pos, dof_vel, desired_delta_yaw, 
        inv_start_rot, noise_scale, max_rear_wheel_velocity):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor

    #root_orientations = torch
    root_position = root_state[:, 0:3]
    root_orientations = root_state[:, 3:7]
    root_linvel = root_state[:, 7:10]
    root_angvel = root_state[:, 10:13]

    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_gogoro(
        root_orientations, inv_start_rot, root_linvel, root_angvel, root_position)

    # Get orientation information from tensors
    roll = normalize_angle(roll).unsqueeze(-1)
    pitch = normalize_angle(pitch).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)

    # 12 index of rear wheel joint
    rear_wheel_vel = dof_vel[:, 12] / max_rear_wheel_velocity
    rear_wheel_vel = rear_wheel_vel.unsqueeze(-1)

    # 13 index of steering joint
    curr_steering_angle = dof_pos[:, 13].unsqueeze(-1)

    delta_yaw = angvel_loc[:, 2].unsqueeze(-1)
    desired_delta_yaw = desired_delta_yaw.unsqueeze(-1)

    # Mount observation vector
    obs = torch.cat([roll, pitch, yaw, delta_yaw, curr_steering_angle, 
        rear_wheel_vel, desired_delta_yaw, angvel_loc[:, :2], vel_loc[:]], dim=1)

    # Generate gaussian noise
    gaussian_noise = torch.randn_like(obs) * noise_scale
    
    # Get noisy observation vector
    noisy_obs = gaussian_noise + obs 
    # Patch intended delta to remove noise
    noisy_obs[:, 6] = obs[:, 6]

    return noisy_obs 
