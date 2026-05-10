import math
import random

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from pybullet_world import create_world
from sim import (
    create_goal_marker,
    reset_robot,
    reset_goal_marker,
    apply_velocity,
    get_lidar_scan,
    get_robot_state,
    check_collision,
    can_spawn_here,
)
SIM_HZ    = 100    
POLICY_HZ = 100    
SIM_STEPS = 1      
class TurtleBotEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    # # Candidate spawn poses  [x, y, theta]
    # SPAWN_LIST = [
    #     ( 0.0, -1.0,  math.pi / 2),
    #     ( 0.0,  1.0,  math.pi / 2),
    #     ( 1.0,  0.0,  0.0),
    #     (-1.0,  0.0,  math.pi / 2),
    #     ( 2.0,  1.5, -math.pi / 4),
    # ]

    # # Candidate goal positions  [x, y]
    # GOAL_LIST = [
    #     ( 0.0, -1.2),
    #     ( 1.3, -0.5),
    #     (-1.2, -1.1),
    #     ( 1.0,  0.0),
    #     (-1.0,  0.0),
    #     ( 1.0,  1.3),
    #     (-1.0,  0.8),
    #     ( 1.4,  1.4),
    #     (-1.4,  1.0),
    # ]

    def __init__(self, render: bool = False):
        super().__init__()

        # Task parameters
        self.LIDAR_RANGE      = 3.5
        self.MAX_LINEAR_VEL   = 0.22      # m/s
        self.MAX_ANGULAR_VEL  = 2.0       # rad/s
        self.MAX_GOAL_DIST    = math.hypot(5.0, 5.0)   # rough arena diagonal

        self.GOAL_TOLERANCE   = 0.05
        self.MIN_SPAWN_DIST   = 0.8
        self.MAX_STEPS        = 6000

        # PyBullet world
        self.physicsClient, self.robotId, self.planeId = create_world(render=render)
        self.goal_marker_id = create_goal_marker(
            self.physicsClient, 0.0, 0.0
        )

        # Runtime state
        self.goal          = np.zeros(2, dtype=np.float32)
        self.x             = 0.0
        self.y             = 0.0
        self.yaw           = 0.0
        self.vel_linear    = 0.0
        self.vel_angular   = 0.0
        self.lidar         = np.full(36, self.LIDAR_RANGE, dtype=np.float32)
        self.prev_distance = None
        self.step_count    = 0
        self.total_episodes = 0

        # Spaces
        n_lidar = 36
        low  = np.array([0.0, -1.0, -1.0,  0.0, -1.0] + [0.0] * n_lidar, dtype=np.float32)
        high = np.array([1.0,  1.0,  1.0,  1.0,  1.0] + [1.0] * n_lidar, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
        )


    def _sync_state(self) -> None:
       #updates the robot state
        state = get_robot_state(self.physicsClient, self.robotId)

        self.x   = state["x"]
        self.y   = state["y"]
        self.yaw = state["yaw"]

        lin_vel = state["lin_vel"]
        ang_vel = state["ang_vel"]
        self.vel_linear  = (lin_vel[0] * math.cos(self.yaw)
                            + lin_vel[1] * math.sin(self.yaw))
        self.vel_angular = ang_vel[2]   # yaw rate

        self.lidar = get_lidar_scan(self.physicsClient, self.robotId)


    def _get_obs(self) -> np.ndarray:
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist      = math.hypot(dx, dy)
        dist_norm = float(np.clip(dist / self.MAX_GOAL_DIST, 0.0, 1.0))
        heading_error = math.atan2(dy, dx) - self.yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        linear_norm  = float(np.clip(self.vel_linear  / self.MAX_LINEAR_VEL,  0.0,  1.0))
        angular_norm = float(np.clip(self.vel_angular / self.MAX_ANGULAR_VEL, -1.0, 1.0))
        lidar_norm   = np.clip(self.lidar / self.LIDAR_RANGE, 0.0, 1.0).astype(np.float32)

        return np.concatenate([
            [dist_norm],
            [math.cos(heading_error)],
            [math.sin(heading_error)],
            [linear_norm],
            [angular_norm],
            lidar_norm,
        ]).astype(np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        linear  = ((float(action[0]) + 1.0) / 2.0) * self.MAX_LINEAR_VEL
        angular = float(action[1]) * self.MAX_ANGULAR_VEL

        apply_velocity(self.physicsClient, self.robotId, linear, angular)

        for _ in range(SIM_STEPS):
            p.stepSimulation(physicsClientId=self.physicsClient)


    def _pick_spawn_and_goal(self):
        for _ in range(200):
            goal = np.array([
                random.uniform(-2.1,2.1),
                random.uniform(-2.1, 2.1)
            ], dtype=np.float32)

            spawn = [
                random.uniform(-2.1, 2.1),
                random.uniform(-2.1, 2.1),
                random.uniform(-math.pi, math.pi)
            ]
            dist = math.hypot(goal[0] - spawn[0], goal[1] - spawn[1])
            if dist < self.MIN_SPAWN_DIST:
                continue

            # Reject if goal is inside an obstacle
            if not can_spawn_here(self.physicsClient, goal[0], goal[1], blocked_ids=set()):
                continue

            # Reject if spawn is inside an obstacle
            if not can_spawn_here(self.physicsClient, spawn[0], spawn[1], blocked_ids=set()):
                continue

            return spawn, goal
        return [0.0, -1.9, math.pi / 2], np.array([0.0, 1.5], dtype=np.float32)


   

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_episodes += 1
        apply_velocity(self.physicsClient, self.robotId, 0.0, 0.0)
        spawn, self.goal = self._pick_spawn_and_goal()

        reset_robot(
            self.physicsClient, self.robotId,
            x=spawn[0], y=spawn[1], theta=spawn[2],
        )
        reset_goal_marker(
            self.physicsClient, self.goal_marker_id,
            x=self.goal[0], y=self.goal[1],
        )

  
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physicsClient)

        self.step_count = 0
        self._sync_state()

        obs = self._get_obs()
        self.prev_distance = float(obs[0])   # normalised distance at episode start

        return obs, {}

    def step(self, action):
        self.step_count += 1

        self._apply_action(action)
        self._sync_state()

        obs = self._get_obs()
        dist_norm = float(obs[0])
        dist_m    = dist_norm
        lidar_min = float(np.min(self.lidar))          # metres

  
        collision = check_collision(
            self.physicsClient, self.robotId, self.planeId
        )
        progress = self.prev_distance - dist_m    
        reward   = 1.00 * progress

        SAFE_DIST = 0.2
        if lidar_min < SAFE_DIST:
            reward += -0.2 * (1.0 - lidar_min / SAFE_DIST) ** 2

        
        reward += -0.005
        reached_goal = dist_m < self.GOAL_TOLERANCE

        if collision:
            reward = -20.0
        elif reached_goal:
            reward = 20.0

        terminated = collision or reached_goal
        truncated  = self.step_count >= self.MAX_STEPS

        if not terminated:
            self.prev_distance = dist_m

        return obs, float(reward), terminated, truncated, {}

    def close(self):
        apply_velocity(self.physicsClient, self.robotId, 0.0, 0.0)
        p.disconnect(physicsClientId=self.physicsClient)


