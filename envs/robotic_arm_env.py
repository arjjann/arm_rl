import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import time

class CustomEnv(gym.Env):
    """Environment for robotic arm control."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,arm_urdf, render=False):
        super().__init__()
        if render:
            self.client=p.connect(p.GUI)
        else:
            self.client=p.connect(p.DIRECT)

        #setup and urdf load
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        self.robot_id=p.loadURDF(arm_urdf,[0,0,0.01],useFixedBase=True)

        #joints
        self.joints=[1,2,3]
        self.goal_pos=np.array([0.5,0.5,0.5]) #initial targets
        #goal marker
        self.goal_marker=self._create_goal_marker()

        # Define action and observation space
        self.action_space=spaces.Box(low=-1.0, high=1.0, shape=(3,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf, shape=(9,),dtype=np.float32)

        #steps
        self.step_size=0.1
        self.max_steps=1000
        self.current_step=0
        self.prev_distance=0.0

        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def _create_goal_marker(self):
        """Creates a red sphere to visualize the goal position"""

        visual_shape=p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                         radius=0.03,
                                         rgbaColor=[1,0,0,0.8])
        marker=p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.goal_pos
        )
        return marker
    
    def _update_goal_marker(self):
        """Moves the end redsphere to the current goal position"""
        p.resetBasePositionAndOrientation(
            self.goal_marker,
            self.goal_pos,
            [0,0,0,1]  #no rotation
        )

    def _get_obs(self):
        """Returns the current state of the environmet."""

        joint_states =p.getJointStates(self.robot_id,self.joints)
        joint_pos=[s[0] for s in joint_states ]
        joint_vel=[s[1] for s in joint_states]

        return np.array(joint_pos + joint_vel + list(self.goal_pos),dtype=np.float32)
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step=0

        #Reset joint position to 0
        for i in self.joints:
            p.resetJointState(self.robot_id,i,0.0)

        #Randomize goal position for training
        min_radius=0.15
        max_radius=0.25
        goal_height=0.03

        #random point in space
        angle=np.random.uniform(0,2*np.pi)
        radius=np.random.uniform(min_radius,max_radius)

        rotating_state = p.getLinkState(self.robot_id, 1)
        base_pos = np.array(rotating_state[0])

        self.goal_pos=np.array([
            base_pos[0]+radius*np.cos(angle),
            base_pos[1]+radius*np.sin(angle),
            base_pos[2]+goal_height], 
            dtype=np.float32)

        self._update_goal_marker()
        #initilize previous state
        ee_state=p.getLinkState(self.robot_id,self.joints[-1])
        ee_pos=np.array(ee_state[0])
        self.prev_distance=np.linalg.norm(ee_pos-self.goal_pos)

        observation=self._get_obs()
        info={}
        return observation, info

    def step(self, action):
        joint_states=p.getJointStates(self.robot_id,self.joints)
        current_pos=[s[0] for s in joint_states]

        #calculate new target position
        new_targets=[current_pos[i] + (action[i]*self.step_size) for i in range(3)]

        #control robot
        p.setJointMotorControlArray(self.robot_id,
                                    self.joints,
                                    p.POSITION_CONTROL, 
                                    targetPositions=new_targets,
                                    forces=[500.0]*3)
        p.stepSimulation()

        #Calculate Reward(The closet the distance higher the reward)
        ee_state=p.getLinkState(self.robot_id,self.joints[-1])
        ee_pos=np.array(ee_state[0])
        distance=np.linalg.norm(ee_pos-self.goal_pos)

        progress=self.prev_distance-distance
        reward=progress*10.0 #reward for each progress
        reward-=0.01 #step penalty
        #Termination condition
        terminated=bool(distance<0.05) #success within 5cm
        if terminated:
            reward+=10.0 #success rewared

            for i in self.joints:
                p.resetJointState(self.robot_id,i,0.0)

            self._spawn_new_goal()

            ee_state = p.getLinkState(self.robot_id, self.joints[-1])
            ee_pos = np.array(ee_state[0])
            self.prev_distance = np.linalg.norm(ee_pos - self.goal_pos)


        #time limit
        self.current_step+=1
        truncated=bool(self.current_step>=self.max_steps) # time limit defined
        # if truncated and not terminated:
        #     reward-=5.0 #time penalty


        observation=self._get_obs()
        info={"distance":distance}
        self.prev_distance=distance

        return observation,reward,terminated, truncated, info
    
    def _spawn_new_goal(self):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.15, 0.25)
        rotating_state = p.getLinkState(self.robot_id, 1)
        base_pos = np.array(rotating_state[0])
        self.goal_pos = np.array([
            base_pos[0] + radius * np.cos(angle),
            base_pos[1] + radius * np.sin(angle),
            base_pos[2] + 0.03
        ], dtype=np.float32)
        self._update_goal_marker()
    
    def render(self):
        pass

    def close(self):
        if hasattr(self,'client'):
            p.disconnect(self.client)