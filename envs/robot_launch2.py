import time
import pybullet as p
import pybullet_data
import math

p.connect(p.GUI)

p.setTimeStep(0.01)
p.setPhysicsEngineParameter(numSolverIterations=50)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.8)

# Load ground
p.loadURDF("plane.urdf")

URDF_PATH = "/home/arjan/ros_spaces/gripper_ws/src/robotic_arm/urdf/robotic_arm.urdf"

# Load robot
p.loadURDF(
    URDF_PATH,
    [0.0, 0.2, 0.01],
    p.getQuaternionFromEuler([0.0, 0.0, math.pi / 2]),
)

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=6,
    cameraYaw=0,
    cameraPitch=-45,
    cameraTargetPosition=[0, 0, 0],
)

# Simulation loop
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)