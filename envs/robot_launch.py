import pybullet as p
import pybullet_data
import time
import os

urdf="/home/arjan/ros_spaces/gripper_ws/src/robotic_arm/urdf/robotic_arm.urdf"
if not os.path.exists(urdf):
    print(f"ERROR:Cannot find urdf {urdf}")
else:
    #connect and load
    physicsClient=p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #Load plane urdf and  robotic_arm
    p.setGravity(0,0,-9.81)
    p.loadURDF("plane.urdf")
    p.loadURDF(urdf, [0,0,0.01])

    #camera 
    p.resetDebugVisualizerCamera(
        cameraDistance=6,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[0,0,0],
    )

while True:
    p.stepSimulation()
    time.sleep(1.0/240.0)