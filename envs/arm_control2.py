import pybullet as p
import pybullet_data
import time
import os

urdf="/home/arjan/arm_rl/urdf/robotic_arm/urdf/robotic_arm.urdf"
#connect and load
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

#Load plane urdf and  robotic_arm
p.setGravity(0,0,-9.81)
p.loadURDF("plane.urdf")
robot_id=p.loadURDF(urdf, [0,0,0.01])

#camera 
p.resetDebugVisualizerCamera(
    cameraDistance=1,
    cameraYaw=0,
    cameraPitch=-45,
    cameraTargetPosition=[0,0,0],
)

# Define Joints
joints=[1,2,3]

#Automatic joint detection
# controllable_joint=[]

# for i in range(p.getNumJoints(robot_id)):
#     info=p.getJointInfo(robot_id,i)
#     joint_name=info[1].decode("utf-8")
#     joint_type=info[2]

#     if joint_type==p.JOINT_REVOLUTE:
#         controllable_joint.append(i)
#         print(f"Found Controllable joint:{joint_name} at index {i}")

# if len(controllable_joint)<3:
#     print("ERROR:Robot must have at least 3 joints.")
#     exit()


limits=[(-3.14,3.14),(-1.57,0.0),(-1.57,0.35)]
#Read limits from urdf
# limits=[]
# for j in joints:
#     info=p.getJointInfo(robot_id,j)
#     lower_limit=info[8]
#     upper_limit=info[9]

#     if lower_limit > upper_limit:
#         lower_limit=-3.14
#         upper_limit=3.14
    
#     limits.append((lower_limit,upper_limit))
#     print(f"joint {info[1].decode('utf-8')} limits: {lower_limit:.2f}.{upper_limit:.2f}")


targets=[0.0]*len(joints)#initial position
step_size=0.005 #How much the joint moves with keyboard press

# print(f"\nSuccessfully mapped {len(joints)} joints.")

print("Controls: \n U/I:Rotating \n H/J:Shoulder \n K/L: Arm")

while True:

    keys=p.getKeyboardEvents()
    if 27 in keys: break # ESC key
    
    #U/I for Rotating joints 
    if ord('u') in keys and keys[ord('u')] & p.KEY_IS_DOWN:
        targets[0]+=step_size
    if ord('i') in keys and keys[ord('i')] & p.KEY_IS_DOWN:
        targets[0]-=step_size
    
    #H/J for shoulder joints 
    if ord('h') in keys and keys[ord('h')] & p.KEY_IS_DOWN:
        targets[1]+=step_size
    if ord('j') in keys and keys[ord('j')] & p.KEY_IS_DOWN:
        targets[1]-=step_size
    #K/L for arm joint
    if ord('k') in keys and keys[ord('k')] & p.KEY_IS_DOWN:
        targets[2]+=step_size
    if ord('l') in keys and keys[ord('l')] & p.KEY_IS_DOWN:
        targets[2]-=step_size
    
    p.setJointMotorControlArray(
        robot_id,
        joints,
        p.POSITION_CONTROL,
        targetPositions=targets,
        forces=[500.0]*len(joints)
    )

    p.stepSimulation()
    time.sleep(1.0/240.0)