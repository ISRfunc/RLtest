import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import gym
from gym import spaces


MAX_EPISODE_LEN = 20*100

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*4*2), np.array([1]*4*2))
        self.observation_space = spaces.Dict(
            {
                "img": spaces.Box(low=0, high=255, shape=(3, 112, 112)),
                "joints": spaces.Box(np.array([-1]*4*2), np.array([1]*4*2)),
            }
        )
        

    def get_reward(self, efPos, objPos): #efPos, objPos, current_phase, boxPos):

        contact_points = p.getContactPoints(self.pandaUid)

        collide = False
        if len(contact_points) > 0:
            for point in contact_points:
                if point[2] != self.objectUid:
                    if not ( (point[3] in [6, 7, 8, 9, 10, 11]) and (point[4] in [6, 7, 8, 9, 10, 11]) ):
                        collide = True
        
        if collide:
            return -5
        


        def quaternion_multiply(quaternion1, quaternion0):
            w0, x0, y0, z0 = quaternion0
            w1, x1, y1, z1 = quaternion1
            return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                             x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                             -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                             x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

        def rotate(quat):
            P = [0., 1., 0., 0.]
            x, y, z, w = quat
            R = [w,  x,  y,  z]
            R_ = [w, -x, -y, -z]

            P_ = quaternion_multiply(quaternion_multiply(R, P), R_)

            return P_

        def compute_cosine_distance(vecA, vecB):
            # Convert vecA and vecB to numpy arrays, if they are not already
            vecA = rotate(vecA)
            vecB = rotate(vecB)
            
            # Compute the Euclidean distance
            distance = np.dot(vecA, vecB)
            return distance


        
        orientDist = compute_cosine_distance(efPos[1], p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.]))



        if self.current_phase == 'move to obj':

            threshold = 0.1

            def compute_distance(vecA, vecB):
                # Convert vecA and vecB to numpy arrays, if they are not already
                vecA = np.array(vecA)
                vecB = np.array(vecB)
                
                # Compute the Euclidean distance
                distance = np.linalg.norm(vecA - vecB)
                return distance

           
            distance = compute_distance(efPos[0], objPos)
            if distance < threshold and objPos[2] > 0.1:
                self.current_phase = 'grip'
            else:
                return 1 - distance * 5 + orientDist
             
        
        
        if self.current_phase == 'grip':
            if not (distance < threshold and objPos[2] > 0.1):
                self.current_phase = 'move to obj'
            else:
                return 2 + objPos[2] * 2 + orientDist
            


        """
        if current_phase == 'move obj to box': 
            
            if dist( bestDropPos(boxPos), objPos ) < threshold: [with picked object]
                return 0

        
        if current_phase == 'drop obj to box': [with picked object]

            if dist(boxPos, objPos) < threshold: 
                return 0

        """
        return -1
        

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        
        fingers = round(action[-1]) * 0.04

        """
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv * 1.5

        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        if self.step_counter == 0:
            self.currentPosition = [currentPosition[0] + 0.23,
                       currentPosition[1] + 0,
                       currentPosition[2] + -0.2]
        if self.step_counter < 5:
            currentPosition = self.currentPosition
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]
        """
        # update target joints        
        jointPoses = np.array(action[:-1]) * 0.3 + self.currentPose
        self.currentPose = jointPoses


        # send command to control joints
        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        for i in range(8*3):
            p.stepSimulation()

        

        # get feedback from environment
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])


        # calculate reward
        """
        if state_object[2]>0.45:
            reward = 1 * 3
            done = True
        else:
            reward = 0
            done = False
        """
        reward = 0
        done = False

        self.step_counter += 1

        # check termination
        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        info = {'object_position': state_object}
        # get observation
        state_robot = p.getLinkState(self.pandaUid, 11)[0:2] # gripper position and orientation
        state_joints = p.getJointStates(self.pandaUid,  [i for i in range(7)] + [9]) # griper joint
        state_joints = [state_joint[0] for state_joint in state_joints]

        img = self.render()
        observation = {
            "img": np.array(img).astype(np.float32),
            "joints": np.array(state_joints).astype(np.float32)
        }

        reward += self.get_reward(state_robot, state_object)

        reward *= 0.001

        return observation, reward, done, False, info
        #return np.array(self.observation).astype(np.float32), reward, done, False, info

    def reset(self):

        self.current_phase = 'move to obj'

        # reset simulator
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-9.81)

        # add floor
        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        # add robot
        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08] #[0]*9#
        self.currentPose = np.array(rest_poses[:-2]) #############################################################################################################
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True, flags=(p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER))

        # set start pose
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        # add trays
        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        trayLeftUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0,0.4,0], useFixedBase=True, globalScaling=0.6)
        trayRightUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0,-0.4,0], useFixedBase=True, globalScaling=0.6)

        # add objects
        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05] #[0.65,0.,0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)

        # get observation
        state_robot = p.getLinkState(self.pandaUid, 11)[0] # gripper position
        state_joints = p.getJointStates(self.pandaUid,  [i for i in range(7)] + [9]) # griper joint
        state_joints = [state_joint[0] for state_joint in state_joints]
        
        #self.observation = state_joints
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        img = self.render()
        observation = {
            "img": np.array(img).astype(np.float32),
            "joints": np.array(state_joints).astype(np.float32)
        }

        return observation, {}
        #return np.array(self.observation).astype(np.float32), {}

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=112,
                                              height=112,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (112, 112, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array.transpose(2, 0, 1) / 255.

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()


from gym.envs.registration import register
from PIL import Image

register(
    id='panda-v0',
    entry_point='trainRL:PandaEnv',
)

#envs = gym.make('panda-v0')

numrobots = 50
numworkers = 10
num_steps = 200
envs = gym.vector.AsyncVectorEnv([
    lambda: gym.make("panda-v0") for i in range(numworkers)
])

obs, info = envs.reset()

print(obs["img"].shape)
print(obs["joints"].shape)
observation_space = envs.get_attr("observation_space")[0]
action_space = envs.get_attr("action_space")[0]


"""
for i in range(num_steps):
    action = envs.action_space.sample()
    print(action)
    obs, rew, terminated, truncated, info = envs.step(action)
"""

from agent import PPO, Actor, Critic, Resnet10

img_height = 112
img_width  = 112
mode = "rgb"

backbone = Resnet10(height=img_height, width=img_height, mode=mode)
actor  = Actor(arch=backbone, n_joints=8).to("cuda")
critic = Critic(arch=backbone).to("cuda")


from torch.utils.tensorboard import SummaryWriter
import torch




# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#dummy_state = torch.randn(10, 3, 112, 112).to("cuda")
    
logging = SummaryWriter(log_dir="./temp") # initialize tensorboard summary writer

ppo = PPO(actor=actor, critic=critic, logging=logging)

"""
means, stds = actor(dummy_state)
print(means.shape)
print(stds.shape)
"""

logging.close() # close the writer 


from utils import RollOut

ro = RollOut(num_steps, numrobots, numworkers, observation_space, action_space) #.to(device)

from torch.distributions import Normal

import gc

count = 0

max_eps = int(1e6)

for ep in range(max_eps):

    for i in range(numrobots // numworkers):

        for step in range(num_steps):

            #action = envs.action_space.sample() * 0
            #action[:, -2] = 0.06
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to("cuda")
            obs_temp = obs

            means, stds = actor(obs)
            means, stds = means.cpu(), stds.cpu()
            dist = Normal(means, stds)
            actions = dist.sample().cpu()
            obs, rew, terminated, truncated, info = envs.step(actions.numpy())

            rewards = torch.from_numpy(rew).view(-1, 1)
            value_preds = critic(obs_temp).cpu()
            for key in obs_temp:
                obs_temp[key] = obs_temp[key].cpu()
            action_log_probs = dist.log_prob(actions).sum(1) #.to(device)
            ro.insert(obs_temp, actions, action_log_probs, value_preds, rewards)

            count += 1
            print("Memory before clearing:", count)
            print(torch.cuda.memory_allocated(), "allocated")
            print(torch.cuda.memory_reserved(), "reserved")

            del means, stds, actions, dist, rewards, value_preds, action_log_probs, obs_temp
            gc.collect()
            torch.cuda.empty_cache()

            print("Memory after clearing:")
            print(torch.cuda.memory_allocated(), "allocated")
            print(torch.cuda.memory_reserved(), "reserved")

        for key in obs:
            obs[key] = torch.from_numpy(obs[key]).to("cuda")
        next_value = critic(obs).cpu()
        
        obs, info = envs.reset()

        gc.collect()
        torch.cuda.empty_cache()



        ro.compute_returns_and_gae(next_value, True, 0.99, 0.95)

    if ep % 20 == 0:
        ppo.save(f"./checkpoint/checkpoint_{ep}")

    ppo.update(ro, numrobots)





"""
def set_target(self, joint_state_target):

        # Set target
        if self.control_mode == self.p.POSITION_CONTROL:
            target = {'targetPositions': joint_state_target.position}
        elif self.control_mode == self.p.VELOCITY_CONTROL:
            target = {'targetVelocity': joint_state_target.velocity}
        elif self.control_mode == self.p.TORQUE_CONTROL:
            target = {'forces': joint_state_target.effort}
        else:
            raise ValueError("did not recognize control mode!")

        # Set gains if supplied by user
        if 'positionGains' in self.pb_obj.config['setJointMotorControlArray']:
            target['positionGains'] = self.pb_obj.config['setJointMotorControlArray']['positionGains']
        if 'velocityGains' in self.pb_obj.config['setJointMotorControlArray']:
            target['velocityGains'] = self.pb_obj.config['setJointMotorControlArray']['velocityGains']

        # Set target
        self.p.setJointMotorControlArray(
            self.p.pandaUid,
            [self.name_to_index(name) for name in joint_state_target.name],
            self.control_mode,
            **target,
        )
"""
