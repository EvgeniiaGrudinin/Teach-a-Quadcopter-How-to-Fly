import numpy as np
from physics_sim import PhysicsSim

class Task():
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=4, target_pos=None):
        self.sim=PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat=3

        self.state_size=self.action_repeat*6
        self.action_low=0
        self.action_high=900
        self.action_size=4

        # Goal
        self.target_pos=target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        reward=0
        negative_reward=0
        reward=10*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum()
        position=self.sim.pose[:3]
        negative_reward+=abs(self.sim.pose[3:6]).sum()
        negative_reward+=abs(position[0]-self.target_pos[0])**2
        negative_reward+=5*abs(position[1]-self.target_pos[1])**2
        negative_reward+=10*abs(position[2]-self.target_pos[2])**2
        distance=np.sqrt((position[0]-self.target_pos[0])**2 + (position[1]-self.target_pos[1])**2 + (position[2]-self.target_pos[2])**2)
        if distance<10:
            reward+=500
        reward+=100
        return reward-0.001*negative_reward

    def step(self, rotor_speeds):
        reward=0
        pose_all=[]
        for _ in range(self.action_repeat):
            done=self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward+=self.get_reward() 
            state=np.concatenate([np.array(self.sim.pose), np.array(self.sim.v), np.array(self.sim.angular_v)])
            pose_all.append(self.sim.pose)
        next_state=np.concatenate(pose_all)
        return next_state, reward, done
    
    def current_state(self):
        state=np.concatenate([np.array(self.sim.pose), np.array(self.sim.v), np.array(self.sim.angular_v)])
        return state


    def reset(self):
        self.sim.reset()
        state=np.concatenate([self.sim.pose]*self.action_repeat) 
        return state