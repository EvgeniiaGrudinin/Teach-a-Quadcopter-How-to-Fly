# Teach-a-Quadcopter-How-to-Fly
In this project, I designed an agent that can fly a quadcopter, and then train it using a reinforcement learning algorithm.
First of all, I defined my task in task.py which represents the straight take-off (the straight until a height defined in the target position).
We include the current speed and angular velocity. We added a threshold for the distance to the target . If the "distance" is below the threshold, we get a reward (reward+=500). Also we added negative reward to control the stability of takeoff and deviation from target.
The agent controls the quadcopter, so I used Actor-Critic model when I designed my agent. For actor.py and for a critic.py I used neural networks.
