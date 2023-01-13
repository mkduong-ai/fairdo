import gym
import numpy as np

'''
This example uses a custom environment BlackboxEnv that takes in the dimension of the binary vector as a parameter, and implements the step and reset methods required by the Gym API. The step method updates the state of the environment based on the action taken, and returns the new state, a reward, and whether the episode is done. The reset method sets the initial state of the environment to a random binary vector.

In this example, I'm assuming you already have a RL agent class named SomeRLAgentClass that you want to use to solve this problem. This class should implement the following methods:

get_action(obs): takes in the current observation and returns an action
update(obs, action, reward, done): takes in the current observation, the action taken, the reward received, and whether the episode is done, and updates the agent's internal state.
The train_agent function trains the agent by repeatedly resetting
'''

class BlackboxEnv(gym.Env):
    def __init__(self, d):
        self.d = d
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(d,), dtype=np.int8)
        self.action_space = gym.spaces.Discrete(2)
    
    def step(self, action):
        self.x[action] = 1 - self.x[action]
        reward = -f(self.x)
        return self.x, reward, True, {}
    
    def reset(self):
        self.x = np.random.randint(2, size=self.d)
        return self.x
    
def create_agent(d, agent_class):
    env = BlackboxEnv(d)
    agent = agent_class(env.observation_space, env.action_space)
    return agent

def train_agent(agent, env, num_steps):
    for i in range(num_steps):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, done)
    return agent

def test_agent(agent, env, num_steps):
    total_reward = 0
    for i in range(num_steps):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward / num_steps

d = 10  # dimension of the binary vector
agent = create_agent(d, SomeRLAgentClass)
trained_agent = train_agent(agent, env, num_steps=1000)
average_reward = test_agent(trained_agent, env, num_steps=100)

# The agent's average reward should be close to the minimum value of the blackbox function
