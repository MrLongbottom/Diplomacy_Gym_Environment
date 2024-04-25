import os.path
from diplomacy_gym_environment import DiplomacyEnvironment
from diplomacy_gym_simple_rl_agent import DiplomacySimpleRlAgent

# This is a template setup on how to actually use the Gym environment I made, by setting up a RL agent.



# This will run through one game taking random actions using a neural network-like output in order to test the gym environment
if __name__ == '__main__':
    env = DiplomacyEnvironment(prints=False, render_path=None)
    agent = DiplomacySimpleRlAgent(env)
    for i in range(100):
        agent.play()
