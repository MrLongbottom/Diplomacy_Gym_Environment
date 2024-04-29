import os.path
from tqdm import tqdm
from diplomacy_gym_environment import DiplomacyEnvironment
from diplomacy_gym_simple_rl_agent import DiplomacySimpleRlAgent

if __name__ == '__main__':
    env = DiplomacyEnvironment(prints=False, render_path=None)
    agent = DiplomacySimpleRlAgent(env, use_nn_states=False)
    pbar = tqdm(range(100), position=1, leave=True)
    should_render = False
    render = False
    for i in pbar:
        pbar.set_description(f"Game {i}")
        if should_render:
            render = i != 0 and i % 25 == 0
        agent.play(render=render)
