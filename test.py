import torch

def test(agent, env, fps):
    input_dim = env.INPUT_SHAPE
    state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
    total_reward = 0
    while(True):
        action = agent.pick_action(state.unsqueeze(0))
        next_state, reward, done, timeout = env.step(action, fps=fps)
        done = done or timeout
        total_reward+=reward
        next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
        state = next_state
        if(done == True): break
    return total_reward, env.plr.size