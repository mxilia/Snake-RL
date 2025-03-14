import torch

def test(agent, env):
    input_dim = env.INPUT_SHAPE
    state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
    while(True):
        action = agent.pick_action(state.unsqueeze(0))
        next_state, reward, done = env.step(action, fps=60)
        next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
        state = next_state
        if(done == True): break
    return