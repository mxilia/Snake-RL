import torch

def test_a2c(agent, env):
    input_dim = env.INPUT_SHAPE
    state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
    while(True):
        action_probability, value = agent.act(state.unsqueeze(0))
        action_distribution = torch.distributions.Categorical(action_probability)
        action = action_distribution.sample()
        next_state, reward, done = env.step(action, fps=60)
        next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
        state = next_state
        if(done == True): break
    return

def test_dqn(agent, env):
    input_dim = env.INPUT_SHAPE
    state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
    while(True):
        action = agent.pick_action(state.unsqueeze(0))
        next_state, reward, done = env.step(action, fps=60)
        next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
        state = next_state
        if(done == True): break
    return