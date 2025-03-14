import torch

def train(agent, env, checkpoint=2000):
    input_dim = env.INPUT_SHAPE

    for i in range(agent.num_episode):
        state = torch.tensor(env.get_frames().clone().detach()).reshape(input_dim)
        total_reward = 0
        while(True):
            action = agent.pick_action(state.unsqueeze(0))
            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state.clone().detach()).reshape(input_dim)
            total_reward+=reward

            agent.remember(state, action, total_reward, next_state, done)
            agent.replay()
            agent.update_values()

            state = next_state
            if(done == True): break

        print(f"ep {i+1}: {total_reward}")
        agent.add_reward(total_reward)
        agent.add_score(env.plr.size)
        env.reset()
        
        if((i+1)%checkpoint == 0):
            agent.save_model(f"snake_ep_{i+1}")
            agent.save_reward()
            agent.save_score()

