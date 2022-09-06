from RandomAgent import RandomAgent

from common.gridworld import GridWorld

episodes = 1000

env = GridWorld()
agent = RandomAgent()

for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step()

        agent.add(state, action, reward)
        if done:
            agent.eval()
            break

        state = next_state
