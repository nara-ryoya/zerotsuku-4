from McAgent import McAgent

from common.gridworld import GridWorld

episodes = 10000

env = GridWorld()
agent = McAgent()

for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action=action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)
