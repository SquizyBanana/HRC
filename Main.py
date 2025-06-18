from BAYAgent import Agent
agent = []

for i in range(10):
    agent.append(Agent()) 

print(agent[1].posterior(["UserIntent"], {"GazeDirection": "GazeLeft"}))
print(agent[1].posterior(["UserIntent"], {"GazeDirection": "GazeLeft", "PreviousDirection":"PrevLeft"}))