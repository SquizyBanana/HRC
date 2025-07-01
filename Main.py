from BAYAgent import Agent
import random
import numpy as np

agent = []

for i in range(1):
    agent.append(Agent(random.randint(-10,10), random.randint(-10,10), random.randint(0,3))) 

print(agent[0].posterior(["UserIntent"], {"GazeDirection": "GazeLeft"}))
print(agent[0].posterior(["UserIntent"], {"GazeDirection": "GazeLeft", "PreviousDirection":"PrevLeft"}))

print("===============================")
print(agent[0].position)

for i in range (15):
    print(agent[0].next_step())
    print(agent[0].position)
    display = np.zeros([21,21])
    display[-(agent[0].position[1]+11),agent[0].position[0]+10] = 1
    print(display)


