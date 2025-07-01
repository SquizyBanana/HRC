from BAYAgent import Agent
import random
import numpy as np

agent = []
# create agents
for i in range(2):
    agent.append(Agent(random.randint(-8,8), random.randint(-8,8), random.randint(0,3))) 

# run simulation
for i in range (15):
    # reset display
    display = np.zeros([17,17])

    # run every agent
    for a in range(len(agent)):
        print("agent " + str(a+1))
        print(agent[a].next_step())
        print(agent[a].position)
        display[-(agent[a].position[1]+9),agent[a].position[0]+8] = (a+1)*10 + agent[a].position[2]
    print(display)


