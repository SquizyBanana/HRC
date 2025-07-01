from BAYAgent import Agent
import random
import numpy as np

agent = []
# create agents
for i in range(3):
    agent.append(Agent(random.randint(0,16), random.randint(0,16), random.randint(0,3))) 

# run simulation
for i in range (15):
    # reset display
    display = np.zeros([17,17])
    nogozone = [[]]
    ngz = []

    # run every agent
    for a in range(len(agent)):
        print("agent " + str(a+1))
        for b in agent[a].estimate_next_step():
            ngz.append(b)
            display[16-(b[1]),b[0]] = 88
        nogozone.append(ngz)
        print(agent[a].position)
        print(agent[a].EstimatedHeading)
        
        # Send all boundries except own
        agent[a].do_next_step((nogozone[:a]+ nogozone[(a+1):]))
        
        display[16-(agent[a].position[1]),agent[a].position[0]] = (a+1)*10 + agent[a].position[2]

    print(display)


