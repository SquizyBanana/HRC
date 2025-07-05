from BAYAgent import Agent
import random
import numpy as np
import copy


agent = []
agents = 6

# create agents
for i in range(agents):
    agent.append(Agent(random.randint(0,16), random.randint(0,16), random.randint(0,3))) 

# run simulation
for i in range (15):
    # reset display
    display = np.zeros([17,17])
    nogozone = [[]]
    ngz = []
    # run every agent

    # Prepare every movement
    print("=================================================")
    print("Prepping next move")
    for a in range(len(agent)):
        print("=================================================")
        print("agent " + str(a+1))
        
        boundries = agent[a].estimate_next_step()
        ngz.append(boundries)
        for b in boundries:
            display[16-(b[1]),b[0]] = 88

        print("pos: " + str(agent[a].position))
        print("EstHead: " + str(agent[a].EstimatedHeading))
        print("BB: " + str(agent[a].boundryBox))
        
        display[16-(agent[a].position[1]),agent[a].position[0]] = (a+1)*10 + agent[a].position[2]
        
    # Prestep the next movement into memory
    print("=================================================")
    print("Commiting next move")
    for a in range(len(agent)):
        print("=================================================")
        print("agent " + str(a+1))
        # remove own boundry from nogozone list
        nogozone = copy.deepcopy(ngz)
        nogozone.pop(a)

        # Perform step
        agent[a].set_boundries(nogozone)
        agent[a].do_next_step()
        

    print(display)


