from BAYAgent import Agent
import random
import numpy as np
import copy


agent = []
agents = 8
debug = False # Debug mode also prints things like the boundary boxes into the terminal, but this clutters the view quite a lot

# create agents
for i in range(agents):
    agent.append(Agent(random.randint(0,16), random.randint(0,16), random.randint(0,3))) 

# run simulation for a certain timesteps. For me, 15 seems to be the max before the terminal runs out of lines.
# This can be "fixed" by allowing more lines to be strored in scrollback
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
        
        # determine boundaries and add them to the display
        boundries = agent[a].estimate_next_step()
        ngz.append(boundries)
        for b in boundries:
            display[16-(b[1]),b[0]] = 88

        # Print data
        print("pos: " + str(agent[a].position))
        print("EstHead: " + str(agent[a].EstimatedHeading))
        if debug:
            print("BB: " + str(agent[a].boundryBox))
        
    for a in range(len(agent)):
        # Add position and heading to display
        display[16-(agent[a].position[1]),agent[a].position[0]] = (a+1)*10 + agent[a].position[2]
        
    print(display)      
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
        if debug:
            print("NGZ: " + str(agent[a].nogozone))
        agent[a].do_next_step()
    
    # Just for readabilities sake
    print("=================================================")
    print("")
    print("")
    print("")

