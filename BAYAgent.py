# Code ajusted from the coding example:https://canvas.utwente.nl/courses/16751/files/4914677?module_item_id=575019 
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random

from pgmpy.models import DiscreteBayesianNetwork


class Agent:

    def __init__(self, posX, posY, rot):
        
        # Define the network structure
        model = DiscreteBayesianNetwork([
            ("UserIntent", "GazeDirection"),
            ("UserIntent", "PreviousDirection")
        ])

        # Preset variables
        self.dir = "Forwards"
        self.prev_dir = "PrevForwards"
        self.position = [posX, posY, rot]
        self.boundryBox = []
        self.depth = 0


        from pgmpy.factors.discrete import TabularCPD

        # Create nodes

        # Intended next direction
        cpd_user_intent = TabularCPD(
            variable="UserIntent",
            variable_card=4,  # Number of states: forwards, left, right, stop
            values=[[0.7], [0.1], [0.1],[0.1]],  # List of probabilities in the same order as states
            state_names={'UserIntent': ['Forwards', 'Left', 'Right', 'Stop']}
        )

        # Direction that the agent is looking
        cpd_gaze_direction = TabularCPD(
            variable="GazeDirection",
            variable_card=5,  # Gaze can be forwards, left, right, backleft, backright
            values=[
                [0.6, 0.15, 0.15, 0.2],  # Probability Gaze=Forwards
                [0.19, 0.50, 0.04, 0.15],  # Probability Gaze=Left
                [0.19, 0.04, 0.50, 0.15],  # Probability Gaze=Right
                [0.01, 0.3, 0.01, 0.25],  # Probability Gaze=BackLeft
                [0.01, 0.01, 0.3, 0.25],  # Probability Gaze=BackRight
            ],
            evidence=["UserIntent"],
            evidence_card=[4],
            state_names={'GazeDirection': ['GazeForwards', 'GazeLeft', 'GazeRight', 'GazeBackLeft', 'GazeBackRight'], 'UserIntent': ['Forwards', 'Left', 'Right', 'Stop']}
        )

        # Direction that the agent moved in last time step
        cpd_previous_direction = TabularCPD(
            variable="PreviousDirection",
            variable_card=4,  # Forwards, Left, Right, Back
            values=[
                [0.6, 0.4, 0.4, 0.1],  # Probability Prev=Forwards given each UserIntent
                [0.15, 0.4, 0.01, 0.1],  # Probability Prev=Left given each UserIntent
                [0.15, 0.01, 0.4, 0.1],  # Probability Prev=Right given each UserIntent
                [0.1, 0.19, 0.19, 0.7],  # Probability Prev=Stop given each UserIntent
            ],
            evidence=["UserIntent"],
            evidence_card=[4],
            state_names={'PreviousDirection': ['PrevForwards', 'PrevLeft', 'PrevRight', 'PrevStop'], 'UserIntent': ['Forwards', 'Left', 'Right', 'Stop']}
        )

        model.add_cpds(cpd_user_intent, cpd_gaze_direction, cpd_previous_direction)

        # Validate the model
        model.check_model()

        self.inference = VariableElimination(model)


    def posterior(self, variables, evidence):
        return self.inference.query(variables, evidence)
    
    def estimate_next_step(self):

        # Generate gaze behaviour
        self.gaze_int = random.randrange(0, 4, 1)
        if self.gaze_int == 0:
            self.gaze = "GazeForwards"
        elif self.gaze_int == 1:
            self.gaze = "GazeLeft"
        elif self.gaze_int == 2:
            self.gaze = "GazeRight"
        elif self.gaze_int == 3:
            self.gaze = "GazeBackLeft"
        elif self.gaze_int == 4:
            self.gaze = "GazeBackRight" 

        # Posterior based on gaze behaviour
        self.nextArray = self.posterior(["UserIntent"], {"GazeDirection": self.gaze, "PreviousDirection":self.prev_dir}).values
        
        # Store previous direction
        self.prev_dir = "Prev" + self.dir

        # Estimate the most likely next direction
        self.EstimatedNextValue = max(self.nextArray)
        self.EstimatedNext = list(self.nextArray).index(self.EstimatedNextValue)


        # Estimate boundrybox

        # base box
        self.boundryBox = [[self.position[0],self.position[1]],
                           [self.position[0]+1,self.position[1]],
                           [self.position[0]-1,self.position[1]],
                           [self.position[0],self.position[1]+1],
                           [self.position[0],self.position[1]-1]
                           ]
        
        # determine estimation box heading
        if self.EstimatedNext == 1:
            self.EstimatedHeading = self.position[2]-1
            if self.EstimatedHeading < 0:
                self.EstimatedHeading = 3
        elif self.EstimatedNext == 2:
            self.EstimatedHeading = self.position[2]+1
            if self.EstimatedHeading > 3:
                self.EstimatedHeading = 0
        else:
            self.EstimatedHeading = self.position[2]
        
        # append the coordinates of the apropriate spots in the extimated heading 

        if self.EstimatedHeading == 0:
            self.boundryBox.append([self.position[0],self.position[1]+2])       #     O
            self.boundryBox.append([self.position[0]+1,self.position[1]+1])     #   O   O 
            self.boundryBox.append([self.position[0]-1,self.position[1]+1])     #     X
        elif self.EstimatedHeading == 1:
            self.boundryBox.append([self.position[0]+1,self.position[1]+1])     #    O
            self.boundryBox.append([self.position[0]+2,self.position[1]])       #   X  O
            self.boundryBox.append([self.position[0]+1,self.position[1]-1])     #    O  
        elif self.EstimatedHeading == 2:
            self.boundryBox.append([self.position[0],self.position[1]-2])       #     X
            self.boundryBox.append([self.position[0]-1,self.position[1]-1])     #   O   O
            self.boundryBox.append([self.position[0]+1,self.position[1]-1])     #     O
        elif self.EstimatedHeading == 3:
            self.boundryBox.append([self.position[0]-1,self.position[1]+1])     #     O
            self.boundryBox.append([self.position[0]-2,self.position[1]])       #   O   X
            self.boundryBox.append([self.position[0]-1,self.position[1]-1])     #     O
        else:
            print("Error: undefined estimated rotation")
        
        # Wrap around the boundary boxes to the other side of the square
        for i in range(len(self.boundryBox)):
            for j in range(len(self.boundryBox[i])):
                if self.boundryBox[i][j] < 0:
                    self.boundryBox[i][j] = 17 + self.boundryBox[i][j]
                elif self.boundryBox[i][j] > 16:
                    self.boundryBox[i][j] = self.boundryBox[i][j] - 17
        
        return self.boundryBox
    
    def set_boundries(self,boundries):
        # Unpack the list of boundaries per agent into a list of all coordinates this agent is not allowed to go to
        self.nogozone = []
        for i in range(len(boundries)):
            for j in range(len(boundries[i])):
                self.nogozone.append(boundries[i][j])
        
    
    
    def do_next_step(self):
        # Pick the true next state and move there
        picker = random.random()
        self.nextPosition = [self.position[0],self.position[1],self.position[2]]

        # Turn direction into heading
        if picker <= self.nextArray[0]:
            self.dir = "Forwards"
        elif picker <= self.nextArray[0] + self.nextArray[1]:
            self.dir = "Left"
            self.nextPosition[2] = self.position[2] -1
            if self.nextPosition[2] < 0:
                self.nextPosition[2] = 3
        elif picker <= self.nextArray[0] + self.nextArray[1] + self.nextArray[2]:
            self.dir = "Right"
            self.nextPosition[2] = self.position[2] +1
            if self.nextPosition[2] > 3:
                self.nextPosition[2] = 0
        else:
            self.dir = "Stop"        
        
        # break recursion (started when trying to walk into a no-go-zone to find a different direction)
        if self.depth > 10:
            self.dir = "Stop"
            self.depth = 0

        # Print the direction the agent will try to walk into
        print("RealDir " + self.dir)

        # Determine next position based on heading
        if self.dir != "Stop":
            if self.position[2] == 0:
                self.nextPosition[1] = self.position[1] +1
                if self.nextPosition[1] > 16:
                    self.nextPosition[1] = 0
            elif self.position[2] == 1:
                self.nextPosition[0] = self.position[0] +1 
                if self.nextPosition[0] > 16:
                    self.nextPosition[0] = 0
            elif self.position[2] == 2:
                self.nextPosition[1] = self.position[1] -1
                if self.nextPosition[1] < 0:
                    self.nextPosition[1] = 16 
            elif self.position[2] == 3:
                self.nextPosition[0] = self.position[0] -1
                if self.nextPosition[1] < 0:
                    self.nextPosition[1] = 16
            else:
                print("Error: undefined rotation")
        
        # Start recursion to find different space if proposed is invalid
        self.nextpos = [self.nextPosition[0],self.nextPosition[1]] # Remove the rotation
        if self.nextpos in self.nogozone:                
            print("in nogozone")
            self.depth = self.depth + 1
            if self.dir != "Stop":
                print("Deciding: " + str(self.depth))
                # Ajust the result from the baysian network to remove the direction that did not work
                self.rebalanceValues(self.dir) 
        
                self.do_next_step()
        
        # Confirm new position
        self.position[0] = self.nextPosition[0]
        self.position[1] = self.nextPosition[1]
        self.position[2] = self.nextPosition[2]

        return self.EstimatedNext
    
    def rebalanceValues(self,dir):
        # If a direction doesn't work, set its likelyhood to 0 and then rebalance scores to add up to 1 again
        if dir == "Forwards":
            self.nextArray[0] = 0
        elif dir == "Left":
            self.nextArray[1] = 0
        elif dir == "Right":
            self.nextArray[2] = 0
        elif dir == "Stop":
            self.nextArray[3] = 0
        else:
            print("Error: RebalanceValues no valid Direction")
        total = 0
        for i in range(len(self.nextArray)):
            total = total+self.nextArray[i]
        self.nextArray[3] = self.nextArray[3] + (1-total)
        
        print("Rebalanced values: " + str(self.nextArray))