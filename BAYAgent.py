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

        cpd_user_intent = TabularCPD(
            variable="UserIntent",
            variable_card=4,  # Number of states: forwards, left, right, stop
            values=[[0.7], [0.1], [0.1],[0.1]],  # List of probabilities in the same order as states
            state_names={'UserIntent': ['Forwards', 'Left', 'Right', 'Stop']}
        )

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

        # Estimate the most likely next state
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
        
        # determine estimation box direction
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
        
        # append estimation box

        if self.EstimatedHeading == 0:
            self.boundryBox.append([self.position[0],self.position[1]+2])
            self.boundryBox.append([self.position[0]+1,self.position[1]+1])
            self.boundryBox.append([self.position[0]-1,self.position[1]+1])
        elif self.EstimatedHeading == 1:
            self.boundryBox.append([self.position[0]+1,self.position[1]+1])
            self.boundryBox.append([self.position[0]+2,self.position[1]])
            self.boundryBox.append([self.position[0]+1,self.position[1]-1])
        elif self.EstimatedHeading == 2:
            self.boundryBox.append([self.position[0],self.position[1]-2])
            self.boundryBox.append([self.position[0]-1,self.position[1]-1])
            self.boundryBox.append([self.position[0]+1,self.position[1]-1])
        elif self.EstimatedHeading == 3:
            self.boundryBox.append([self.position[0]-1,self.position[1]+1])
            self.boundryBox.append([self.position[0]-2,self.position[1]])
            self.boundryBox.append([self.position[0]-1,self.position[1]-1])
        else:
            print("Error: undefined estimated rotation")
        for i in range(len(self.boundryBox)):
            for j in range(len(self.boundryBox[i])):
                if self.boundryBox[i][j] < 0:
                    self.boundryBox[i][j] = 17 + self.boundryBox[i][j]
                elif self.boundryBox[i][j] > 16:
                    self.boundryBox[i][j] = self.boundryBox[i][j] - 17
        
        return self.boundryBox
    
    def set_boundries(self,boundries):
        self.nogozone = []
        for i in range(len(boundries)):
            for j in range(len(boundries[i])):
                self.nogozone.append(boundries[i][j])
        print("NGZ: " + str(self.nogozone))
    
    
    def do_next_step(self):
        # Pick the true next state and move there
        picker = random.random()
        self.nextPosition = [self.position[0],self.position[1]]

        if picker <= self.nextArray[0]:
            self.dir = "Forwards"
        elif picker <= self.nextArray[0] + self.nextArray[1]:
            self.dir = "Left"
            self.position[2] = self.position[2] -1
            if self.position[2] < 0:
                self.position[2] = 3
        elif picker <= self.nextArray[0] + self.nextArray[1] + self.nextArray[2]:
            self.dir = "Right"
            self.position[2] = self.position[2] +1
            if self.position[2] > 3:
                self.position[2] = 0
        else:
            self.dir = "Stop"
        print("RealDir " + self.dir)
        
        
        # break recursion
        if self.depth > 10:
            self.dir = "Stop"

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
        if self.nextPosition in self.nogozone:                
            print("in nogozone")
            self.depth = self.depth + 1
            print("Deciding: " + str(self.depth))
            self.do_next_step()
        else:
            self.position[0] = self.nextPosition[0]
            self.position[1] = self.nextPosition[1]

        return self.EstimatedNext