# Code ajusted from the coding example:https://canvas.utwente.nl/courses/16751/files/4914677?module_item_id=575019 
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random

from pgmpy.models import DiscreteBayesianNetwork


class Agent:

    def __init__(self):
        
        # Define the network structure
        model = DiscreteBayesianNetwork([
            ("UserIntent", "GazeDirection"),
            ("UserIntent", "PreviousDirection")
        ])
        self.dir = "forwards"

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
    
    def next_step(self):

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
        self.next = self.posterior(["UserIntent"], {"GazeDirection": self.gaze, "PreviousDirection":self.prev_dir}).values
        

        # Store previous direction
        self.prev_dir = self.dir
        return self.next