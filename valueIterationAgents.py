# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        self.values = self.valueIterate(self)

    def valueIterate(self):
        iterator = 0
        while iterator < range(self.iterations):  # while loop as we don't utilize the extra varible in a for loop

            temp_values = util.counter()  # Temporary counter to maintain a dict of values associated to states which will be discovered below

            for state in self.mdp.getStates():  # Get each state so that we can perform calculations on it

                if not self.mdp.isTerminal(state):  # If the state is the terminal (end point)

                    values = []  # Array to store the q-values associated to states and actions
                    possible_actions = self.mdp.getPossibleActions(
                        state)  # possible actions that can be done in a state
                    # maximum_value = 0 # Varible to store the maximum reward found in all actions
                    for action in possible_actions:
                        values.append(self.computeQValueFromValues(state,
                                                                   action))  # create a list of all the q-values for all possible moves

                    print(values)
                    temp_values[state] = max(values)  # find the maximum value for all possible moves

                else:  # if the state is terminal, we should return 0?
                    temp_values[state] = 0

        return temp_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Q^pi = Set(Rt+n * gamma ^ n ) * (St, aT)
        # Q = (r + g) * (T) where G = Discount * values
        # need to get ...
        #   r = reward
        #   g = d * v
        #   d = discount (gamma)
        #   v = values
        #   T = prob
        "*** YOUR CODE HERE ***"
        values = [] # Array to store q-values associated to states and actions
        next_state = self.mdp.getTransitionStatesandProbs(state, action) # Get the next state probabilites and actions 

        for state_prime, probs in next_state: # Following the equation stated from the textbook
            d = self.discount
            v = self.values
            g = d * v
            r = self.mdp.getReward(state, action, state_prime)
            q_value = probs * (r + g)
            values.append(q_value)
        
        return sum(values) # return the sum of the array (this is the set E)




        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        import numpy as np
        possible_actions = self.mdp.getPossibleActions(state)

        if not possible_actions: # If there are no possible actions, return None.
            return None
        else:
            values = []
            actions = []
            for action in possible_actions:
                q_value = self.computeQValueFromValues(state, action)
                values.append(q_value)
                actions.append(action)

            max_action = np.argmax(values, axis=0) #find the

            return actions[max_action]


        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
