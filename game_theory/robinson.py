import numpy as np
from numpy.random import randint, choice

class RobinsonSolver:
    """Robinson's iterative algorithm for solving two players zero-sum game
    Note: slow convergence
    
    Parameters:
        pay_matrix: A 2D NumPy array representing the payoff matrix of the game.
        max_iters: The maximum number of iterations to run the algorithm.
    """
    class IterationState:
        """An internal class to store the state of the algorithm at each iteration.

        Attributes:
            iter: The current iteration number.
            _pay_matrix: The reference to the original payoff matrix.
            chosen_A: The index of the strategy chosen by player A in the current iteration.
            outcome_A: The vector of payoffs for player A for each of player B's strategies.
            chosen_B: The index of the strategy chosen by player B in the current iteration.
            outcome_B: The vector of payoffs for player B for each of player A's strategies.
            low_V: The lower bound on the value of the game.
            high_V: The upper bound on the value of the game.
            avg_V: The average of the lower and upper bounds.
            counter_A: A vector of counts for the number of times each strategy was chosen by player A.
            counter_B: A vector of counts for the number of times each strategy was chosen by player B.

        Parameters:
            pay_matrix: A 2D NumPy array representing the payoff matrix of the game.
        """

        def __init__(self, pay_matrix):
            self.iter = 1
            self._pay_matrix = pay_matrix

            # Initialize the outcome vectors and chosen strategies
            self.chosen_A = randint(self._pay_matrix.shape[0]) 
            self.outcome_A = self._pay_matrix[self.chosen_A].copy()
            self.chosen_B = choice(np.flatnonzero(self.outcome_A == self.outcome_A.min()))
            self.outcome_B = self._pay_matrix[:, self.chosen_B].copy()

            # Initialize the bounds and average value
            self.low_V = self.outcome_A[self.chosen_B]
            self.high_V = np.max(self.outcome_B)
            self.avg_V = (self.low_V + self.high_V) / 2

            # Initialize the strategy counts
            self.counter_A = np.zeros(self._pay_matrix.shape[0])
            self.counter_B = np.zeros(self._pay_matrix.shape[1])
            self.counter_A[self.chosen_A] += 1
            self.counter_B[self.chosen_B] += 1

        def step(self):
            """Performs one step of the Robinson's iterative algorithm."""
            self.iter += 1
            
            # Update the chosen strategies and outcome vectors
            self.chosen_A = choice(np.flatnonzero(self.outcome_B == self.outcome_B.max()))
            self.outcome_A += self._pay_matrix[self.chosen_A]
            self.chosen_B = choice(np.flatnonzero(self.outcome_A == self.outcome_A.min()))
            self.outcome_B += self._pay_matrix[:, self.chosen_B]

            # Update the bounds and average value
            self.low_V = self.outcome_A[self.chosen_B] / self.iter
            self.high_V = np.max(self.outcome_B) / self.iter
            self.avg_V = (self.low_V + self.high_V) / 2

            # Update the strategy counts
            self.counter_A[self.chosen_A] += 1
            self.counter_B[self.chosen_B] += 1
            
        def get_probs(self):
            """Returns the mixed strategies for players A and B.

            Returns:
                A tuple containing the mixed strategies for players A and B, respectively.
            """
            return self.counter_A / self.iter, self.counter_B / self.iter
        
        def to_list(self):
            """Converts the IterationState object to a list of values.

            Returns:
                A list of values representing the current state of the algorithm.
            """
            return [self.iter, self.chosen_A + 1, list(np.round(self.outcome_A, 3)), self.chosen_B + 1, list(np.round(self.outcome_B, 3)), round(self.low_V, 3), round(self.high_V, 3), round(self.avg_V, 3)]
            
        
    def __init__(self, pay_matrix, max_iters=50):
        self.pay_matrix = pay_matrix
        self.max_iters = max_iters 
        
    def solve(self):
        """Runs the Robinson's algorithm for the set amount of 'max_iters'.
        
        Returns:
            Last iteration state and a tuple containing the mixed strategies for players A and B, respectively.
        """
        self.history = []
        self.iter_state = self.IterationState(self.pay_matrix)
        self.history.append(self.iter_state.to_list())
        for _ in range(self.max_iters - 1):
            self.iter_state.step()
            self.history.append(self.iter_state.to_list())
        return self.history[-1], self.iter_state.get_probs()