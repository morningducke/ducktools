import numpy as np
from numpy.random import randint, rand, choice

class GA:
    """Implements a basic genetic algorithm for optimization.

    Parameters:
        n_bits: The number of bits in each chromosome.
        n_pop: The initial population size.
        n_gen: The number of generations to run.
        fitness: The fitness function to evaluate each chromosome.
        bounds: A list of lists specifying the bounds for each variable. For example, `[[0, 1], [-1, 2]]` represents two variables, one with bounds from 0 to 1 and another with bounds from -1 to 2.
        cross_prob: The probability of crossover.
        mut_prob: The probability of mutation.
        select: The selection method to use, either "roulette" or "tourn".

    Attributes:
        fitness: The fitness function used to evaluate chromosomes.
        bounds: The bounds for each variable.
        scores: The fitness values of all chromosomes in the population.
        best_chromosome: The chromosome with the best fitness value.

    Methods:
        _to_int(bitstring): Converts a bitstring to an integer.
        _decode(bitstring): Decodes a bitstring into a list of values within the specified bounds.
        _selection_tournament(): Selects a chromosome using a tournament selection method.
        _selection_roulette_stochacc(max_fitness, scores_scaled): Selects a chromosome using a roulette wheel selection method with stochastic acceptance.
        _selection_roulette(selection_probs): Selects a chromosome using a roulette wheel selection method.
        _crossover(p1, p2): Performs crossover on two chromosomes.
        _mutation(child): Performs mutation on a chromosome.
        _step(): Performs one step of the genetic algorithm, including selection, crossover, and mutation.
        fit(): Runs the genetic algorithm for the specified number of generations and returns the best chromosome and its fitness.
        get_best(): Returns the best chromosome and its fitness.
    """

    def __init__(self, n_bits, n_pop, n_gen, fitness, bounds, cross_prob, mut_prob, select: str, eps = 1e-6):
        self.n_bits_ = n_bits
        self.pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
        self.n_gen = n_gen
        self.fitness = fitness
        self.bounds = bounds
        self.scores = self.pop.copy()
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.best_chromosome = self.pop[np.argmin(self.scores)]
        self.select = select
        self.eps = eps

    def _to_int(self, bitstring):
        """Converts a bitstring to an integer.

        Parameters:
            bitstring: A list of 0s and 1s.

        Returns:
            The integer representation of the bitstring.
        """
        res = 0
        pow = len(bitstring) - 1
        for bit in bitstring:
            res += bit * (2 ** pow)
            pow -= 1
        return res  
        
    def _decode(self, bitstring):
        """Decodes a bitstring into a list of values within the specified bounds.

        Parameters:
            bitstring: A list of 0s and 1s.

        Returns:
            A list of decoded values.
        """

        decoded = []
        largest = 2 ** self.n_bits_
        for i in range(len(self.bounds)):
            l = i * self.n_bits_
            r = l + self.n_bits_
            val = self._to_int(bitstring[l:r])
            val = self.bounds[i][0] + (val / largest) * (self.bounds[i][1] - self.bounds[i][0])
            decoded.append(val)
        return decoded
        

    def _selection_tournament(self, k=3):
        """Selects a chromosome using a tournament selection method.

        Parameters:
            k: The number of chromosomes to compete in the tournament.

        Returns:
            The index of the chromosome
        """

        participants = randint(0, len(self.pop), k)
        return participants[np.argmin([self.scores[idx] for idx in participants])]

    def _selection_roulette_stochacc(self, max_fitness, scores_scaled):
        """Selects a chromosome using a roulette method with stochastic acceptance.

        Parameters:
            max_fitness: The maximum fitness value in the population.
            scores_scaled: The scaled fitness values of the chromosomes.

        Returns:
            The index of the selected chromosome.
        """
        while True:
            participant = randint(0, len(self.pop))
            if rand() < (scores_scaled[participant] / max_fitness):
                return participant
    
    def _selection_roulette(self, selection_probs):
        """Selects a chromosome using a roulette wheel selection method.

        Parameters:
            selection_probs: A list of probabilities corresponding to the fitness values of the chromosomes.

        Returns:
            The index of the selected chromosome.
        """
        return choice(len(self.pop), p=selection_probs)
        
    def _crossover(self, p1, p2):
        """Performs single-point crossover on two chromosomes.

        Parameters:
            p1: The first chromosome.
            p2: The second chromosome.

        Returns:
            A tuple containing two new chromosomes created by crossover.
        """
        c1, c2 = p1.copy(), p2.copy()
        if rand() < self.cross_prob:
            pt = randint(1, len(p1) - 2) # crossover point
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return c1, c2

    def _mutation(self, child):
        """Performs bit-flip mutation on a chromosome.

        Parameters:
            child: The chromosome to mutate.

        Returns:
            A mutated version of the chromosome.
        """
        return [(1 - bit if rand() < self.mut_prob else bit) for bit in child]
    
        
    def _step(self):
        """Performs one step of the genetic algorithm. This includes selection, crossover, and mutation.

        Returns:
            The best chromosome and its fitness value in the current generation.
        """
        # next gen
        selected = []
        if self.select == "roulette":
            scores_scaled = -(self.scores - np.max(self.scores))/np.ptp(self.scores)
            max_fitness = max(scores_scaled)
            selected = [self._selection_roulette_stochacc(max_fitness, scores_scaled) for _ in range(len(self.pop))]
        elif self.select == "tourn":
            selected = [self._selection_tournament() for _ in range(len(self.pop))]

        children = []
        for i in range(0, len(self.pop), 2):
            for c in self._crossover(self.pop[selected[i]], self.pop[selected[i + 1]]):
             children.append(self._mutation(c))  
        self.pop = children
        
        decoded = [self._decode(p) for p in self.pop]
        self.scores = [self.fitness(d) for d in decoded]
        # best chromosome in gen and its eval
        cur_best = np.argmin(self.scores)
        return self.pop[cur_best], self.scores[cur_best]

    def fit(self):
        """Runs the genetic algorithm for the specified number of generations and returns the best chromosome and its fitness.

        Returns:
            The best chromosome and its fitness.
        """
        decoded = [self._decode(p) for p in self.pop]
        self.scores = [self.fitness(d) for d in decoded]
        history = [(np.min(self.scores), self._decode(self.best_chromosome))]
        stop_counter = 0
        for i in range(1, self.n_gen):
            # if no improvement in the last ten generations -> stop
            if stop_counter == 10:
                break
                
            fit = self.fitness(self._decode(self.best_chromosome))
            chromosome, eval = self._step()
            if fit - eval > self.eps:
                self.best_chromosome = chromosome
                stop_counter = 0
            else:
                stop_counter += 1
            history.append((eval, self._decode(chromosome)))

        decoded_best = self._decode(self.best_chromosome)
        return decoded_best, self.fitness(decoded_best), history   

    def get_best(self):
        """Returns the best chromosome and its fitness.

        Returns:
            The best chromosome and its fitness.
        """
        decoded_best = self._decode(self.best_chromosome)
        return decoded_best, self.fitness(decoded_best)