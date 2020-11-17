import Reporter
import numpy as np
import random


def alias_setup(probabilities):
    """
    Sets up the tables used in the alias method
    :param probabilities: The probabilities of all the discrete variables to sample from
    :return: Two tables to be used in the alias_draw method
    """
    k = len(probabilities)
    q = np.zeros(k)
    j = np.zeros(k, dtype=np.int)

    # Sort the data into the outcomes with probabilities
    smaller = []
    larger = []
    for kk, prob in enumerate(probabilities):
        q[kk] = k * prob
        if q[kk] < 1:
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop through and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        j[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return j, q


def alias_draw(j, q):
    """
    Draw a random item from the discrete probabilities in constant time.
    :param j: The j matrix from the alias_setup method.
    :param q: The q matrix from the alias_setup method.
    :return: An index from the discrete distribution where for the selection from it, the probabilities are taken into
    account.
    """
    k = len(j)

    # Draw from the overall uniform mixture
    kk = int(np.floor(np.random.rand() * k))

    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]


# TODO: Add local search operators before elimination
# TODO: Add elitism
# TODO: Diversity promotion schemes
# TODO: ANN + Perm4 for initialization
# TODO: OX and SCX recombination
# TODO: Try starting tour always from 0 to make distance more easily computable
# TODO: multiprocessing
# TODO: Dynamic mutation rate
# TODO: Test all combinations of mutation and recombination to find best solution
# TODO: Decaying rank selection based on fitness
# TODO: Convergence criterion
class r0701014:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.distance_matrix = None
        self.tour_size = 0
        self.generation = 0

        # EA parameters
        self.population_size = 100                              # Population size
        self.offspring_size = int(0.5 * self.population_size)   # Offspring size
        self.k = 3                                              # k used in k-tournament
        self.alpha = 0.10                                       # Mutation probability
        self.sigma = 4                                          # Neighborhood distance (hamming)
        self.selection_pressure = 0.01                          # Selection pressure for rank selection
        self.selection_pressure_decay = 0                       # Decay of selection pressure for rank decay

        # EA functions
        self.selection_function = self.selection_roulette_wheel
        self.recombination_function = self.pmx
        self.mutation_function = self.reverse_sequence_mutation
        self.elimination_function = self.lambda_and_mu_elimination

        # EA scores
        self.mean_objective = np.inf
        self.best_objective = 0
        self.best_solution = None
        self.last_mean_objective = 0

        # Function bases logic
        if self.selection_function == self.selection_rank_geometric_decay:
            self.selection_pressure = 0.999

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        self.distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()
        self.tour_size = self.distance_matrix.shape[0] - 1

        self.set_selection_pressure()

        population = self.initialization()

        while not self.is_converged():

            offspring = self.recombination(population)
            joined_population = self.mutation(population, offspring)
            population, scores = self.elimination(joined_population)
            self.update_scores(population[0], scores)
            # self.update_mutation_rate(scores[-1])

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            time_left = self.reporter.report(self.mean_objective, self.best_objective, self.best_solution)
            if time_left < 0:
                break

        # Your code here.
        return 0

    ####################
    #  INITIALIZATION  #
    ####################

    def initialization(self) -> np.array:
        """
        :return: Returns an initial population for the evolutionary algorithm to use. The population is a  numpy array
        of dimensions population_size by tour_size. Each row in the array represents an individual in the population.
        Each individual is a numpy array as well and contains a random permutation of the numbers up to the tour_size.
        This permutation represents the order in which the individual visits the cities.
        """
        population = np.empty([self.population_size, self.tour_size], dtype=np.int32)
        for i in range(self.population_size):
            individual = np.arange(self.tour_size)
            np.random.shuffle(individual)
            individual = self.local_search_2_opt(individual)
            population[i] = individual
        return population

    ###############
    #  SELECTION  #
    ###############

    def selection_k_tournament(self, population: np.array, n: int = 1) -> np.array:
        """
        Performs a k-tournament selection on the population supplied. The k value used is the one provided in self.k.
        :param population: The population to select individuals from.
        :param n: The number of individuals to select
        :return: Returns n individuals in a numpy array
        """
        selection = np.empty((n, self.tour_size))
        for i in range(n):
            individuals = population[np.random.choice(self.population_size, self.k, replace=True), :]
            objective_values = self.length(individuals)
            perm = np.argsort(objective_values)
            selection[i] = individuals[perm[0]]
        return selection.astype('int')

    def selection_roulette_wheel(self, population: np.array, n: int = 1) -> np.array:
        """
        Roulette wheel selection (fitness proportionate selection) using the alias method to draw an index in constant
        time.
        https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        https://www.keithschwarz.com/darts-dice-coins/
        :param population: The population ordered by the fitness of the individuals
        :param n: The number of individuals to select
        :return: The selected individuals in an array
        """
        probabilities = 1 / self.length(population)
        total = np.sum(probabilities)
        probabilities /= total
        j, q = alias_setup(probabilities)
        selection = np.zeros([n, self.tour_size])
        for i in range(n):
            selection[i] = population[int(alias_draw(j, q))]
        return selection.astype('int')

    def selection_rank_linear(self, population: np.array, n: int = 1) -> np.array:
        """
        Selection operator used to select n individuals from the population. This operator uses rank based selection
        with linear decay and constant selection pressure.
        :param population: The population to select the individuals from.
        :param n: The number of individuals to select.
        :return: The selected individuals.
        """
        total_rank = self.population_size * (self.population_size + 1) / 2  # Sum 1..N = N * (N-1) / 2
        probabilities = np.arange(1, self.population_size + 1)[::-1] / total_rank
        return population[np.random.choice(self.population_size, size=n, replace=True, p=probabilities), :]

    def selection_rank_geometric_decay(self, population: np.array, n: int = 1) -> np.array:
        """
        Selection operator used to select n individuals from the population. This operator uses rank based selection
        with exponential decay and a geometric decaying selection pressure.
        :param population: The population to select the individuals from.
        :param n: The number of individuals to select.
        :return: The selected individuals.
        """
        self.selection_pressure *= self.selection_pressure_decay
        a = np.log(self.selection_pressure) / (self.population_size - 1)
        probabilities = np.exp(a * (np.arange(1, self.population_size + 1) - 1))
        probabilities /= np.sum(probabilities)
        return population[np.random.choice(self.population_size, size=n, replace=True, p=probabilities)]

    ###################
    #  RECOMBINATION  #
    ###################

    def recombination(self, population: np.array) -> np.array:
        """
        The method selects two parents for the recombination operator using the selection method specified in
        self.selection_function and supplies the selected parents to the recombination operator specified in
        self.recombination_function.
        :param population: The population to select the parents from.
        :return: Returns the offspring created by the recombination operator. The number of offspring created is
        specified by self.offspring_size.
        """
        offspring = np.empty([self.offspring_size, self.tour_size])
        selection = self.selection_function(population, self.offspring_size)
        for i in range(0, self.offspring_size, 2):
            offspring[i], offspring[i + 1] = self.recombination_function([selection[i], selection[i + 1]])

        return offspring.astype('int')

    def pmx(self, individuals):
        parent1 = individuals[0]
        parent2 = individuals[1]
        child1 = parent1.copy()
        child2 = parent2.copy()
        p1 = np.zeros(self.tour_size, dtype=np.int)
        p2 = np.zeros(self.tour_size, dtype=np.int)

        # Initialize the position of each indices in the individuals
        for i in range(self.tour_size):
            p1[child1[i]] = i
            p2[child2[i]] = i
        # Choose crossover points
        # print('size ' + str(size))
        cxpoint1 = random.randint(0, self.tour_size)
        cxpoint2 = random.randint(0, self.tour_size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        # print('slicing between ' + str(cxpoint1) + ' and ' + str(cxpoint2))
        # Apply crossover between cx points
        for i in range(cxpoint1, cxpoint2):
            # Keep track of the selected values
            temp1 = child1[i]
            temp2 = child2[i]
            # Swap the matched value
            child1[i], child1[p1[temp2]] = temp2, temp1
            child2[i], child2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        return child1, child2

    ##############
    #  MUTATION  #
    ##############

    def mutation(self, population: np.array, offspring: np.array) -> np.array:
        """
        This method selects individuals to be mutated. Each individual has a self.alpha percent chance to be mutated.
        In order to mutate the selected individuals, the mutation function in self.mutation_function is used. After
        mutation the ne population is returned.
        :param population: The population from the previous generation.
        :param offspring: The offspring created in the current generation.
        :return: Returns the combined population from the original population and the offspring in which some
        individuals are mutated.
        """
        joined_population = np.vstack((population, offspring))
        mask = np.random.random(joined_population.shape[0]) < self.alpha
        mutated_population = self.mutation_function(joined_population[mask])
        return np.vstack((joined_population[~mask], mutated_population))

    def swap(self, population: np.array) -> np.array:
        """
        The swap mutation swaps two random cities in the permutation with each other.
        :param population: The population to be mutated.
        :return: Returns the mutated population.
        """
        random_indexes = np.floor(np.random.random([population.shape[0], 2]) * self.tour_size).astype('int')
        for i, individual in enumerate(population):
            individual[random_indexes[i, 0]], individual[random_indexes[i, 1]] = individual[random_indexes[i, 1]], \
                                                                                 individual[random_indexes[i, 0]]
        return population

    def reverse_sequence_mutation(self, population: np.array) -> np.array:
        """
        This mutation operator chooses two random points in the sequence and flips the subsequence between these
        two crossover points. This operator is described in the research paper linked in the HRPM method.
        :param population: The population to mutate.
        :return: The mutated population.
        """
        points = np.floor(np.random.random([population.shape[0], 2]) * self.tour_size).astype('int')
        for i, individual in enumerate(population):
            if points[i, 0] < points[i, 1]:
                a, b = points[i]
            else:
                b, a = points[i]
            population[i] = np.hstack([individual[:a], np.flip(individual[a:b]), individual[b:]])
        return population

    def partial_shuffle_mutation(self, population: np.array) -> np.array:
        """
        This mutation operator chooses two random points in the sequence and randomly shuffles the subsequence between
        these two crossover points. This operator is described in the research paper linked in the HRPM method.
        :param population: The population to mutate.
        :return: The mutated population.
        """
        points = np.floor(np.random.random([population.shape[0], 2]) * self.tour_size).astype('int')
        for i, individual in enumerate(population):
            if points[i, 0] < points[i, 1]:
                a, b = points[i]
            else:
                b, a = points[i]
            population[i] = np.hstack([individual[:a], np.random.permutation(individual[a:b]), individual[b:]])
        return population

    def hybridizing_psm_rsm_mutation(self, population: np.array) -> np.array:
        """
        A combination of the PSM and RSM mutation operator described in the following research paper:
        https://www.researchgate.net/publication/282732991_A_New_Mutation_Operator_for_Solving_an_NP-Complete_Problem_Travelling_Salesman_Problem
        :param population: The population to mutate
        :return: The mutated population
        """
        points = np.floor(np.random.random([population.shape[0], 2]) * self.tour_size).astype('int')
        for i, individual in enumerate(population):
            if points[i, 0] < points[i, 1]:
                a, b = points[i]
            else:
                b, a = points[i]
            while a < b:
                individual[a], individual[b] = individual[b], individual[a]
                if np.random.random() < self.alpha:
                    c = np.random.randint(0, self.tour_size)
                    individual[a], individual[c] = individual[c], individual[a]
                a += 1
                b -= 1
            population[i] = individual
        return population

    #################
    #  ELIMINATION  #
    #################

    def elimination(self, joined_population: np.array) -> (np.array, np.array):
        return self.elimination_function(joined_population)

    def lambda_and_mu_elimination(self, joined_population: np.array) -> (np.array, np.array):
        """
        Performs (lambda + mu)-elimination on the population and offspring (joined population). The number of
        individuals returned is equal to self.population_size.
        :param joined_population: The joined population (population of the previous generation and offspring of the
        current generation).
        :return: A new population ordered by their fitness in ascending order and the scores of the individuals in this
        new population.
        """
        objective_values = self.length(joined_population)
        perm = np.argsort(objective_values)
        return joined_population[perm[:self.population_size]], objective_values[perm[:self.population_size]]

    def elitism(self, population: np.array) -> np.array:
        """
        This scheme is applied to the population after elimination to make sure that the fittest member of the previous
        generation will also be part of the new generation (unless a better individual is already present in the current
        generation.
        :param population: The population to potentially add the fittest member of the previous generation to.
        :return: The (altered) population.
        """
        if (population[0] != self.best_solution).any() and self.best_solution is not None:
            if self.length_individual(population[0]) > self.best_objective:
                population = np.concatenate(([self.best_solution], population[:-1]))

        else:
            return population

    ########################
    #  OBJECTIVE FUNCTIONS #
    ########################

    def length(self, individuals: np.array) -> np.array:
        """
        :param individuals: The population array of which to compute the objective function.
        :return: Returns the objective function of all the individuals in the given array. The result is a numpy array
        with all the objective values.
        """
        return np.apply_along_axis(self.length_individual, 1, individuals)

    def length_individual(self, individual: np.array) -> float:
        """
        :param individual: The individual to compute the objective function from.
        :return: A floating point number indicating the value of the objective function of the given individual.
        """
        distance = 0
        for i in range(self.tour_size - 1):
            distance += self.distance_matrix[individual[i]][individual[i + 1]]
        distance += self.distance_matrix[individual[-1]][self.tour_size]
        distance += self.distance_matrix[self.tour_size][individual[0]]
        return distance

    #########################
    #  DIVERSITY PROMOTION  #
    #########################

    def distances(self, individual: np.array, population: np.array) -> np.array:
        pass

    def distance(self, perm1, perm2: list):
        distance = 0
        for i in range(self.tour_size - 1):
            element = perm1[i]
            x = perm2.index(element)

            y = 0 if x == self.tour_size - 1 else x + 1
            if perm1[i + 1] != perm2[y]:
                distance += 1

        element = perm1[-1]
        x = perm2.index(element)
        if perm1[0] != perm2[x + 1]:
            distance += 1

        return distance

    def distance_hamming(self, perm1: np.array, perm2: np.array) -> int:
        """
        Returns the hamming distance between two permutations (number of elements that are different).
        :param perm1: The first permutation.
        :param perm2: The second permutation.
        :return: The hamming distance between two permutations (number of elements that are different).
        """
        return np.count_nonzero(perm1 != perm2)

    ############################
    #  LOCAL SEARCH OPERATORS  #
    ############################

    def local_search_2_opt(self, individual: np.array) -> np.array:
        improved_tour = np.array(individual, copy=True)
        k = 0
        while k <= 5:
            for i in np.arange(self.tour_size - 1):
                city1 = individual[i]
                city2 = individual[i + 1]
                distance_12 = self.distance_matrix[city1][city2]
                for j in np.arange(i + 2, self.tour_size - 1):
                    city3 = individual[j]
                    city4 = individual[j + 1]
                    distance_13 = self.distance_matrix[city1][city3]
                    distance_34 = self.distance_matrix[city3][city4]
                    distance_24 = self.distance_matrix[city2][city4]
                    if distance_12 + distance_34 > distance_13 + distance_24:
                        improved_tour[i + 1], improved_tour[j] = improved_tour[j], improved_tour[i + 1]
                if self.length_individual(improved_tour) > self.length_individual(individual):
                    improved_tour = individual
                else:
                    individual = improved_tour
            k += 1
        return improved_tour

    ######################
    #  HELPER FUNCTIONS  #
    ######################

    def is_converged(self) -> bool:
        """
        :return: Returns True if the algorithm has converged, False if not.
        """
        # All solutions are the same
        # if self.mean_objective == self.best_objective:
        #     return True

        # # No improvement
        if self.last_mean_objective == self.mean_objective and self.generation > 1000:
            return True

        # Not converged
        else:
            return False
        # return False

    def update_scores(self, individual: np.array, scores: np.array) -> None:
        """
        Updates the best and mean objective value according to the new population. Also sets the new best solution in the
        population. The number of the generation is also updated.
        :param individual: The best individual from the entire population
        :param scores: The scores from all the individuals in the population, ordered in ascending order.
        """
        self.last_mean_objective = self.mean_objective
        self.mean_objective = np.mean(scores)
        self.best_objective = scores[0]
        self.best_solution = np.concatenate([[self.tour_size], individual])
        self.generation += 1
        if self.generation % 1500 == 0:
            self.alpha = max(0.01, self.alpha - 0.01)

    def set_selection_pressure(self) -> None:
        """
        Sets the selection pressure according to the problem size. This makes it that the algorithm should not converge
        too quickly.
        """
        if self.selection_function == self.selection_rank_geometric_decay:
            self.selection_pressure = float('0.' + '9' * int(self.tour_size / 100 + 3))
            self.selection_pressure_decay = self.selection_pressure
        else:
            self.selection_pressure = 0.01

    def update_mutation_rate(self, worst_score: float) -> None:
        """
        Sets the new mutation rate (alpha) based on the best, worst and average objective value on this point
        :param worst_score: The worst objective score of this generation.
        """
        self.alpha = 0.1 * (self.best_objective - worst_score) / (self.best_objective - self.mean_objective)


TSP = r0701014()
TSP.optimize('tour29.csv')
