import ctypes
from functools import partial
from multiprocessing import Pool, RawArray
from typing import Tuple

import numpy as np
import Reporter

# Dictionary to store the location of the arrays used by the parallel processes
var_dict: dict = {}


def distance_function_parallel(perm2: np.array, perm1: np.array) -> float:
    """
    Computes the distance between two permutations by counting the edges of the
    first permutation that are not in the second permutation.
    :param perm1: The first permutation
    :param perm2: The second permutation
    :return: The distance between the two permutations
    """
    tour_size = var_dict["tour_size"]
    distance = 0
    for i in range(tour_size - 1):
        element = perm1[i]
        x = np.where(perm2 == element)[0][0]
        y = 0 if x == tour_size - 1 else x + 1
        if perm1[i + 1] != perm2[y]:
            distance += 1
    element = perm1[-1]
    x = np.where(perm2 == element)[0][0]
    if perm1[0] != perm2[(x + 1) % tour_size]:
        distance += 1

    return distance


def parallel_3_opt(individual: np.array) -> np.array:
    """
    This function performs local search on the individual. It can be performed in parallel by using the RawArrays from
    the multiprocessing library. This array cannot be locked. The location of the array can be accessed from the global
    dictionary var_dict.
    :param individual: The individual to perform local search on.
    :return: The (improved) individual
    """
    tour_size = var_dict["tour_size"]
    nearest_neighbors = np.frombuffer(
        var_dict["nearest_neighbors"], dtype=np.int
    ).reshape(tour_size, 15)

    individual = np.roll(individual, -np.random.randint(tour_size))
    for point1 in range(tour_size):
        v1 = individual[0]
        v2 = individual[point1 - 1]
        v3 = individual[point1]
        v6 = individual[-1]

        for i, neighbor in enumerate(nearest_neighbors[v2]):
            point2 = np.where(individual == neighbor)[0][0]
            if point2 > point1:
                v4 = individual[point2 - 1]
                v5 = individual[point2]
                individual = check_for_3_opt_move(
                    individual, point1, point2, v1, v2, v3, v4, v5, v6
                )

    return individual


def check_for_3_opt_move(
    individual: np.array,
    point1: int,
    point2: int,
    v1: int,
    v2: int,
    v3: int,
    v4: int,
    v5: int,
    v6: int,
):
    """
    This function checks for a possible 3 opt move for the arcs between endpoints v1-v2, v3-v4 and v5-v6.
    :param individual: The individual to try the 3 opt move on
    :param point1: The first crossing point, the point between v2 and v3
    :param point2: The second crossing point, the point between v4 and v5
    :param v1: The beginning of the first arc
    :param v2: The ending of the first arc
    :param v3: The beginning of the second arc
    :param v4: The ending of the second arc
    :param v5: The beginning of the third arc
    :param v6: The ending of the third arc
    :return: The individual with the 3 opt move applied (if it is better than not)
    """
    tour_size = var_dict["tour_size"]
    distance_matrix = np.frombuffer(
        var_dict["distance_matrix"], dtype=np.float64
    ).reshape(tour_size, tour_size)
    old_distance = (
        distance_matrix[v2][v3] + distance_matrix[v4][v5] + distance_matrix[v6][v1]
    )
    new_distance = (
        distance_matrix[v2][v5] + distance_matrix[v6][v3] + distance_matrix[v4][v1]
    )

    if new_distance < old_distance:
        a = individual[:point1]
        b = individual[point1:point2]
        c = individual[point2:]
        individual = np.concatenate([a, c, b])
    return individual


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


class r0701014:
    def __init__(self) -> None:
        self.reporter = Reporter.Reporter(
            self.__class__.__name__
        )  # The reporter for the results
        self.distance_matrix: np.array = None  # Distance matrix
        self.tour_size: int = 0  # Number of cities in a tour
        self.generation: int = 0  # Current generation
        self.nearest_neighbors = None  # List with the nearest neighbors for each node, used in the 3-opt local search

        # EA parameters
        self.population_size: int = 16  # Number of individuals in population
        self.offspring_size: int = 50  # Number of children created per generation
        self.k: int = 3  # The k used in k-tournament selection
        self.selection_pressure: float = 0.0  # The selection pressure used
        self.selection_pressure_decay = (
            0.0  # The factor for the decay of the selection pressure in geometric decay
        )
        self.alpha: float = 0.15  # The mutation rate
        self.rcl: float = (
            0.1  # Fraction that a solution can be longer than the greedy solution
        )
        self.number_of_nearest_neighbors: int = (
            15  # Number of NN used in the 3 opt local search
        )
        self.sigma: int = (
            0  # Sigma used in the fitness sharing, will be set after tour_size is known
        )
        self.percentage_local_search: float = (
            0.5  # Percentage of individuals to perform local search on
        )

        # Arrays for parallel execution
        self.raw_distance_matrix = (
            None  # RawArray to be shared between parallel processes
        )
        self.raw_nearest_neighbors = (
            None  # RawArray to be shared between parallel processes
        )

        # EA options
        self.use_random_initialization: bool = (
            False  # Use a random initialization instead of heuristic methods
        )
        self.local_search_on_all: bool = (
            False  # Perform local search on the entire population
        )
        self.use_multiple_mutation_operators: bool = (
            False  # Combine different mutation operators
        )

        # EA functions
        self.selection_function = (
            self.selection_roulette_wheel
        )  # Selection function to use
        self.recombination_function = (
            self.order_crossover
        )  # Recombination function to use
        self.mutation_function = (
            self.reverse_sequence_mutation
        )  # Mutation function to use
        self.elimination_function = (
            self.lambda_and_mu_elimination
        )  # Elimination function to use
        self.local_search_operator = (
            self.local_search_optimized_3_opt
        )  # Local search operator to use

        # EA scores
        self.mean_objective: float = (
            np.inf
        )  # Mean objective value of the current generation
        self.best_objective: float = (
            np.inf
        )  # Best objective value of the current generation
        self.best_solution: np.array = None  # Best solution of the current generation
        self.last_mean_objective: float = (
            0  # Mean objective value of the previous generation
        )
        self.last_best_objective: float = (
            0  # Best objective value of the previous generation
        )
        self.same_best_objective: int = (
            0  # Streak where last best objective == current best objective
        )
        self.time_left = 300

        self.set_selection_pressure()  # Depending on the selection function, the selection pressure will be different

    def optimize(self, filename: str):
        """
        The main loop of the genetic algorithm.
        :param filename: The filename of the tour for which a candidate solution needs to be found.
        """
        # Read the distance matrix from file
        data = np.loadtxt(filename, delimiter=",")
        self.tour_size = data.shape[0]
        self.raw_distance_matrix = RawArray(
            ctypes.c_double, self.tour_size * self.tour_size
        )
        self.distance_matrix = np.frombuffer(
            self.raw_distance_matrix, dtype=np.float64
        ).reshape(self.tour_size, self.tour_size)
        np.copyto(self.distance_matrix, data)

        self.sigma = np.floor(0.05 * self.tour_size)
        self.build_nearest_neighbor_list()
        self.init_dictionary()

        population = self.initialize_population()

        while not self.is_converged():
            offspring = self.recombination(population)
            mutated_population = self.mutation(population, offspring)

            # optimized_population = self.local_search(mutated_population)
            optimized_population = self.local_search_parallel(mutated_population)

            # population, scores = self.fitness_sharing_elimination(optimized_population)

            population, scores = self.elimination(optimized_population)
            population, scores = self.eliminate_duplicate_individuals(
                population, scores
            )
            population, scores = self.elitism(population, scores)

            self.update_scores(population[0], scores)

            if self.same_best_objective % 20 == 0 and self.same_best_objective != 0:
                if self.same_best_objective > 40:
                    break
                self.use_random_initialization = True
                population = self.initialize_population()
                population[0] = self.best_solution

            if self.mean_objective != np.inf:
                self.alpha = (
                    0.2
                    * self.mean_objective
                    / (self.best_objective + self.mean_objective)
                )

            time_left = self.reporter.report(
                self.mean_objective, self.best_objective, self.best_solution
            )
            if time_left < 0:
                break

        return 0

    ####################
    #  INITIALIZATION  #
    ####################

    def initialize_population(self) -> np.array:
        """
        :return: Returns an initial population for the evolutionary algorithm to use. The population is a  numpy array
        of dimensions population_size by tour_size. Each row in the array represents an individual in the population.
        Each individual is a numpy array as well and contains a random permutation of the numbers up to the tour_size.
        This permutation represents the order in which the individual visits the cities.
        """
        if not self.use_random_initialization:
            if self.tour_size > self.population_size:
                # Create random heuristic solutions
                population = self.all_nearest_neighbors()
            else:
                # Create all heuristic solutions and randomly generate the rest
                heuristic_population = self.all_nearest_neighbors()
                random_population = self.random_population(
                    self.population_size - self.tour_size
                )
                population = np.concatenate([heuristic_population, random_population])
        else:
            # Completely random population
            population = self.random_population(self.population_size)
        return population

    def all_nearest_neighbors(self) -> np.array:
        """
        Creates an initial population using a greedy algorithm starting from every possible starting point.
        :return: The population of size self.tour_size in case that the tour size is smaller than the population size,
        otherwise the population has size population_size.
        """
        if self.population_size < self.tour_size:
            # Less individuals than the length of the tour
            nn = np.zeros([self.population_size, self.tour_size], dtype=np.int)
            random_tours = np.random.choice(
                np.arange(self.tour_size), self.population_size, replace=False
            )
            minimal_value_index = np.argwhere(
                self.distance_matrix
                == np.min(self.distance_matrix[self.distance_matrix != 0])
            )[0][0]
            if minimal_value_index not in random_tours:
                random_tours[0] = minimal_value_index

            for i, x in enumerate(random_tours):
                nn[i] = self.make_greedy_tour(x)
            return nn
        else:
            # Create all heuristic individuals and fill the rest with random individuals
            nn = np.zeros([self.tour_size, self.tour_size], dtype=np.int)
            for i in range(self.tour_size):
                new_tour = self.make_greedy_tour(i)
                k = 1
                while self.length_individual(new_tour) == np.inf:
                    new_tour = self.make_greedy_tour(i + k)
                    k += 1
                nn[i] = new_tour
            return nn

    def make_greedy_tour(self, i: int) -> np.array:
        """
        The greedy algorithm to create a tour given a starting point i. The algorithm visits the closest city that has
        not yet been visited.
        :param i: The starting point.
        :return: The tour created by the greedy algorithm.
        """
        individual = np.zeros(self.tour_size, dtype=np.int)
        individual[0] = i
        not_used = set(range(self.tour_size))
        not_used.remove(i)
        for j in range(1, self.tour_size, 1):
            minimum_value = np.min(self.distance_matrix[i][list(not_used)])
            nearest_city = np.where(self.distance_matrix[i] == minimum_value)
            for city in nearest_city[0]:
                try:
                    not_used.remove(city)
                    individual[j] = city
                    i = city
                    break
                except KeyError:
                    # The city is already used, because there are multiple minimal values
                    pass
        return individual

    def random_population(self, n: int) -> np.array:
        """
        Creates a random population of size n
        :param n: The size of the population to generate.
        :return: The generated population.
        """
        population = np.empty([n, self.tour_size], dtype=np.int32)
        for i in range(n):
            individual = np.arange(self.tour_size)
            np.random.shuffle(individual)
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
        selection = np.empty((n, self.tour_size), dtype=np.int)
        for i in range(n):
            individuals = population[
                np.random.choice(self.population_size, self.k, replace=True), :
            ]
            objective_values = self.length(individuals)
            perm = np.argsort(objective_values)
            selection[i] = individuals[perm[0]]
        return selection

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
        self.selection_pressure *= self.selection_pressure_decay
        a = np.log(self.selection_pressure) / (self.population_size - 1)
        probabilities = np.exp(a * (np.arange(1, self.population_size + 1) - 1))
        probabilities /= np.sum(probabilities)
        j, q = alias_setup(probabilities)
        selection = np.zeros([n, self.tour_size], dtype=np.int)
        for i in range(n):
            selection[i] = population[int(alias_draw(j, q))]
        return selection

    def selection_rank_linear(self, population: np.array, n: int = 1) -> np.array:
        """
        Selection operator used to select n individuals from the population. This operator uses rank based selection
        with linear decay and constant selection pressure.
        :param population: The population to select the individuals from.
        :param n: The number of individuals to select.
        :return: The selected individuals.
        """
        total_rank = (
            self.population_size * (self.population_size + 1) / 2
        )  # Sum 1..N = N * (N-1) / 2
        probabilities = np.arange(1, self.population_size + 1)[::-1] / total_rank
        return population[
            np.random.choice(
                self.population_size, size=n, replace=True, p=probabilities
            ),
            :,
        ]

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
        offspring = np.empty([self.offspring_size, self.tour_size], dtype=np.int)
        if self.recombination_function != self.sequential_constructive_crossover:
            # Produce two children per two parents
            selection = self.selection_function(population, self.offspring_size)
            for i in range(0, self.offspring_size, 2):
                offspring[i], offspring[i + 1] = self.recombination_function(
                    [selection[i], selection[i + 1]]
                )
        else:
            # Produce one child per two parents
            selection = self.selection_function(population, self.offspring_size * 2)
            for i in range(self.offspring_size):
                offspring[i] = self.recombination_function(
                    [selection[i], selection[i + 1]]
                )

        return offspring

    def pmx(self, individuals):
        """
        Perform partially mapped crossover (PMX) on the parents. This chooses two crossover points at random and copies
        the partial tour between these points to each of the children. The rest of the nodes are then mapped according
        to the structure of both parents to guarantee no duplicate nodes. This algorithm is described in the book of
        Eiben and Smith.
        :param individuals: The two parents to apply the crossover on
        :return: The two children created from the parents.
        """
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
        point1 = np.random.randint(0, self.tour_size)
        point2 = np.random.randint(0, self.tour_size - 1)
        if point2 >= point1:
            point2 += 1
        else:  # Swap the two cx points
            point1, point2 = point2, point1

        # Apply crossover between cx points
        for i in range(point1, point2):
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

    def order_crossover(self, individuals: np.array) -> np.array:
        """
        Order crossover (OX) proposed by Davis.
        https://www.hindawi.com/journals/cin/2017/7430125/
        :param individuals: The two parents to recombine.
        :return: The children created from the two parents.
        """
        parent1 = individuals[0]
        parent2 = individuals[1]
        child1 = np.zeros(self.tour_size)
        child2 = np.zeros(self.tour_size)
        cx1, cx2 = np.sort(np.random.randint(self.tour_size, size=2))
        child1[cx1:cx2] = parent1[cx1:cx2]
        child2[cx1:cx2] = parent2[cx1:cx2]
        used_in_c1 = set(parent1[cx1:cx2])
        used_in_c2 = set(parent2[cx1:cx2])

        sequence_parent1 = []
        sequence_parent2 = []
        i = cx2
        start = True
        while i != cx2 or start:
            start = False
            if parent1[i] not in used_in_c2:
                sequence_parent1.append(parent1[i])
            if parent2[i] not in used_in_c1:
                sequence_parent2.append(parent2[i])
            i = (i + 1) % self.tour_size

        while len(sequence_parent1) != 0:
            child1[i] = sequence_parent2.pop(0)
            child2[i] = sequence_parent1.pop(0)
            i = (i + 1) % self.tour_size

        return child1, child2

    def sequential_constructive_crossover(self, individuals: np.array) -> np.array:
        """
        Sequential constructive crossover (SCX) by Ahmed
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.7771&rep=rep1&type=pdf
        :param individuals: The parents to use
        :return: The child created
        """
        parent1, parent2 = individuals
        child = np.zeros(self.tour_size)
        child[0] = 0
        nodes_used = {0}

        for i in range(1, self.tour_size, 1):
            previous_node = int(child[i - 1])

            node_parent1 = self.find_first_node(previous_node, parent1, nodes_used)
            node_parent2 = self.find_first_node(previous_node, parent2, nodes_used)

            cost_parent1 = self.distance_matrix[previous_node][node_parent1]
            cost_parent2 = self.distance_matrix[previous_node][node_parent2]
            if cost_parent1 < cost_parent2:
                nodes_used.add(node_parent1)
                child[i] = node_parent1

            else:
                nodes_used.add(node_parent2)
                child[i] = node_parent2

        return child

    def find_first_node(
        self, previous_node: int, parent: np.array, nodes_used: set
    ) -> int:
        """
        Finds the node after a given node that is not already used.
        :param previous_node: The node to start from.
        :param parent: The parent sequence in which to search the next possible node.
        :param nodes_used: The nodes that are already used in the child.
        :return: The first possible node found in the parent that matches the two criteria: if comes after previous_node
        and it is not in the nodes_used set.
        """
        index_previous_node = np.where(parent == previous_node)[0][0]
        for i in range(index_previous_node + 1, self.tour_size, 1):
            if parent[i] not in nodes_used:
                return parent[i]

        for i in range(1, self.tour_size):
            if i not in nodes_used:
                return i

        return -1

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
        if not self.use_multiple_mutation_operators:
            mutated_population = self.mutation_function(joined_population[mask])
            return np.vstack((joined_population[~mask], mutated_population))
        else:
            probabilities = [0.50, 0.50]  # RSM, PSM, HRPM
            number_of_individuals = joined_population[mask].shape[0]
            first = round(number_of_individuals * probabilities[0])

            p1 = self.reverse_sequence_mutation(joined_population[mask][:first])
            p2 = self.hybridizing_psm_rsm_mutation(joined_population[mask][first:])
            return np.vstack((joined_population[mask], p1, p2))

    def swap(self, population: np.array) -> np.array:
        """
        The swap mutation swaps two random cities in the permutation with each other.
        :param population: The population to be mutated.
        :return: Returns the mutated population.
        """
        random_indexes = np.floor(
            np.random.random([population.shape[0], 2]) * self.tour_size
        ).astype("int")
        for i, individual in enumerate(population):
            individual[random_indexes[i, 0]], individual[random_indexes[i, 1]] = (
                individual[random_indexes[i, 1]],
                individual[random_indexes[i, 0]],
            )
        return population

    def reverse_sequence_mutation(self, population: np.array) -> np.array:
        """
        This mutation operator chooses two random points in the sequence and flips the subsequence between these
        two crossover points. This operator is described in the research paper linked in the HRPM method.
        :param population: The population to mutate.
        :return: The mutated population.
        """
        points = np.floor(
            np.random.random([population.shape[0], 2]) * self.tour_size
        ).astype("int")
        for i, individual in enumerate(population):
            if points[i, 0] < points[i, 1]:
                a, b = points[i]
            else:
                b, a = points[i]

            individual = np.hstack(
                [individual[:a], np.flip(individual[a:b]), individual[b:]]
            )
            population[i] = individual
        return population

    def partial_shuffle_mutation(self, population: np.array) -> np.array:
        """
        This mutation operator chooses two random points in the sequence and randomly shuffles the subsequence between
        these two crossover points. This operator is described in the research paper linked in the HRPM method.
        :param population: The population to mutate.
        :return: The mutated population.
        """
        points = np.floor(
            np.random.random([population.shape[0], 2]) * self.tour_size
        ).astype("int")
        for i, individual in enumerate(population):
            if points[i, 0] < points[i, 1]:
                a, b = points[i]
            else:
                b, a = points[i]
            population[i] = np.hstack(
                [individual[:a], np.random.permutation(individual[a:b]), individual[b:]]
            )
        return population

    def hybridizing_psm_rsm_mutation(self, population: np.array) -> np.array:
        """
        A combination of the PSM and RSM mutation operator described in the following research paper:
        https://www.researchgate.net/publication/282732991_A_New_Mutation_Operator_for_Solving_an_NP-Complete_Problem_Travelling_Salesman_Problem
        :param population: The population to mutate
        :return: The mutated population
        """
        points = np.floor(
            np.random.random([population.shape[0], 2]) * self.tour_size
        ).astype("int")
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

    def elimination(self, joined_population: np.array) -> Tuple[np.array, np.array]:
        """
        Calls the elimination scheme specified in self.elimination_function
        :param joined_population: The population to perform the elimination on
        :return: The population after elimination sorted by fitness value and a list of the corresponding fitness values
        """
        return self.elimination(joined_population)

    def lambda_and_mu_elimination(
        self, joined_population: np.array
    ) -> Tuple[np.array, np.array]:
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
        return (
            joined_population[perm[: self.population_size]],
            objective_values[perm[: self.population_size]],
        )

    def elitism(self, population: np.array, scores: np.array) -> np.array:
        """
        This scheme is applied to the population after elimination to make sure that the fittest member of the previous
        generation will also be part of the new generation (unless a better individual is already present in the current
        generation.
        :param scores:
        :param population: The population to potentially add the fittest member of the previous generation to.
        :return: The (altered) population.
        """
        if (population[0] == self.best_solution).all() or self.best_solution is None:
            return population, scores
        else:
            if self.length_individual(population[0]) < self.best_objective:
                return population, scores
            else:
                population[-1] = self.best_solution
                scores[-1] = self.best_objective
                population[0], population[-1] = population[-1], population[0]
                scores[0], scores[-1] = scores[-1], scores[0]
                return population, scores

    ###############
    #  DIVERSITY  #
    ###############

    def fitness_sharing_elimination(self, population: np.array) -> Tuple[np.array, np.array]:
        """
        This function tries to promote diversity in the population by giving a penalty to individuals that resemble too
        much an individual already in the next generation.
        :param population: The population to perform this scheme on
        :return: The new population after the elimination scheme and their respective scores
        """
        scores = self.length(population)
        perm = np.argsort(scores)
        population = population[perm]
        scores = scores[perm]

        new_population = np.zeros([self.population_size, self.tour_size], dtype=np.int)
        new_population[0] = population[0]
        penalties = np.ones([population.shape[0]])
        penalties = self.compute_penalties(population[0], population, penalties)
        for i in range(1, self.population_size):
            corrected_scores = scores * penalties
            new_population[i] = population[np.argsort(corrected_scores)[0]]
            penalties = self.compute_penalties(new_population[i], population, penalties)
        return new_population, self.length(new_population)

    def compute_penalties(
        self, individual_to_compute_distance_from, population, penalties
    ) -> np.array:
        """
        This function computes the penalties introduced by an individual already in the next generation on the
        remaining individuals in parallel.
        :param individual_to_compute_distance_from: The individual that introduces the penalty on the rest of the
        population
        :param population: The population to add the penalties to
        :param penalties: The penalties that already exist due to other individuals.
        :return: The list with the penalties
        """
        distance_function_parallel_partial = partial(
            distance_function_parallel, perm1=individual_to_compute_distance_from
        )
        with Pool(processes=2) as pool:
            result = pool.map(distance_function_parallel_partial, population)

        result = np.array(result)

        result = result / -self.sigma
        result += 1
        result[result < 0] = 0

        return penalties + result

    def distance_function(self, perm1: np.array, perm2: np.array) -> float:
        """
        Computes the distance between two permutations by counting the edges of the first permutation that are not in
        the second permutation.
        :param perm1: The first permutation
        :param perm2: The second permutation
        :return: The distance between the two permutations
        """
        distance = 0
        for i in range(self.tour_size - 1):
            element = perm1[i]
            x = np.where(perm2 == element)[0][0]
            y = 0 if x == self.tour_size - 1 else x + 1
            if perm1[i + 1] != perm2[y]:
                distance += 1
        element = perm1[-1]
        x = np.where(perm2 == element)[0][0]
        if perm1[0] != perm2[(x + 1) % self.tour_size]:
            distance += 1

        return distance

    def eliminate_duplicate_individuals(
        self, joined_population: np.array, objective_values: np.array
    ) -> Tuple[np.array, np.array]:
        """
        This function is an elimination scheme that tries to introduce diversity into the population. If there are two
        identical individuals in the population then one of them will be replaced by a greedy randomized individual.
        https://arxiv.org/pdf/1702.03594.pdf
        :param joined_population: The population to perform the elimination scheme on.
        :param objective_values: The objective values of the population in the same order
        :return: Returns the new population and the respective scores.
        """
        new_population = np.zeros([self.population_size, self.tour_size], dtype=np.int)
        perm = np.argsort(objective_values)
        sorted_population = joined_population[perm]
        sorted_objective_values = objective_values[perm]
        new_population[0] = sorted_population[0]
        for i in range(self.population_size - 1, 0, -1):
            if sorted_objective_values[i] == sorted_objective_values[i - 1]:
                new_population[i] = self.greedy_randomized_algorithm()
            else:
                new_population[i] = sorted_population[i]
        new_scores = self.length(new_population)
        return new_population, new_scores

    ############################
    #  LOCAL SEARCH OPERATORS  #
    ############################

    def local_search_parallel(self, population: np.array) -> np.array:
        """
        This method performs local search on the population in a parallel way by using the multiprocessing library. The
        distance matrix is stored in a RawArray to make sure no locks can be applied and all child processes can use the
        matrix without needed to pickle it (used in Queues and Pipes). In self.local_search_on_all is specified if the
        operator should be applied on all individuals or on 50% of them.
        :param population: The population to perform local search on.
        :return: The (improved) population
        """
        if self.local_search_on_all:
            with Pool(processes=2) as pool:
                result = pool.map(parallel_3_opt, population)
            return np.vstack(result)
        else:
            random_numbers = np.random.random(population.shape[0])
            random_numbers[
                0
            ] = 1  # Always perform local search on the fittest individual
            population_to_search = population[
                random_numbers > self.percentage_local_search
            ]
            rest = population[random_numbers <= self.percentage_local_search]
            with Pool(processes=2) as pool:
                result = pool.map(parallel_3_opt, population_to_search)

            return np.vstack([result, rest])

    def local_search(self, population: np.array) -> np.array:
        """
        Performs local search on the population. The local search operator to use is specified in
        self.local_search_operator. In self.local_search_on_all is specified if the operator should be applied on
        all individuals or on 50% of them.
        :param population: The population to perform local search on
        :return: The (improved) population
        """
        if self.local_search_on_all:
            for i, individual in enumerate(population):
                population[i] = self.local_search_operator(individual)
        else:
            random_numbers = np.random.random(population.shape[0])
            random_numbers[
                0
            ] = 1  # Always perform local search on the fittest individual
            for i, individual in enumerate(population):
                if random_numbers[i] > self.percentage_local_search:
                    population[i] = self.local_search_operator(individual)

        return population

    def local_search_swap(self, individual: np.array) -> np.array:
        """
        This local search operator loops through the individual and if a combination ABCD is found for which the
        combination ACBD is shorter then it swaps nodes B and C. This operator is very fast because it will only loop
        through the individual once. But therefore it will not create very good solutions because it can only look at
        four nodes at a time.
        :param individual: The individual to perform the local search on.
        :return: The (improved) individual
        """
        for i in range(self.tour_size):
            a = individual[i - 1]
            b = individual[i]
            c = individual[(i + 1) % self.tour_size]
            d = individual[(i + 2) % self.tour_size]

            normal_distance = (
                self.distance_matrix[a][b]
                + self.distance_matrix[b][c]
                + self.distance_matrix[c][d]
            )
            new_distance = (
                self.distance_matrix[a][c]
                + self.distance_matrix[c][b]
                + self.distance_matrix[b][d]
            )
            if new_distance < normal_distance:
                individual[i], individual[(i + 1) % self.tour_size] = (
                    individual[(i + 1) % self.tour_size],
                    individual[i],
                )
        return individual

    def local_search_naive_3_opt(self, individual: np.array) -> np.array:
        """
        This local search operator searches for a local minimum by swapping two arcs in the individual. For an
        individual with arcs ABC it creates individual ACB if the solution is shorter. To find possible shorter
        solutions it loops through the list and creates two splitting points (the third is between the start and end
        of the representation) to create the arcs. This is a very expensive approach, but the results are better than
        the local_search_swap operator because of the bigger neighborhood structure.
        :param individual: The individual to perform the local search on.
        :return: The optimized individual.
        """
        for point1 in range(self.tour_size):
            v1 = individual[0]
            v2 = individual[point1 - 1]
            v3 = individual[point1]
            v6 = individual[-1]
            for point2 in range(point1 + 1, self.tour_size):
                v4 = individual[point2 - 1]
                v5 = individual[point2]
                individual = self.check_for_3_opt_move(
                    individual, point1, point2, v1, v2, v3, v4, v5, v6
                )
        return individual

    def local_search_optimized_3_opt(self, individual: np.array) -> np.array:
        """
        This local search operator searches for a local minimum by swapping two arcs in the individual. For an
        individual with arcs ABC it creates individual ACB if the solution is shorter. To find possible shorter
        solutions it loops through the list of nearest neighbors for the starting point and creates two splitting points
        (the third is between the start and end of the representation) to create the arcs. To change the start and end
        point of the individual the individual is rolled a random value first. This operation is less expensive than the
        naive 3 opt local search, but has slightly worse results. The difference will probably not get noticed because
        more iterations are possible due to the operator being less expensive. The idea to use the neighbourhood list
        came from:
        https://www-sciencedirect-com.kuleuven.ezproxy.kuleuven.be/science/article/pii/S0957417412002734#b0025
        https://dl-acm-org.kuleuven.ezproxy.kuleuven.be/doi/10.5555/320176.320186
        Sometimes also called the Kanellakis-Papadimitriou algorithm.
        :param individual: The individual to perform the local search on.
        :return: The optimized individual.
        """
        # Use negative number to not go out of bounds
        individual = np.roll(individual, -np.random.randint(self.tour_size))

        for point1 in range(self.tour_size):
            v1 = individual[0]
            v2 = individual[point1 - 1]
            v3 = individual[point1]
            v6 = individual[-1]

            for i, neighbor in enumerate(self.nearest_neighbors[v2]):
                point2 = np.where(individual == neighbor)[0][0]
                v4 = individual[point2 - 1]
                v5 = individual[point2]
                if point2 > point1:
                    individual = self.check_for_3_opt_move(
                        individual, point1, point2, v1, v2, v3, v4, v5, v6
                    )

        return individual

    def check_for_3_opt_move(
        self,
        individual: np.array,
        point1: int,
        point2: int,
        v1: int,
        v2: int,
        v3: int,
        v4: int,
        v5: int,
        v6: int,
    ) -> np.array:
        """
        This function checks for a possible 3 opt move for the arcs between endpoints v1-v2, v3-v4 and v5-v6.
        :param individual: The individual to try the 3 opt move on
        :param point1: The first crossing point, the point between v2 and v3
        :param point2: The second crossing point, the point between v4 and v5
        :param v1: The beginning of the first arc
        :param v2: The ending of the first arc
        :param v3: The beginning of the second arc
        :param v4: The ending of the second arc
        :param v5: The beginning of the third arc
        :param v6: The ending of the third arc
        :return: The individual with the 3 opt move applied (if it is better than not)
        """
        old_distance = (
            self.distance_matrix[v2][v3]
            + self.distance_matrix[v4][v5]
            + self.distance_matrix[v6][v1]
        )
        new_distance = (
            self.distance_matrix[v2][v5]
            + self.distance_matrix[v6][v3]
            + self.distance_matrix[v4][v1]
        )
        if new_distance < old_distance:
            a = individual[:point1]
            b = individual[point1:point2]
            c = individual[point2:]
            individual = np.concatenate([a, c, b])
        return individual

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
        distance += self.distance_matrix[individual[-1]][individual[0]]
        return distance

    #####################
    # HELPER FUNCTIONS  #
    #####################

    def is_converged(self) -> bool:
        """
        :return: Returns True if the algorithm has converged, False if not.
        """
        return False
        # if self.same_best_objective >= 30:
        #     return True
        # return False

    def update_scores(self, individual: np.array, scores: np.array) -> None:
        """
        Updates the best and mean objective value according to the new population. Also sets the new best solution in
        the population. The number of the generation is also updated.
        :param individual: The best individual from the entire population
        :param scores: The scores from all the individuals in the population, ordered in ascending order.
        """
        self.last_best_objective = self.best_objective
        self.last_mean_objective = self.mean_objective
        self.mean_objective = np.mean(scores)
        self.best_objective = scores[0]
        if self.best_objective == self.last_best_objective:
            self.same_best_objective += 1
        else:
            self.same_best_objective = 0
        self.best_solution = individual
        self.generation += 1

    def set_selection_pressure(self) -> None:
        """
        Sets the selection pressure according to the problem size. This makes it that the algorithm should not converge
        too quickly.
        """
        if self.selection_function == self.selection_roulette_wheel:
            self.selection_pressure = 0.99
            self.selection_pressure_decay = self.selection_pressure
        else:
            self.selection_pressure = 0.01

    def greedy_randomized_algorithm(self) -> np.array:
        """
        Create a greedy randomized individual. Instead of always taking the greedy option, it is possible to take an
        option that is almost as good as the greedy option. This allows for some extra randomness in the creation of
        the individuals.
        :return: The individual created.
        """
        individual = np.zeros(self.tour_size, dtype=np.int)
        individual[0] = np.random.randint(0, self.tour_size)
        for i in range(1, self.tour_size, 1):
            restricted_candidate_list = self.build_restricted_candidate_list(
                individual[:i]
            )
            individual[i] = np.random.choice(restricted_candidate_list, 1)[0]
        return individual

    def build_restricted_candidate_list(self, indices: np.array) -> np.array:
        """
        Find all next possible nodes that are almost as good as the greedy solution. This set can be used in
        :param indices: The indices already used in the individual the algorithm is trying to create.
        :return: The list of all candidate nodes.
        """
        last_index = indices[-1]
        not_used = set(range(self.tour_size)) - set(indices[:-1])
        not_used.remove(last_index)
        candidates = []
        minimal_distance = np.min(self.distance_matrix[last_index][list(not_used)])
        allowed_distance = (1 + self.rcl) * minimal_distance
        for x in not_used:
            if self.distance_matrix[last_index][x] <= allowed_distance:
                candidates.append(x)
        return np.array(candidates)

    def build_nearest_neighbor_list(self):
        """
        This function creates the list of the nearest neighbors for all nodes in the tour. This list is then used in
        the optimized 3 opt local search operator.
        """
        self.raw_nearest_neighbors = RawArray(
            ctypes.c_long, self.tour_size * self.number_of_nearest_neighbors
        )
        self.nearest_neighbors = np.frombuffer(
            self.raw_nearest_neighbors, dtype=np.int
        ).reshape(self.tour_size, self.number_of_nearest_neighbors)
        for i in range(self.tour_size):
            self.nearest_neighbors[i] = np.argsort(self.distance_matrix[i])[
                1: self.number_of_nearest_neighbors + 1
            ]

    def init_dictionary(self) -> None:
        var_dict["distance_matrix"] = self.raw_distance_matrix
        var_dict["nearest_neighbors"] = self.raw_nearest_neighbors
        var_dict["tour_size"] = self.tour_size
        var_dict["individuals_size"] = self.population_size + self.offspring_size


# with open('testHP194.csv', 'w', newline='') as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerow(['Generations', 'Time', 'Mean', 'Best', 'Population', 'Offspring', 'Local search'])

# for _ in range(3):
#     TSP = r0701014()
#     population_sizes = [16, 25, 32]
#     offspring_sizes = [40, 50, 60]
#     local_search = [0.5]
#     all_combinations = itertools.product(population_sizes, offspring_sizes, local_search)
#     print(all_combinations)
#     for pop_size, offspring_size, local in all_combinations:
#         TSP.__init__()
#         TSP.population_size = pop_size
#         TSP.offspring_size = offspring_size
#         if local == 1:
#             TSP.local_search_on_all = True
#         else:
#             TSP.percentage_local_search = local
#         TSP.optimize('tour194.csv')
#
#         with open('testHP194.csv', 'a') as file:
#             writer = csv.writer(file, delimiter=',')
#             writer.writerow(
#                 [TSP.generation, TSP.time_left, TSP.mean_objective, TSP.best_objective, pop_size, offspring_size,
#                  local])

TSP = r0701014()
TSP.optimize("tour100.csv")
