import math
import random
import time

import matcher_utils

def timed_optimizer_run(func):
    """
    A decorator to provided timing for optimizer runs.
    """
    def wrapper(*args):
        # get the optimizer instance whose method was invoked
        instance = args[0]
        # the start time for the run
        start_time = time.time()

        # run the method
        method_return = func(*args)

        # the end time for the run
        end_time = time.time()
        # the run duration
        duration = end_time - start_time

        # set the computed duration as the last run duration, in the selected instance
        instance._set_last_run_duration(duration)

        # return the method's return
        return method_return

    # return the wrapper to be called in the place
    # of the selected instance
    return wrapper

class SolutionGeneratorNotAvailableException(Exception):
    pass

class SolutionEvaluatorNotAvailableException(Exception):
    pass

class Optimizer(object):
    """
    Holds the search strategy used to optimize the problem,
    using the provided generator and evaluator.
    """

    def __init__(self, solution_generator, solution_evaluator, solution_visualizer=None):
        self.set_solution_generator(solution_generator)

        self.set_solution_evaluator(solution_evaluator)

        if solution_visualizer:
            self.set_solution_visualizer(solution_visualizer)

        # initialize the time budget
        self.time_budget = None

        # initialize the iterations budget
        self.iterations_budget = None

        # initialize the iterations counter
        self.last_run_iterations = 0

        # initialize the duration time
        self.last_run_duration = 0

    @timed_optimizer_run
    def optimize(self, time_budget=None):
        raise NotImplementedError, \
              "%s is an abstract class" % self.__class__.__name__

    def reset_termination_conditions(self):
        # calculate the time to stop looking for better solutions
        self.start_time = time.time()
        self.stop_time = None
        if self.time_budget:
            self.stop_time = self.start_time + self.time_budget

        # initialize the iteration counter
        self.last_run_iterations = 0

    def termination_conditions_met(self):
        # if there's a time budget, and it has been exceeded: stop
        if self.stop_time and time.time() >= self.stop_time:
            return True

        # if there's an iteration budget, and it has been exceeded: stop
        if self.iterations_budget and self.last_run_iterations >= self.iterations_budget:
            return True

        # update the iteration counter
        self.last_run_iterations += 1

        # keep running, termination conditions not met
        return False

    def get_time_budget(self):
        return self.time_budget

    def set_time_budget(self, time_budget):
        self.time_budget = time_budget

    def get_iterations_budget(self):
        return self.iterations_budget

    def set_iterations_budget(self, iterations_budget):
        self.iterations_budget = iterations_budget

    def get_last_run_iterations(self):
        """
        Returns the number of iterations executed, in the optimizer run.
        """
        return self.last_run_iterations

    def get_last_run_duration(self):
        """
        Returns the duration of the last optimizer run.
        """
        return self.last_run_duration

    def _set_last_run_duration(self, duration):
        """
        Sets the duration of the last optimizer run.
        """
        self.last_run_duration = duration

    def set_solution_generator(self, solution_generator):
        self.solution_generator = solution_generator

    def set_solution_evaluator(self, solution_evaluator):
        self.solution_evaluator = solution_evaluator

    def set_solution_visualizer(self, solution_visualizer):
        self.solution_visualizer = solution_visualizer

class RandomSearchOptimizer(Optimizer):

    @timed_optimizer_run
    def optimize(self):
        self.reset_termination_conditions()

        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

        # the solution generator is the solution iterator method of the constraint problem object
        best_score = None
        best_solution = None

        while not self.termination_conditions_met():
            solution = self.solution_generator.get_solution()
            if not solution:
                break

            # evaluate the current solution
            utility = self.solution_evaluator.evaluate(parameters, solution)
            score = utility["score"]

            # if the current score is better than the best score so far
            if score > best_score:
                # store the new best result
                best_score = score
                best_utility = utility
                best_solution = solution

        if not best_solution:
            raise matcher_constraint.NoSolutionAvailableException

        return best_solution

class HillClimbingOptimizer(Optimizer):

    @timed_optimizer_run
    def optimize(self):
        """
        Hill Climbing Algorithm
        currentNode = startNode;
        loop do
            L = NEIGHBORS(currentNode);
            nextEval = -INF;
            nextNode = NULL;
            for all x in L
                if (EVAL(x) > nextEval)
                    nextNode = x;
                    nextEval = EVAL(x);
            if nextEval <= EVAL(currentNode)
                //Return current node since no better neighbors exist
                return currentNode;
            currentNode = nextNode;
        """
        self.reset_termination_conditions()

        debug = False

        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

        current_node = self.solution_generator.get_solution()
        current_node_utility = self.solution_evaluator.evaluate(parameters, current_node)
        current_node_score = current_node_utility["score"]

        while not self.termination_conditions_met():
            neighborhood = self.solution_generator.get_neighborhood(current_node)

            next_node_score = None
            next_node = None

            for node in neighborhood:
                node_utility = self.solution_evaluator.evaluate(parameters, node)
                node_score = node_utility["score"]

                # if the current node's score is better
                # than the candidate node's, replace the latter with the first
                if node_score > next_node_score:
                    next_node = node
                    next_node_score = node_score

                    # display the current status (subject to double buffering)
                    if debug:
                        self.solution_visualizer.display(parameters, node, node_utility)

            # if no better neighbors exist, return the current node
            if next_node_score <= current_node_score:
                return current_node

            current_node = next_node

        return current_node

class SimulatedAnnealingOptimizer(Optimizer):

    @timed_optimizer_run
    def optimize(self, iterations_budget=10000):
        """
        s <- s0; e <- E(s)                             // Initial state, energy.
        sb <- s; eb <- e                               // Initial "best" solution
        k <- 0                                        // Energy evaluation count.
        while k < kmax and e > emax                  // While time remains and not good enough:
          sn<- neighbour(s)                          //   Pick some neighbour.
          en <- E(sn)                                 //   Compute its energy.
          if en < eb then                            //   Is this a new best?
            sb <- sn; eb <- en                         //     Yes, save it.
          if P(e, en, temp(k/kmax)) > random() then  //   Should we move to it?
            s <- sn; e <- en                           //     Yes, change state.
          k <- k + 1                                  //   One more evaluation done
        return sb                                    // Return the best solution found.
        """
        self.reset_termination_conditions()

        debug = False

        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

        # initial state
        state = self.solution_generator.get_solution()

        # get the initial state's score
        state_utility = self.solution_evaluator.evaluate(parameters, state)
        state_score = state_utility["score"]

        # initial energy
        energy = 1000

        # while termination conditions not met
        while not self.termination_conditions_met():
            # get the current state's neighborhood
            state_neighborhood = self.solution_generator.get_neighborhood(state)

            # get a random neighbor
            # @todo: use random.choice on list
            next_state = self.pick_at_random(state_neighborhood)

            # if a new state is not available continue
            if not next_state:
                continue

            # get the next state score
            next_state_utility = self.solution_evaluator.evaluate(parameters, next_state)
            next_state_score = next_state_utility["score"]

            if next_state_score > state_score:
                state = next_state
            else:
                state = self.apply_acceptance_criterion(state, state_score, next_state, next_state_score, energy)

            # if a state transition occurred, show the new state
            if debug and state == next_state:
                # show the new best solution
                self.solution_visualizer.display_solution(parameters, next_state)
                self.solution_visualizer.display_utility(next_state_utility)

            energy = self.apply_cooling_schedule(energy)

        return state

    def pick_at_random(self, solution_list):
        if not solution_list:
            return None

        maximum_index = len(solution_list) - 1

        random_index = int(round(random.random() * maximum_index))

        random_item = solution_list[random_index]

        return random_item

    def apply_acceptance_criterion(self, state, state_score, next_state, next_state_score, energy):
        energy_difference = float(next_state_score) - state_score
        acceptance_probability = math.exp(energy_difference / energy)

        random_value = random.random()

        if random_value <= acceptance_probability:
            return next_state
        else:
            return state

    def apply_cooling_schedule(self, energy):
        cooling_alpha = 0.9 # (generally in the range 0.8 <= alpha <= 1)

        return float(energy) * cooling_alpha

class AntColonyOptimizer(Optimizer):
    """
    procedure ACOMetaheuristic
        ScheduleActivities
            ConstructAntsSolutions
            UpdatePheromones
            DaemonActions
        end-ScheduleActivites
    end-procedure
    """
    pass

class ParticleSwarmOptimizer(Optimizer):
    """
    // Initialize the particle positions and their velocities
    for I = 1 to number of particles n do
      for J = 1 to number of dimensions m do
         X[I][J] = lower limit + (upper limit - lower limit) * uniform random number
         V[I][J] = 0
      enddo
    enddo

    // Initialize the global and local fitness to the worst possible
    fitness_gbest = inf;
    for I = 1 to number of particles n do
       fitness_lbest[I] = inf
    enddo

    // Loop until convergence, in this example a finite number of iterations chosen
    for k = 1 to number of iterations to do

      // evaluate the fitness of each particle
      fitness_X = evaluate_fitness(X)

      // Update the local bests and their fitness
     for I = 1 to number of particles n do
        if (fitness_X[I] < fitness_lbest[I])
          fitness_lbest[I] = fitness_X[I]
          for J = 1 to number of dimensions m do
            X_lbest[I][J] = X[I][J]
          enddo
        endif
      enddo

      // Update the global best and its fitness
      [min_fitness, min_fitness_index] = min(fitness_X)
      if (min_fitness < fitness_gbest)
          fitness_gbest = min_fitness
          for J = 1 to number of dimensions m do
            X_gbest[J] = X(min_fitness_index,J)
          enddo
      endif

      // Update the particle velocity and position
      for I = 1 to number of particles n do
        for J = 1 to number of dimensions m do
          R1 = uniform random number
          R2 = uniform random number
          V[I][J] = w*V[I][J]
                 + C1*R1*(X_lbest[I][J] - X[I][J])
                 + C2*R2*(X_gbest[J] - X[I][J])
          X[I][J] = X[I][J] + V[I][J]
        enddo
      enddo

    enddo
    """
    @timed_optimizer_run
    def optimize(self, iterations_budget=10000):
        self.reset_termination_conditions()

        debug = False

        number_particles = 10
        inertial_constant = 0.9
        cognitive_weight = 2
        social_weight = 2

        particles = range(number_particles)
        variables = self.solution_generator.get_variables()

        # initialize the particle positions and their velocities
        particle_solutions = []
        particle_velocities = []
        for particle in particles:
            # get a random initial solution
            particle_solution = self.solution_generator.get_solution()

            # initialize a zero velocity vector
            particle_velocity = {}
            for variable in variables:
                particle_velocity[variable] = 0

            # add the created solution and velocity
            particle_solutions.append(particle_solution)
            particle_velocities.append(particle_velocity)

        # initialize the global best solution
        global_best_solution = None
        # initialize the global fitness best score
        global_best_fitness = None

        # initialize the local best solutions
        particle_solutions_local_bests = []
        # initialize the local fitness best scores
        particle_fitnesses_local_bests = []

        for particle in particles:
            particle_solutions_local_bests.append(None)
            particle_fitnesses_local_bests.append(None)

        # calculate the time to stop looking for better solutions
        start_time = time.time()
        stop_time = None
        if self.time_budget:
            stop_time = time.time() + self.time_budget

        # initialize the iteration counter
        self.last_run_iterations = 0

        # loop until convergence
        while not self.termination_conditions_met():
            # calculate the fitness of each particle
            particle_fitnesses = self.evaluate_swarm(particle_solutions)

            # update the local bests and their fitness
            for particle in particles:
                if(particle_fitnesses[particle] > particle_fitnesses_local_bests[particle]):
                    # update the local best
                    particle_solutions_local_bests[particle] = particle_solutions[particle]
                    # update the local best fitness
                    particle_fitnesses_local_bests[particle] = particle_fitnesses[particle]

            # update the global best and its fitness
            best_particle = self.get_best_particle(particle_fitnesses_local_bests)
            if particle_fitnesses_local_bests[best_particle] > global_best_fitness:
                global_best_solution = particle_solutions_local_bests[best_particle]
                global_best_fitness = particle_fitnesses_local_bests[best_particle]

            # update the particle velocity and position
            for particle in particles:
                # initialize the candidate solution to compute for the current particle
                particle_candidate_solution = {}

                # for all the variables in the solution, compute the velocity and candidate solution component
                for variable in variables:
                    # get the random uniform weights
                    random_cognitive_weight = random.random()
                    random_social_weight = random.random()

                    # get solution component values for the current particle and variable
                    variable_value = particle_solutions[particle][variable]
                    local_best_variable_value = particle_solutions_local_bests[particle][variable]
                    global_best_variable_value = global_best_solution[variable]

                    # update the particle velocity
                    inertial_component = inertial_constant * particle_velocities[particle][variable]
                    cognitive_component = cognitive_weight * random_cognitive_weight * (local_best_variable_value - variable_value)
                    social_component = social_weight * random_social_weight * (global_best_variable_value - variable_value)

                    particle_velocities[particle][variable] = inertial_component + cognitive_component + social_component

                    particle_candidate_solution[variable] = variable_value + particle_velocities[particle][variable]

                # get the closest valid solution
                particle_next_solution = self.get_closest_valid_solution(particle_candidate_solution)

                # update the particle solution
                particle_solutions[particle] = particle_next_solution

                # visualize the current solution
                if debug:
                    print "Particle ", particle
                    parameters = self.solution_generator.get_parameters()
                    self.solution_visualizer.display_solution(parameters, particle_next_solution)

            # if there's a time budget, and it has been exceeded: stop
            if stop_time and time.time() >= stop_time:
                break

            # if there's an iteration budget, and it has been exceeded: stop
            if self.iterations_budget and self.last_run_iterations >= self.iterations_budget:
                break

            # update the iteration counter
            self.last_run_iterations += 1

        return global_best_solution

    def evaluate_swarm(self, particle_solutions):
        parameters = self.solution_generator.get_parameters()

        particle_fitnesses = []

        for particle_solution in particle_solutions:
            particle_utility = self.solution_evaluator.evaluate(parameters, particle_solution)
            # @TODO utility -> fitness
            particle_score = particle_utility["score"]

            particle_fitnesses.append(particle_score)

        return particle_fitnesses

    def get_best_particle(self, particle_fitnesses_local_bests):
        best_particle_fitness = max(particle_fitnesses_local_bests)
        best_particle = particle_fitnesses_local_bests.index(best_particle_fitness)

        return best_particle

    def get_closest_valid_solution(self, particle_candidate_solution):
        """
        Uses the solution generator to retrieve a valid approximation to the
        computed candidate solution.
        """
        # @TODO: missing constraints for development purposes
        #return self.solution_generator.get_closest_valid_solution(particle_candidate_solution)
        return particle_candidate_solution


FITNESS = 1
"""
The position to access the fitness in an individual tuple from an evaluated population.
"""

class GeneticAlgorithmOptimizer(Optimizer):
    """
    1. Choose initial population
    2. Evaluate the fitness of each individual in the population
    3. Repeat until termination: (time limit or sufficient fitness achieved)
        1. Select best-ranking individuals to reproduce
        2. Breed new generation through crossover and/or mutation (genetic operations) and give birth to offspring
        3. Evaluate the individual fitnesses of the offspring
        4. Replace worst ranked part of population with offspring
    """

    def __init__(self, solution_generator, solution_evaluator, solution_visualizer=None):
        Optimizer.__init__(self, solution_generator, solution_evaluator, solution_visualizer)

        self.initial_population_size = 10
        self.reproduction_sample_size = 4
        self.population_size = 10
        self.number_replacements = 4
        self.maximum_trait_value = None

    @timed_optimizer_run
    def optimize(self):
        self.reset_termination_conditions()

        # get the generator's parameters
        self.generator_parameters = self.solution_generator.get_parameters()

        # choose initial population
        initial_population = self.create_population()

        # evaluate the fitness of each individual in the population
        population_evaluated = self.evaluate_fitness(initial_population)

        # the solution generator is the solution iterator method of the constraint problem object
        best_solution = None

        while not self.termination_conditions_met():

            # select best-ranking individuals to reproduce
            fittest_population_evaluated = self.get_fittest(population_evaluated, self.reproduction_sample_size)

            # @todo: debug the crossover
            # @todo: implement mutation
            # breed new generation through crossover and/or mutation (genetic operations) and give birth to offspring
            offspring = self.breed_generation(fittest_population_evaluated)

            # evaluate the individual fitness of the offspring
            offspring_evaluated = self.evaluate_fitness(offspring)

            # replace worst ranked part of population with offspring
            population_evaluated = self.replace_worst(population_evaluated, offspring_evaluated, self.number_replacements)

        # get the best individual from the last generation
        fittest_population_evaluated = self.get_fittest(population_evaluated, 1)

        # get the solution from the individual-fitness tuple list
        best_solution, best_score = fittest_population_evaluated[0]

        if not best_solution:
            raise matcher_constraint.NoSolutionAvailableException

        return best_solution

    def create_population(self):
        population = []

        for i in range(self.initial_population_size):
            individual = self.solution_generator.get_solution()
            population.append(individual)

        return population

    def evaluate_fitness(self, population):
        # create a list of (individual, fitness) tuples
        population_fitness = [(individual, self.evaluate(individual)) for individual in population]

        # return the list of population with respective fitness
        return population_fitness

    def get_fittest(self, population_evaluated, number_fittest):
        population_evaluated.sort(self.compare_individuals)

        # return the top number_fittest individuals
        return population_evaluated[-number_fittest:]

    def breed_generation(self, fittest_population_evaluated):
        offspring = []

        fittest_population = [individual for individual, score in fittest_population_evaluated]

        couples = list(matcher_utils.grouper(2, fittest_population))

        # for all pairs of individuals:
        for couple in couples:
            individual_a, individual_b = couple

            # get the individual genome for each
            individual_a_chromosomes = self.create_chromosomes(individual_a)
            individual_b_chromosomes = self.create_chromosomes(individual_b)

            # apply crossover operator
            child_a_chromosomes, child_b_chromosomes = self.crossover(individual_a_chromosomes, individual_b_chromosomes)

            # apply mutation operators
            #mutated_child_chromosomes = self.mutate(child_chromosomes)

            # convert the genome back to an individual
            child_a = self.create_individual(child_a_chromosomes)
            child_b = self.create_individual(child_b_chromosomes)

            # use the child if valid or get the closest valid solution
            child_a = self.solution_generator.get_closest_valid_solution(child_a)
            child_b = self.solution_generator.get_closest_valid_solution(child_b)

            # append the children to the offspring
            offspring.append(child_a)
            offspring.append(child_b)

        return offspring

    def replace_worst(self, population_evaluated, offspring_evaluated, number_replacements):
        new_population = []

        population_evaluated.sort(self.compare_individuals)
        offspring_evaluated.sort(self.compare_individuals)

        top_offspring = offspring_evaluated[-self.number_replacements:]
        top_population = population_evaluated[self.number_replacements:0]

        new_population = top_offspring + top_population

        return new_population

    def create_chromosomes(self, individual):
        chromosomes_map = {}

        for trait in individual:
            trait_binary_string = self.create_binary_string(individual[trait])
            chromosomes_map[trait] =  trait_binary_string

        return chromosomes_map

    def crossover(self, individual_a_chromosomes, individual_b_chromosomes):
        child_a_chromosomes = {}
        child_b_chromosomes = {}

        for trait in individual_a_chromosomes:
            individual_a_chromosome = individual_a_chromosomes[trait]
            individual_b_chromosome = individual_b_chromosomes[trait]

            # crossover by the half
            crossover_point = int(len(individual_a_chromosome) / 2)

            child_a_chromosome = individual_a_chromosome[0:crossover_point] + individual_b_chromosome[crossover_point:]
            child_b_chromosome = individual_b_chromosome[0:crossover_point] + individual_a_chromosome[crossover_point:]

            child_a_chromosomes[trait] = child_a_chromosome
            child_b_chromosomes[trait] = child_b_chromosome

        return (child_a_chromosomes, child_a_chromosomes)

    def mutate(self, individual_chromosomes):
        pass

    def create_individual(self, individual_chromosomes):
        individual = {}

        for trait in individual_chromosomes:
            trait_value = int(individual_chromosomes[trait], 2)
            individual[trait] = trait_value

        return individual

    def create_binary_string(self, trait_value):
        """
        Returns a binary string, filled to match the standard number of bits.
        """

        trait_value_binary_string = bin(trait_value)

        bit_characters = trait_value_binary_string.split("0b")[0]
        bit_characters = bit_characters.zfill(self.maximum_chromosome_length - len("0b"))
        binary_string = "0b" + bit_characters

        return binary_string

    # @todo: create getters and setters for the optimizer parameters
    def evaluate(self, individual):
        individual_utility = self.solution_evaluator.evaluate(self.generator_parameters, individual)
        individual_score = individual_utility["score"]

        return individual_score

    def compare_individuals(self, individual_a,individual_b):
        individual_a_fitness = individual_a[FITNESS]
        individual_b_fitness = individual_b[FITNESS]

        return int(individual_a_fitness - individual_b_fitness)

    def set_maximum_trait_value(self, maximum_trait_value):
        maximum_trait_value_binary_string = bin(maximum_trait_value)
        maximum_binary_string_length = len(maximum_trait_value_binary_string)

        self.maximum_trait_value = maximum_trait_value
        self.maximum_chromosome_length = maximum_binary_string_length
