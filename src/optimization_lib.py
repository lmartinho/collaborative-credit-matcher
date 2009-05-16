import math
import random
import time

class SolutionGeneratorNotAvailableException(Exception):
    pass

class SolutionEvaluatorNotAvailableException(Exception):
    pass

class Optimizer(object):
    """
    Holds the search strategy used to optimize the problem,
    using the provided generator and evaluator.
    """

    def __init__(self, solution_generator, solution_evaluator, solution_visualizer=None, time_budget=None):
        self.set_solution_generator(solution_generator)

        self.set_solution_evaluator(solution_evaluator)

        if solution_visualizer:
            self.set_solution_visualizer(solution_visualizer)

        # start with no time budget
        self.set_time_budget(time_budget)

    def set_time_budget(self, time_budget):
        self.time_budget = time_budget

    def set_solution_generator(self, solution_generator):
        self.solution_generator = solution_generator

    def set_solution_evaluator(self, solution_evaluator):
        self.solution_evaluator = solution_evaluator

    def set_solution_visualizer(self, solution_visualizer):
        self.solution_visualizer = solution_visualizer

    def optimize(self, time_budget=None):
        raise NotImplementedError, \
              "%s is an abstract class" % self.__class__.__name__

class RandomSearchOptimizer(Optimizer):

    def optimize(self):
        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

        # the solution generator is the solution iterator method of the constraint problem object
        best_score = None
        best_solution = None

        # calculate the time to stop looking for better solutions
        end_time = None
        if self.time_budget:
            end_time = time.time() + self.time_budget

        # get the generator's solution iterator
        solution_iterator = self.solution_generator.get_solution_iterator()

        for solution in solution_iterator:
            utility = self.solution_evaluator.evaluate(parameters, solution)
            score = utility["score"]
            if score > best_score:
                # store the new best result
                best_score = score
                best_utility = utility
                best_solution = solution

            # if there's a time budget, and it has been exceeded: stop
            if end_time and time.time() >= end_time:
                break

            # display the current status (subject to double buffering)
            #self.solution_visualizer.display(parameters, best_solution, utility)

        if not best_solution:
            raise matcher_constraint.NoSolutionAvailableException

        return best_solution

class HillClimbingOptimizer(Optimizer):

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
        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

        current_node = self.solution_generator.get_solution()
        current_node_utility = self.solution_evaluator.evaluate(parameters, current_node)
        current_node_score = current_node_utility["score"]

        while True:
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
                    self.solution_visualizer.display(parameters, node, node_utility)

            # if no better neighbors exist, return the current node
            if next_node_score <= current_node_score:
                return current_node

            current_node = next_node

class SimulatedAnnealingOptimizer(Optimizer):
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

        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

        # initial state
        state = self.solution_generator.get_solution()

        # get the initial state's score
        state_utility = self.solution_evaluator.evaluate(parameters, state)
        state_score = state_utility["score"]

        # initial energy
        energy = 1000

        iterations = 0
        # while termination conditions not met
        while iterations < iterations_budget:
            # get the current state's neighborhood
            state_neighborhood = self.solution_generator.get_neighborhood(state)

            # get a random neighbor
            next_state = self.pick_at_random(state_neighborhood)

            # get the next state score
            next_state_utility = self.solution_evaluator.evaluate(parameters, next_state)
            next_state_score = next_state_utility["score"]

            if next_state_score > state_score:
                state = next_state
            else:
                state = self.apply_acceptance_criterion(state, state_score, next_state, next_state_score, energy)

            energy = self.apply_cooling_schedule(energy)
            iterations += 1

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
