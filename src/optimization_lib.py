class SolutionGeneratorNotAvailableException(Exception):
    pass

class SolutionEvaluatorNotAvailableException(Exception):
    pass

class Optimizer(object):
    def __init__(self, solution_generator, solution_evaluator, solution_visualizer=None):
        self.set_solution_generator(solution_generator)

        self.set_solution_evaluator(solution_evaluator)

        if solution_visualizer:
            self.set_solution_visualizer(solution_visualizer)

        # start with no time budget
        self.time_budget = None

    def set_budget(self, time_budget):
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
    """ Holds the search strategy used to optimize the problem, using the provided generator and evaluator. """

    def __init__(self, solution_generator, solution_evaluator, solution_visualizer=None):
        self.set_solution_generator(solution_generator)

        self.set_solution_evaluator(solution_evaluator)

        if solution_visualizer:
            self.set_solution_visualizer(solution_visualizer)

        # start with no time budget
        self.time_budget = None

    def set_budget(self, time_budget):
        self.time_budget = time_budget

    def set_solution_generator(self, solution_generator):
        self.solution_generator = solution_generator

    def set_solution_evaluator(self, solution_evaluator):
        self.solution_evaluator = solution_evaluator

    def set_solution_visualizer(self, solution_visualizer):
        self.solution_visualizer = solution_visualizer

    def optimize(self, time_budget=None):
        print "Searching for solutions"

        if not self.solution_generator:
            raise SolutionGeneratorNotAvailableException

        if not self.solution_evaluator:
            raise SolutionEvaluatorNotAvailableException

        if time_budget:
            current_time_budget = time_budget
        elif self.time_budget:
            current_time_budget = time_budget
        else:
            current_time_budget = None

        return self.search(current_time_budget)

    def search(self, time_budget=None):
        # the solution generator is the solution iterator method of the constraint problem object
        best_score = None
        best_solution = None

        # calculate the time to stop looking for better solutions
        end_time = None
        if time_budget:
            end_time = time.time() + time_budget

        # get the generator's parameters
        parameters = self.solution_generator.get_parameters()

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

    def optimize(self, time_budget=None):
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
