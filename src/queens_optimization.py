from constraint import *
import logging
import sys

import matcher_constraint

class QueensSolutionGenerator(object):
    """
    Generates matcher solutions for the configured problem,
    according to a standard parameter specification for credit matching problems.
    Uses the matcher constraint library as the key solution generation mechanism.

    @capability: solution_generator(set_problem, get_solution, get_solution_iterator)
    """

    def __init__(self, parameters=None):
        self.solver = matcher_constraint.NeighborhoodBacktrackingSolver()
        self.problem = matcher_constraint.MatcherProblem(self.solver, [self.increment_variable, self.decrement_variable])

        size = parameters["n_queens"]
        cols = range(size)
        rows = range(size)
        self.problem.addVariables(cols, rows)
        for col1 in cols:
            for col2 in cols:
                if col1 < col2:
                    self.problem.addConstraint(lambda row1, row2, col1=col1, col2=col2:
                                          abs(row1-row2) != abs(col1-col2) and
                                          row1 != row2, (col1, col2))

    def increment_variable(self, variable, solution, domains):
        new_solution = solution.copy()

        # apply the operator to the specified variable
        new_solution[variable] = solution[variable] + 1

        # if the new variable value belongs to the domain
        if new_solution[variable] in domains[variable]:
            # return the computed solution
            return new_solution
        # if the resulting solution from applying the operator is
        # not in the variable's domain, return None
        else:
            return None

    def decrement_variable(self, variable, solution, domains):
        new_solution = solution.copy()

        new_solution[variable] = solution[variable] - 1

        # if the new variable value belongs to the domain
        if new_solution[variable] in domains[variable]:
            # return the computed solution
            return new_solution
        # if the resulting solution from applying the operator is
        # not in the variable's domain, return None
        else:
            return None

    def get_solution(self):
        logging.debug("solution requested")
        # @todo: return the solution affected by MAX_RATE to standardize the generator API (always 1=100%)
        solution = self.solution_iterator.next()
        logging.debug("solution retrieved")

        return solution

    def get_solutions(self):
        logging.debug("all solutions requested")
        # @todo: return the solution affected by MAX_RATE to standardize the generator API (always 1=100%)
        solutions = self.problem.getSolutions()
        logging.debug("all solutions retrieved")

        return solutions

    def get_solution_iterator(self):
        return self.solution_iterator

    def get_parameters(self):
        return self.parameters

    def get_neighborhood(self, solution):
        return self.problem.getNeighborhood(solution)

    def get_closest_valid_solution(self, candidate_solution):
        logging.debug("closest solution requested")
        solution = self.problem.getClosestValidSolution(candidate_solution)
        logging.debug("closest solution retrieved")

        return solution

    def get_variables(self):
        return self.problem.getVariables()

class QueensSolutionEvaluator(object):
    """
    Evaluates a solution according to a selected utility function.
    """

    def __init__(self):
        pass

    def evaluate(self, parameters, solution):
        """
        Aggregates the standard format solution into a result map, similar to the problem parameters.
        Compute the utility function for the specified solution.
        """
        return 1


class QueensSolutionVisualizer(object):
    def __init__(self, refresh_display_buffer=0):
        pass

    def showSolution(self, solution, size):
        sys.stdout.write("   %s \n" % ("-"*((size*4)-1)))
        for i in range(size):
            sys.stdout.write("  |")
            for j in range(size):
                if solution[j] == i:
                    sys.stdout.write(" %d |" % j)
                else:
                    sys.stdout.write("   |")
            sys.stdout.write("\n")
            if i != size-1:
                sys.stdout.write("  |%s|\n" % ("-"*((size*4)-1)))
        sys.stdout.write("   %s \n" % ("-"*((size*4)-1)))

    def display(self, parameters, solution, utility=0):
        size = parameters["n_queens"]
        self.showSolution(solution, size)
