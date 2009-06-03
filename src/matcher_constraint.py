import constraint
import random

import matcher_utils

class MatcherProblem(constraint.Problem):
    def __init__(self, solver, operators):
        constraint.Problem.__init__(self, solver)
        self.setOperators(operators)

    def getNeighborhood(self, solution):
        """
        Find and return the neighboring solutions to a given solution.
        """
        domains, constraints, vconstraints = self._getArgs()
        operators = self._operators
        if not domains:
            return []
        return self._solver.getNeighborhood(solution, domains, constraints, vconstraints, operators)

    def setOperators(self, operators):
        self._operators = operators

    def getVariables(self):
        return self._variables.keys()

    def getClosestValidSolution(self, candidate_solution):
        domains, constraints, vconstraints = self._getArgs()
        operators = self._operators
        if not domains:
            return []
        return self._solver.getClosestValidSolution(candidate_solution, domains, constraints, vconstraints, operators)

class NoSolutionAvailableException(Exception):
    pass

class MaxWeightedAverageConstraint(constraint.Constraint):
    """
    Constraint enforcing that values of given variables, linearly combined with
    a second set of variables sum up below a given maximum value.

    Example:

    >>> problem = Problem()
    >>> problem.addVariables(["a", "b", "c", "d"], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> problem.addConstraint(MaxWeightedAverageConstraint(3))
    >>> sorted(sorted(x.items()) for x in problem.getSolutions())
    [[('a', 3), ('b', 1), ('c', 3), ('d', 2)], ...]
    """#"""

    def __init__(self, max_weighted_average):
        """
        @param max_weighted_average: Value to be considered as the maximum weighted average
        @type max_weighted_average: number
        """
        self._max_weighted_average = max_weighted_average

    def preProcess(self, variables, domains, constraints, vconstraints):
        # call super
        constraint.Constraint.preProcess(self, variables, domains,
                              constraints, vconstraints)

        # not doing any additional pruning
        return

    def __call__(self, variables, domains, assignments, forwardcheck=False):
        # get the max weighted average value of the constraint
        max_weighted_average = self._max_weighted_average

        # the running average
        cummulative_weighted_average = 0
        # the running weight sum
        cummulative_weight_sum = 0

        grouped_variables = matcher_utils.grouper(2, variables, 1)

        for weight_variable, value_variable in grouped_variables:
            if weight_variable in assignments and value_variable in assignments:
                weight = assignments[weight_variable]
                value = assignments[value_variable]
                # update the running average
                if cummulative_weight_sum + weight > 0:
                    cummulative_weighted_average = (cummulative_weight_sum * cummulative_weighted_average + weight * value) / (cummulative_weight_sum + weight)
                else:
                    cummulative_weighted_average = 0
                cummulative_weight_sum += weight
        if cummulative_weighted_average > max_weighted_average:
            return False
        return True

class MinWeightedAverageOrDefaultConstraint(constraint.Constraint):
    """
    Constraint enforcing that values of given variables, linearly combined with
    a second set of variables sum up above a given minimum value.

    Example:

    >>> problem = Problem()
    >>> problem.addVariables(["a", "b", "c", "d"], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> problem.addConstraint(MinWeightedAverageConstraint(3))
    >>> sorted(sorted(x.items()) for x in problem.getSolutions())
    [[('a', 3), ('b', 1), ('c', 3), ('d', 2)], ...]
    """#"""

    def __init__(self, min_weighted_average, default_value):
        """
        @param min_weighted_average: Value to be considered as the minimum weighted average
        @type min_weighted_average: number
        """
        self._min_weighted_average = min_weighted_average
        self._default_value = default_value

    def preProcess(self, variables, domains, constraints, vconstraints):
        # call super
        constraint.Constraint.preProcess(self, variables, domains,
                              constraints, vconstraints)

        # not doing any additional pruning
        return

    def __call__(self, variables, domains, assignments, forwardcheck=False):
        # get the min weighted average value of the constraint
        min_weighted_average = self._min_weighted_average
        default_value = self._default_value

        # the running average
        cummulative_weighted_average = 0
        # the running weight sum
        cummulative_weight_sum = 0

        grouped_variables = matcher_utils.grouper(2, variables, 1)

        for weight_variable, value_variable in grouped_variables:
            if weight_variable in assignments and value_variable in assignments:
                weight = assignments[weight_variable]
                value = assignments[value_variable]
                # update the running average
                if cummulative_weight_sum + weight > 0:
                    cummulative_weighted_average = (cummulative_weight_sum * cummulative_weighted_average + weight * value) / (cummulative_weight_sum + weight)
                else:
                    cummulative_weighted_average = 0
                cummulative_weight_sum += weight
        if cummulative_weighted_average < min_weighted_average and cummulative_weighted_average != default_value:
            return False
        return True

class NeighborhoodBacktrackingSolver(constraint.BacktrackingSolver):

    def get_solution_by_changing_variable(self, domains, constraints, vconstraints, solution, var):
        iterator = self.get_solution_by_changing_variable_iterator(domains, constraints, vconstraints, solution, var)

        try:
            return iterator.next()
        except StopIteration:
            return None

    def get_solution_by_changing_variable_iterator(self, domains, constraints, vconstraints, solution, var):
        forwardcheck = self._forwardcheck
        assignments = solution.copy()
        
        # delete the assignment for the variable to be fixed
        del assignments[var]

        queue = []

        while True:

            # Mix the Degree and Minimum Remaing Values (MRV) heuristics
            lst = [(-len(vconstraints[variable]),
                    len(domains[variable]), variable) for variable in domains]
            lst.sort()
            for item in lst:
                if item[-1] not in assignments:
                    # Found unassigned variable
                    variable = item[-1]
                    values = domains[variable][:]
                    if forwardcheck:
                        pushdomains = [domains[x] for x in domains
                                                   if x not in assignments and
                                                      x != variable]
                    else:
                        pushdomains = None
                    break
            else:
                # No unassigned variables. We've got a solution. Go back
                # to last variable, if there's one.
                yield assignments.copy()
                if not queue:
                    return
                variable, values, pushdomains = queue.pop()
                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()

            while True:
                # We have a variable. Do we have any values left?
                if not values:
                    # No. Go back to last variable, if there's one.
                    del assignments[variable]
                    while queue:
                        variable, values, pushdomains = queue.pop()
                        if pushdomains:
                            for domain in pushdomains:
                                domain.popState()
                        if values:
                            break
                        del assignments[variable]
                    else:
                        return

                # Got a value. Check it.
                assignments[variable] = values.pop()

                if pushdomains:
                    for domain in pushdomains:
                        domain.pushState()

                for constraint, variables in vconstraints[variable]:
                    if not constraint(variables, domains, assignments,
                                      pushdomains):
                        # Value is not good.
                        break
                else:
                    break

                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()

            # Push state before looking for next variable.
            queue.append((variable, values, pushdomains))

        raise RuntimeError, "Can't happen"

    def getNeighborhood(self, solution, domains, constraints, vconstraints, operators):
        """
        Generates the neighborhood of solutions, for a specified solution.
        Applies the provided operators to all the existing variables, to generate all the possible paths.
        Each generated solutions is tested for validity, according to the existing constraints.
        The set of valid generated solutions is returned as a list.
        """

        neighbor_solutions = []

        for variable in domains:
            for operator in operators:
                neighbor_solution = operator(variable, solution, domains)
                
                if not neighbor_solution:
                    continue

                if self.isValidSolution(neighbor_solution, domains, vconstraints):
                    neighbor_solutions.append(neighbor_solution)
                else:
                    # get an adjacent variable
                    variables = domains.keys()
                    neighbor_variable = variables[variables.index(variable) - 1]
                    # try to fix the solution
                    neighbor_solution = self.get_solution_by_changing_variable(domains, constraints, vconstraints, neighbor_solution, neighbor_variable)
                    
                    if neighbor_solution:
                        neighbor_solutions.append(neighbor_solution)

        return neighbor_solutions

    def isValidSolution(self, solution, domains, vconstraints):
        if not solution:
            return False

        assignments = solution
        lst = domains.keys()
        random.shuffle(lst)

        for variable in lst:
            # check if variable is not in conflict
            for constraint, variables in vconstraints[variable]:
                # if a single constraint is broken, the solution is invalid
                if not constraint(variables, domains, assignments):
                    return False

        # if all the constraints apply, the solution is valid
        return True

    def getClosestValidSolution(self, candidate_solution, domains, constraints, vconstraints, operators):
        # recursively get the neighborhoods, until a valid solution is obtained
        neighbor_candidate_solutions = []

        for variable in domains:
            for operator in operators:
                neighbor_solution = operator(variable, candidate_solution, domains)

                # if the neighbor solution is valid, return it
                if self.isValidSolution(neighbor_solution, domains, vconstraints):
                    return neighbor_solution
                # else store it in the neighboring candidate solutions, if it is not None
                elif neighbor_solution:
                    neighbor_candidate_solutions.append(neighbor_solution)

        # if no neighbor was valid enter recursively and seek valid solutions
        for neighbor_candidate_solution in neighbor_candidate_solutions:
            # the base recursive step, find the neighbors' valid neighbor solutions
            neighbor_candidate_solution = self.getClosestValidSolution(neighbor_candidate_solution, domains, constraints, vconstraints, operators)

            # if the neighbor had a valid neighbor solution return it
            if neighbor_candidate_solution:
                return neighbor_candidate_solution

        # if no valid solution was found, return None
        return None
