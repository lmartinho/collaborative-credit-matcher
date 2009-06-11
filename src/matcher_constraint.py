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
        if not domains:
            return []
        return self._solver.getClosestValidSolution(candidate_solution, domains, constraints, vconstraints)

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

    def getSolutionsReassignVariable(self, domains, constraints, vconstraints, solution, variable):
        return list(self.getSolutionReassignVariableIterator(domains, constraints, vconstraints, solution, variable))

    def getSolutionReassignVariable(self, domains, constraints, vconstraints, solution, variable):
        iterator = self.getSolutionReassignVariableIterator(domains, constraints, vconstraints, solution, variable)

        try:
            return iterator.next()
        except StopIteration:
            return None

    def getSolutionReassignVariableIterator(self, domains, constraints, vconstraints, solution, reassignment_variable):
        forwardcheck = self._forwardcheck
        assignments = solution.copy()

        # delete the assignment for the variable to be re-assigned
        del assignments[reassignment_variable]

        queue = []

        while True:

            # Mix the Degree and Minimum Remaing Values (MRV) heuristics
            lst = [(-len(vconstraints[reassignment_variable]),
                    len(domains[reassignment_variable]), reassignment_variable)]
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

        Applies the provided operators to all the existing variables, to
        generate all the possible paths.
        After each operator is applied, each solution is tested for validity.
        If the solutions is not valid, a set of alternate solutions is
        generated by adjusting all the other variables.
        Each generated solutions is tested for validity, according to the
        existing constraints.
        The set of valid generated solutions is returned as a list.
        """

        neighbor_solutions = []

        for variable in domains:
            # get a list of all the variable names
            variables = domains.keys()

            # remove the current variable
            variables.remove(variable)

            other_variables = variables

            for operator in operators:
                neighbor_solution = operator(variable, solution, domains)

                # if no result is yield by applying the operator, move on
                if not neighbor_solution:
                    continue

                # if the result is a valid solution
                if self.isValidSolution(neighbor_solution, domains, vconstraints):
                    neighbor_solutions.append(neighbor_solution)
                    continue

                # fix the neighbor_solution in order to obtain a valid solution

                # pick one of the other variables to adjust into a valid value
                adjustment_variable = random.choice(other_variables)

                # get the neighbor solution by adjusting the selected variable
                adjustment_neighbor_solution = self.getSolutionReassignVariable(domains, constraints, vconstraints, neighbor_solution, adjustment_variable)

                # if a valid solution is available append it to the neighbor solutions list
                if adjustment_neighbor_solution:
                    neighbor_solutions.append(adjustment_neighbor_solution)
                    continue

        #print "Found %d neighbors for solutions %s" % (len(neighbor_solutions), solution)
        #neighbor_solutions = matcher_utils.unique(neighbor_solutions)
        #print "Found %d unique neighbors for solutions %s" % (len(neighbor_solutions), solution)
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

    def getClosestValidSolution(self, solution, domains, constraints, vconstraints):
        # if solution is available do not proceed
        if not solution:
            return None

        # if the result is a valid solution
        if self.isValidSolution(solution, domains, vconstraints):
            return solution

        # fix the neighbor_solution in order to obtain a valid solution
        variables = domains.keys()

        # pick one of the variables to adjust into a valid value
        adjustment_variable = random.choice(variables)

        # get the neighbor solution by adjusting the selected variable
        adjustment_solution = self.getSolutionReassignVariable(domains, constraints, vconstraints, solution, adjustment_variable)

        # if a valid solution is available append it to the neighbor solutions list
        if adjustment_solution:
            return adjustment_solution
