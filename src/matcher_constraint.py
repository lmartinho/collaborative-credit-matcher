import constraint
import random
import logging

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

    def getClosestValidSolution(self, candidate_solution):
        domains, constraints, vconstraints = self._getArgs()
        if not domains:
            return None
        return self._solver.getClosestValidSolution(candidate_solution, domains, constraints, vconstraints)

    def getVariables(self):
        return self._variables.keys()

    def getOperators(self):
        return self._operators

    def setOperators(self, operators):
        self._operators = operators

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

    def __call__(self, variables, domains, assignments, forwardcheck=True):
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
        if forwardcheck:
            for weight_variable, value_variable in grouped_variables:
                if weight_variable not in assignments and value_variable not in assignments:
                    # get the variables domains
                    weight_domain = domains[weight_variable]
                    value_domain = domains[value_variable]

                    for weight in weight_domain[:]:
                        for value in value_domain[:]:
                            tmp_cummulative_weight_sum = cummulative_weight_sum
                            tmp_cummulative_weighted_average = cummulative_weighted_average
                            # update the running average
                            if tmp_cummulative_weight_sum + weight > 0:
                                tmp_cummulative_weighted_average = (cummulative_weight_sum * cummulative_weighted_average + weight * value) / (cummulative_weight_sum + weight)
                            else:
                                tmp_cummulative_weighted_average = 0

                            if tmp_cummulative_weighted_average > max_weighted_average:
                                weight_domain.hideValue(weight)
                                value_domain.hideValue(value)
                        if not value_domain:
                            return False
                    if not weight_domain:
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

    def __call__(self, variables, domains, assignments, forwardcheck=True):
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
        if forwardcheck:
            for weight_variable, value_variable in grouped_variables:
                if weight_variable not in assignments and value_variable not in assignments:
                    # get the variables domains
                    weight_domain = domains[weight_variable]
                    value_domain = domains[value_variable]

                    for weight in weight_domain[:]:
                        for value in value_domain[:]:
                            tmp_cummulative_weight_sum = cummulative_weight_sum
                            tmp_cummulative_weighted_average = cummulative_weighted_average
                            # update the running average
                            if tmp_cummulative_weight_sum + weight > 0:
                                tmp_cummulative_weighted_average = (cummulative_weight_sum * cummulative_weighted_average + weight * value) / (cummulative_weight_sum + weight)
                            else:
                                tmp_cummulative_weighted_average = 0

                            if tmp_cummulative_weighted_average < min_weighted_average and cummulative_weighted_average != default_value:
                                weight_domain.hideValue(weight)
                                value_domain.hideValue(value)
                        if not value_domain:
                            return False
                    if not weight_domain:
                        return False

        return True

class InvalidSolutionError(Exception):
    pass

class InvalidParametersError(Exception):
    pass

class NeighborhoodBacktrackingSolver(constraint.BacktrackingSolver):

    def __init__(self, forwardcheck=True):
        constraint.BacktrackingSolver.__init__(self, forwardcheck)

        self.neighbor_adjustment_mode = "closest_valid"

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
        return list(self.getNeighborhoodIterator(solution, domains, constraints, vconstraints, operators))

    def getNeighborhoodIterator(self, solution, domains, constraints, vconstraints, operators):
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

        for variable in domains:
            # get a list of all the variable names
            variables = domains.keys()

            # remove the current variable
            variables.remove(variable)

            other_variables = variables[:]

            for operator in operators:
                neighbor_solution = operator(variable, solution, domains)

                # if no result is yield by applying the operator, move on
                if not neighbor_solution:
                    continue

                # if the result is a valid solution, checks only the constraints of the specified variable
                if self.isValidSolutionVariable(neighbor_solution, domains, vconstraints, variable):
                    yield neighbor_solution

    def getClosestValidSolution(self, solution, domains, constraints, vconstraints):
        if not solution:
            return None

        assignments = solution.copy()
        lst = domains.keys()
        random.shuffle(lst)

        for variable in lst:
            # check if variable is not in conflict
            for variable_constraint, variables in vconstraints[variable]:
                # if a single constraint is broken, the variable is in violation
                if not variable_constraint(variables, domains, assignments):
                    # unassign the variable
                    del assignments[variable]
                    # this variable has been processed
                    break

        for variable in assignments:
            # set the domain, to be restricted to the assignment
            domains[variable] = constraint.Domain([assignments[variable]])

        queue = self.createBacktrackingQueue(assignments, domains)

        # start the normal assignment process
        try:
            solution = self.getAssignmentsSolutionIter(assignments, domains, constraints, vconstraints, queue).next()
        except StopIteration:
            solution = None

        return solution

    def getAssignmentsSolutionIter(self, assignments, domains, constraints, vconstraints, queue):
        forwardcheck = self._forwardcheck

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
                if self.isValidSolution(assignments, domains, vconstraints):
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

    def createBacktrackingQueue(self, assignments, domains):
        # initialize the queue
        queue = []
        # retrieve all variable names
        all_variables = domains.keys()
        # get a list of the assigned variables
        assigned_variables = assignments.keys()
        # get forward check flag
        forwardcheck = self._forwardcheck

        # create a stack element, for each assigned variable
        for assigned_variable in assigned_variables:
            variable = assigned_variable
            values = domains[variable][:] # not true, but the best of our knowledge

            # remove the current variable from all variables
            # as we go down the assignment simulation
            all_variables.remove(variable)

            if forwardcheck:
                # tricky: all_variables represents all the unassigned variables so far
                pushdomains = [domains[x] for x in all_variables]
            else:
                pushdomains = None

            if pushdomains:
                for domain in pushdomains:
                    domain.pushState()

            # skipping the constraint checking
            # as the specified assignments uphold all relevant constraints

            queue.append((variable, values, pushdomains))

        return queue

    def isValidSolution(self, solution, domains, vconstraints):
        if not solution:
            return False

        lst = domains.keys()
        random.shuffle(lst)

        for variable in lst:
            if not self.isValidSolutionVariable(solution, domains, vconstraints, variable):
                return False

        # if all the constraints apply, the solution is valid
        return True

    def isValidSolutionVariable(self, solution, domains, vconstraints, variable):
        # check if variable is not in conflict
        for constraint, variables in vconstraints[variable]:
            # if a single constraint is broken, the solution is invalid
            if not constraint(variables, domains, solution):
                return False

        return True

    def getNeighborAdjustmentMode(self):
        return self.neighbor_adjustment_mode

    def setNeighborAdjustmentMode(self, neighbor_adjustment_mode):
        self.neighbor_adjustment_mode = neighbor_adjustment_mode
