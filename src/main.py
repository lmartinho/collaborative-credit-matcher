import constraint
import itertools


class MaxWeightedAverageConstraint(constraint.Constraint):
    """
    Constraint enforcing that values of given variables, linearly combined with 
    a second set of variables sum up to a given amount.

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
        
        grouped_variables = grouper(2, variables, 1)
    
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

def grouper(n, iterable, fillvalue=None):
    """
    grouper(3, 'ABCDEFG', 'x') yields ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def flatten(l):
    list(itertools.chain(*l))

def weighted_average(*args):  
    # group the list into 2 value tuples
    grouped_arguments_iterator = grouper(2, args, 1)
    
    weighted_total = 0
    total_weight = 0
    for weight, value in grouped_arguments_iterator:
        print "weight: %s, value : %s" % (weight, value)
        weighted_total += weight * value
        total_weight += weight
    
    # compute the weighted average
    weighted_average = weighted_total / total_weight
    
    return weighted_average

# Objective function taken as the sum of values in the solution (the higher the rates, the better the solution)
def f(solution):
    result = 0
    for variable, value in solution.items():
        result += value
    return result

def solve():
    # declare constraints
    # lender_min_rates[i]: the minimum rate at which lender i will lend money
    lender_min_rates = [5, 6, 7]
    # borrower_max_rate[j]: the maximum rate at which borrower j will borrow money
    borrower_max_rates = [8, 9, 10]

    # lender_max_amount[i]: the maximum amount lender i is willing to invest
    lender_max_amounts = [100, 200, 300]
    # lender_min_amount[i]: the minimum amount lender i is willing to invest
    lender_min_amounts = [10, 20, 30]
    # borrower_max_amount[j]: the maximum amount borrower j wants to receive 
    borrower_max_amounts = [100, 200, 300]
    # borrower_min_amount[j]: the minimum amount borrower j wants to receive, if he gets a loan
    borrower_min_amounts = [10, 20, 30]

    # TODO: add diversification parameters

    #problem = constraint.Problem(BacktrackingSolver())
    #problem = constraint.Problem(RecursiveBacktrackingSolver())
    problem = constraint.Problem()

    number_lenders = len(lender_min_rates)
    number_borrowers = len(borrower_max_rates)

    for i in range(number_lenders):
        lender_max_amount = lender_max_amounts[i]
        for j in range(number_borrowers):
            # each rate can range between 0 and 100%
            print "Adding rate_" + str(i) + "_" + str(j)
            problem.addVariable("rate_" + str(i) + "_" + str(j), range(0, 100))

            # each amount is capped by the lender's availability
            print "Adding amount_" + str(i) + "_" + str(j)
            problem.addVariable("amount_" + str(i) + "_" + str(j), range(0, lender_max_amount))
    
    # TODO Understand why the solver isn't returning any solution in "finite" time
    for i in range(number_lenders):
        lender_amounts = ["amount_" + str(i) + "_" + str(j) for j in range(number_borrowers)]
        print "Lender %i: %s" % (i, lender_amounts)
        problem.addConstraint(constraint.MaxSumConstraint(lender_max_amounts[i]), lender_amounts)
        problem.addConstraint(constraint.MinSumConstraint(lender_min_amounts[i]), lender_amounts)
        
        lender_rates = ["rate_" + str(i) + "_" + str(j) for j in range(number_borrowers)]
        print "Lender %i: %s" % (i, lender_rates)
        # @fixme: only the average rate should be above the min, not each individual rate
        # still it is a start
        for lender_rate in lender_rates:
            pass
            # @todo: create a new MaxWeightedAverageConstraint that receives a list of weights and a list of values
#            problem.addConstraint(lambda rate: rate >= lender_min_rates[i], [lender_rate])
#
#        lender_variable_pairs = zip(lender_amounts, lender_rates)
#        problem.addConstraint(constraint.FunctionConstraint(weighted_average), lender_variable_pairs)
#        problem.addConstraint(lambda rates, amounts: sum(map(lambda rate, amount: rate * amount, rates, amounts)) >= lender_min_rates[i], lender_rates, lender_amounts)
#        problem.addConstraint(lambda pair: rate, amount = pair, lender_variable_pairs)
        # As the lambda function will be called for each variable, look at the design of MaxSumConstraint for inspiration

    for j in range(number_borrowers):
        borrower_amounts = ["amount_" + str(i) + "_" + str(j) for i in range(number_lenders)]
        print "Borrower %i: %s" % (j, borrower_amounts)
        problem.addConstraint(constraint.MaxSumConstraint(borrower_max_amounts[j]), borrower_amounts)
        problem.addConstraint(constraint.MinSumConstraint(borrower_min_amounts[j]), borrower_amounts)

        borrower_rates = ["rate_" + str(i) + "_" + str(j) for i in range(number_lenders)]
        print "Borrower %i: %s" % (j, borrower_rates)
        # @fixme: only the average rate should be below the max, not each individual rate
        # still it is a start
        for borrower_rate in borrower_rates:
            pass 
#            problem.addConstraint(lambda rate: rate <= borrower_max_rates[i], [borrower_rate])
        borrower_amounts_rates = flatten(zip(borrower_amounts, borrower_rates))
        problem.addConstraint(MaxWeightedAverageConstraint(borrower_max_rates[i]), borrower_amounts_rates)

    # the solution generator is the solution iterator method of the constraint problem object
    best_score = None
    best_solution = None

#   best_solution = problem.getSolution()
#   best_score = f(best_solution)
    for solution in problem.getSolutionIter():
        score = f(solution)
        if(score > best_score):
            print "New current best found"
            print "Solution: " + str(solution) + ", score: " + str(score)
            best_score = score
            best_solution = solution

    return (best_solution, best_score)

# run the solver
print solve()

