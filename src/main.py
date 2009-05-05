import constraint
import itertools

import matcher_constraint
import matcher_utils

MAX_RATE = 10000

# Objective function taken as the sum of values in the solution (the higher the rates, the better the solution)
def f(solution):
    result = 0
    for variable, value in solution.items():
        result += value
    return result

def print_solution(solution):
    for variable, value in solution.items():
	if (variable.find("rate") > -1):
		print "%s: %-3.2f %%" % (variable, value / 100.0) 
	elif(variable.find("amount") > -1):
		print "%s: %-5.2f EUR" % (variable, value)

def solve():
    # declare constraints
    # lender_min_rates[i]: the minimum rate at which lender i will lend money
    lender_min_rates = [100, 200, 300]
    # borrower_max_rate[j]: the maximum rate at which borrower j will borrow money
    borrower_max_rates = [800, 900, 1000]

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

    print "Adding variables: "
    for i in range(number_lenders):
        lender_max_amount = lender_max_amounts[i]
        for j in range(number_borrowers):
            # each rate can range between 0.00 and 100.00%
            print "Adding rate_" + str(i) + "_" + str(j), 
            problem.addVariable("rate_" + str(i) + "_" + str(j), range(0, MAX_RATE))

            # each amount is capped by the lender's availability
            print "Adding amount_" + str(i) + "_" + str(j) 
            problem.addVariable("amount_" + str(i) + "_" + str(j), range(0, lender_max_amount))

    print 
    print "Adding amount and rate constraints: "    
    # TODO Understand why the solver isn't returning any solution in "finite" time
    for i in range(number_lenders):
        lender_amounts = ["amount_" + str(i) + "_" + str(j) for j in range(number_borrowers)]
        print "Lender %i: %s" % (i, lender_amounts)
        problem.addConstraint(constraint.MaxSumConstraint(lender_max_amounts[i]), lender_amounts)
        problem.addConstraint(constraint.MinSumConstraint(lender_min_amounts[i]), lender_amounts)
        
        lender_rates = ["rate_" + str(i) + "_" + str(j) for j in range(number_borrowers)]
        print "Lender %i: %s" % (i, lender_rates)

        lender_amounts_rates = matcher_utils.flatten(zip(lender_amounts, lender_rates))
	print lender_amounts_rates
	print lender_min_rates[i]
        problem.addConstraint(matcher_constraint.MinWeightedAverageOrDefaultConstraint(lender_min_rates[i], 0), lender_amounts_rates)

    for j in range(number_borrowers):
        borrower_amounts = ["amount_" + str(i) + "_" + str(j) for i in range(number_lenders)]
        print "Borrower %i: %s" % (j, borrower_amounts)
        problem.addConstraint(constraint.MaxSumConstraint(borrower_max_amounts[j]), borrower_amounts)
        problem.addConstraint(constraint.MinSumConstraint(borrower_min_amounts[j]), borrower_amounts)

        borrower_rates = ["rate_" + str(i) + "_" + str(j) for i in range(number_lenders)]
        print "Borrower %i: %s" % (j, borrower_rates)

        borrower_amounts_rates = matcher_utils.flatten(zip(borrower_amounts, borrower_rates))
        problem.addConstraint(matcher_constraint.MaxWeightedAverageOrDefaultConstraint(borrower_max_rates[j], 0), borrower_amounts_rates)

    # the solution generator is the solution iterator method of the constraint problem object
    best_score = None
    best_solution = None

#   best_solution = problem.getSolution()
#   best_score = f(best_solution)
    for solution in problem.getSolutionIter():
        score = f(solution)
        if(score > best_score):
            print "New current best found"
	    print_solution(solution)
            print "Score: " + str(score)
            best_score = score
            best_solution = solution

    return (best_solution, best_score)

# run the solver
solve()

