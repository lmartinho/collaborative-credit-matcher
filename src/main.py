import constraint
import itertools

import matcher_constraint
import matcher_utils

MAX_RATE = 10000

def f(solution):
    results = create_results(solution)
    return utility_function(parameters, results)

# @todo: write a factory method that create dictionaries containing a structured form of the retrieved solution
def create_results(solution)
    pass

def utility_function(parameters, results):
    # utility function must take into account several factors
    
    # calculate the overall member rate margin    
    lender_rates_margin = calculate_lender_rates_margin(parameters, results)
    borrower_rates_margin = calculate_borrower_rates_margin(parameters, results)
    
    member_rates_margin =  lender_rates_margin + borrower_rates_margin

    # calculate the overall member amount margin
    lender_amounts_margin = calculate_lender_amounts_margin(parameters, results)
    borrower_amounts_margin = calculate_borrower_amounts_margin(parameters, results)

    member_amounts_margin = lender_amounts_margin + borrower_amounts_margin 

    # calculate the total capital amount traded
    total_traded_amount = calculate_total_traded_amount(results)
    
    # calculate the tightness/fairness of the results
    tightness = calculate_tightness(parameters, results)

    utility = member_rates_margin + member_amounts_margin + total_traded_amount + tightness     
    
    return utility_function

def calculate_lender_rates_margin(parameters, result):
    lender_rate_margin = 0
    
    lender_parameters = parameters["lender"]
    lender_results = results["lender"]
    lenders = lender_parameters.keys()

    for lender in lenders:
        effective_rate = lender_results[lender]["rate"]
        minimum_rate = lender_parameters[lender]["minimum_rate"]

        lender_rate_margin += effective_rate - minimum_rate

    return lender_rate_margin

def calculate_borrower_rates_margin(parameters, result):
    borrower_rate_margin = 0
    
    borrower_parameters = parameters["borrower"]
    borrower_results = results["borrower"]
    borrowers = borrower_parameters.keys()

    for borrower in borrowers:
        effective_rate = result[borrower]["rate"]
        maximum_rate = parameters[borrower]["maximum_rate"]

        borrower_rate_margin += effective_rate - maximum_rate

    return borrower_rate_margin

def calculate_lender_amounts_margin(parameters, result):
    return 0

def calculate_borrower_amounts_margin(parameters, result):
    return 0

def calculate_total_traded_amount(parameters, result):
    return 0

def calculate_tightness(result):
    return 0

def print_solution(solution):
    solution_list = build_solution_list(solution)
    print_solution_list(solution_list)

def build_solution_list(solution):
    solutions = []
    lenders = 3
    borrowers = 3
    for i in range(lenders):
        solutions.append([])
        for j in range(borrowers):
            rate = solution["rate_%d_%d" % (i,j)]
            amount = solution["amount_%d_%d" % (i,j)]
            solution_tuple = (amount, rate)
            solutions[i].append(solution_tuple)
    
    return solutions

def print_match(match):
    amount, rate = match

    print "(%7.2fEUR" % amount,
    print("@"),
    print "%5.2f%%)" % (rate / 10000.0),

def print_solution_list(solution_list):
    lenders = len(solution_list)
    borrowers = len(solution_list[0])
    print "              ",
    for i in range(borrowers):
        print "B%d                      " % i,
    print ""
    for i in range(borrowers):
        print "--------------------------",
    print ""

    i = 0
    for lender_solutions in solution_list:
        print "L%d | " % i, 
        for borrower_solution in lender_solutions:
            print_match(borrower_solution),
            print "  ",
        print ""
        i += 1
    
    print "   -------------------------------------------------------------------"

def get_lender_minimum_rates(parameters):
    lender_parameters = parameters["lenders"]
    lender_minimum_rates = [lender["minimum_rate"] for lender in lender_parameters]

    return lender_minimum_rates 

def get_borrower_maximum_rates(parameters):
    borrower_parameters = parameters["borrowers"]
    borrower_maximum_rates = [borrower["maximum_rate"] for borrower in borrower_parameters]

    return borrower_maximum_rates 

def get_lender_maximum_amounts(parameters):
    lender_parameters = parameters["lenders"]
    lender_maximum_amounts = [lender["maximum_amount"] for lender in lender_parameters]

    return lender_maximum_amounts 

def get_lender_minimum_amounts(parameters):
    lender_parameters = parameters["lenders"]
    lender_minimum_amounts = [lender["minimum_amount"] for lender in lender_parameters]

    return lender_minimum_amounts 

def get_borrower_maximum_amounts(parameters):
    borrower_parameters = parameters["borrowers"]
    borrower_maximum_amounts = [borrower["maximum_amount"] for borrower in borrower_parameters]

    return borrower_maximum_amounts 

def get_borrower_minimum_amounts(parameters):
    borrower_parameters = parameters["borrowers"]
    borrower_minimum_amounts = [borrower["minimum_amount"] for borrower in borrower_parameters]

    return borrower_minimum_amounts 

def solve():
    parameters = {"lenders" : [{"minimum_rate" : 0.01, "minimum_amount" : 10, "maximum_amount" : 100},
                               {"minimum_rate" : 0.02, "minimum_amount" : 20, "maximum_amount" : 200},
                               {"minimum_rate" : 0.03, "minimum_amount" : 30, "maximum_amount" : 300}],
                  "borrowers" : [{"maximum_rate" : 0.08, "minimum_amount" : 10, "maximum_amount" : 100},
                                 {"maximum_rate" : 0.09, "minimum_amount" : 20, "maximum_amount" : 200},
                                 {"maximum_rate" : 0.1, "minimum_amount" : 30, "maximum_amount" : 300} ]}

    # @todo: convert rate 100% = 10000
    # lender_min_rates[i]: the minimum rate at which lender i will lend money
    lender_min_rates = get_lender_minimum_rates(parameters)
    # borrower_max_rate[j]: the maximum rate at which borrower j will borrow money
    borrower_max_rates = get_borrower_maximum_rates(parameters)

    # lender_max_amount[i]: the maximum amount lender i is willing to invest
    lender_max_amounts = get_lender_maximum_amounts(parameters)
    # lender_min_amount[i]: the minimum amount lender i is willing to invest
    lender_min_amounts = get_lender_minimum_amounts(parameters)

    # borrower_max_amount[j]: the maximum amount borrower j wants to receive 
    borrower_max_amounts = get_borrower_maximum_amounts(parameters)
    # borrower_min_amount[j]: the minimum amount borrower j wants to receive, if he gets a loan
    borrower_min_amounts = get_borrower_minimum_amounts(parameters)

    # @TODO: add diversification parameters

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
        if(score != best_score):
            print "New current best found"
            print_solution(solution)
            print "Score: " + str(score)
            best_score = score
            best_solution = solution

    return (best_solution, best_score)
      
# run the solver
solutions = []
solution = solve()
solution_list = build_solution_list(solution)

print_solution_list(solution_list)
