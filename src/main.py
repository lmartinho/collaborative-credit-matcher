import constraint
import itertools

import matcher_constraint
import matcher_utils

MAX_RATE = 10000

def solve(parameters):   
    problem = constraint.Problem(constraint.BacktrackingSolver())

    print "Adding variables: "
    addVariables(problem, parameters)
    
    print "Adding amount and rate constraints: "
    addConstraints(problem, parameters)

    print "Searching for solutions"
    return optimize(problem, parameters)    

def addVariables(problem, parameters):
    lenders = parameters["lenders"]
    borrowers = parameters["borrowers"]

    # each rate can range between 0.00 and 100.00%    
    rate_range = range(0, MAX_RATE)

    for lender_id, lender in lenders.items():
        # each match amount is capped by the lender's availability
        lender_amount_range = range(0, lender["maximum_amount"])

        for borrower_id in borrowers:
            # add the rate match variable for the lender, borrower pair
            rate_variable = "rate_%s_%s" % (lender_id, borrower_id)
            problem.addVariable(rate_variable, rate_range)
            
            amount_variable = "amount_%s_%s" % (lender_id, borrower_id)
            # add the amount match variable for the lender, borrower pair
            problem.addVariable(amount_variable, lender_amount_range)

def addConstraints(problem, parameters):
    lenders = parameters["lenders"]
    borrowers = parameters["borrowers"]

    # add lender constraints
    for lender_id, lender in lenders.items():
        # build a list of all the amount variables for the current lender,
        lender_amounts = ["amount_%s_%s" % (lender_id, borrower_id) for borrower_id in borrowers]

        # constraint the amount variables' sum to the lender's maximum and minimum amounts
        lender_maximum_amount_constraint = constraint.MaxSumConstraint(lender["maximum_amount"])
        lender_minimum_amount_constraint = constraint.MinSumConstraint(lender["minimum_amount"])

        # apply the constraints to the problem
        problem.addConstraint(lender_maximum_amount_constraint, lender_amounts)
        problem.addConstraint(lender_minimum_amount_constraint, lender_amounts)
        
        # build a list of all the rate variables for the current lender
        lender_rates = ["rate_%s_%s" % (lender_id, borrower_id) for borrower_id in borrowers]
        # build a list of amount, rate pairs to use as input for the weighted average rate
        lender_amounts_rates = matcher_utils.flatten(zip(lender_amounts, lender_rates))        

        # constraint the average rate (weighed by the amount) to the lender's minimum accepted rate 
        lender_minimum_rate = round(lender["minimum_rate"] * MAX_RATE)
        lender_minimum_rate_constraint = matcher_constraint.MinWeightedAverageOrDefaultConstraint(lender_minimum_rate, 0)

        # apply the constraint to the problem
        problem.addConstraint(lender_minimum_rate_constraint, lender_amounts_rates)
    
    # add borrower constraints
    for borrower_id, borrower in borrowers.items():
        # build a list of all the amount variables for the current borrower
        borrower_amounts = ["amount_%s_%s" % (lender_id, borrower_id) for borrower_id in borrowers]

        # constraint the amount variables' sum to the lender's maximum and minimum amounts
        borrower_maximum_amount_constraint = constraint.MaxSumConstraint(borrower["maximum_amount"])
        borrower_minimum_amount_constraint = constraint.MinSumConstraint(borrower["minimum_amount"])

        # apply the constraints to the problem
        problem.addConstraint(borrower_minimum_amount_constraint, borrower_amounts)
        problem.addConstraint(borrower_minimum_amount_constraint, borrower_amounts)

        # build a list of all the rate variables for the current borrower
        borrower_rates = ["rate_%s_%s" % (lender_id, borrower_id) for lender_id in lenders]
        # build a list of amount, rates pairs to use as input for the weighted average rate
        borrower_amounts_rates = matcher_utils.flatten(zip(borrower_amounts, borrower_rates))

        # constraint the average rate (weighed by the amount) to the borrower's maximum offered rate
        borrower_maximum_rate = round(borrower["maximum_rate"] * MAX_RATE)
        borrower_maximum_rate_constraint = matcher_constraint.MaxWeightedAverageOrDefaultConstraint(borrower_maximum_rate, 0)

        # apply the constraint to the problem
        problem.addConstraint(borrower_maximum_rate_constraint, borrower_amounts_rates)

def optimize(problem, parameters):    
    # the solution generator is the solution iterator method of the constraint problem object
    best_score = None
    best_solution = None

    for solution in problem.getSolutionIter():
        score = evaluate(utility_function, parameters, solution)
        if score > best_score:
            print "New current best found"
            print "Score: " + str(score)
            matcher_utils.print_solution(parameters, solution)

            # store the new best result
            best_score = score
            best_solution = solution

            import time; time.sleep(1)

    return (best_solution, best_score)

def evaluate(utility_function, parameters, solution):
    # compute the aggregate results, for each participant
    results = calculate_aggregate_results(parameters, solution)

    # evaluate the utility function for the specified parameters and 
    # the results computed from the specified solution
    return utility_function.__call__(parameters, results)


def calculate_aggregate_results(parameters, solution):
    results = {}
    
    lenders = parameters["lenders"]
    borrowers = parameters["borrowers"]
    
    # the map with the results of each lender
    lender_results = {}
    
    for lender_id in lenders:
        lender_amount = 0
        lender_matches = []
        
        for borrower_id in borrowers:
            lender_borrower_match_amount_variable = "amount_%s_%s" % (lender_id, borrower_id) 
            lender_borrower_match_amount = solution[lender_borrower_match_amount_variable]
            lender_amount += lender_borrower_match_amount 
            
            lender_borrower_match_rate_variable = "rate_%s_%s" % (lender_id, borrower_id) 
            lender_borrower_match_rate = solution[lender_borrower_match_rate_variable]
            lender_borrower_match_rate = float(lender_borrower_match_rate) / MAX_RATE
            
            # store the amount and rate for the current lender, borrower match together
            match = (lender_borrower_match_amount, lender_borrower_match_rate)
            lender_matches.append(match)

        # calculate the rate for the current lender, based on the weighted average
        # of the matches it obtained
        lender_rate = matcher_utils.weighted_average(lender_matches)

        # create the results map for the current lender
        lender_result = {"amount": lender_amount, "rate": lender_rate}
        
        # store the current lender's results, in the lender results map
        lender_results[lender_id] = lender_result
        
    # store the lenders' results in the results map
    results["lenders"] = lender_results
    
    # the map with the results of each borrower
    borrower_results = {}

    for borrower_id in borrowers:
        borrower_amount = 0
        borrower_matches = []
        
        for lender_id in lenders:
            lender_borrower_match_amount_variable = "amount_%s_%s" % (lender_id, borrower_id) 
            lender_borrower_match_amount = solution[lender_borrower_match_amount_variable]
            borrower_amount += lender_borrower_match_amount 
            
            lender_borrower_match_rate_variable = "rate_%s_%s" % (lender_id, borrower_id) 
            lender_borrower_match_rate = solution[lender_borrower_match_rate_variable]
            lender_borrower_match_rate = float(lender_borrower_match_rate) / MAX_RATE
            
            # store the amount and rate for the current borrower, lender match together
            match = (lender_borrower_match_amount, lender_borrower_match_rate)
            borrower_matches.append(match)

        # calculate the rate for the current borrower, based on the weighted average
        # of the matches it obtained
        borrower_rate = matcher_utils.weighted_average(borrower_matches)

        # create the results map for the current borrower
        borrower_result = {"amount": borrower_amount, "rate": borrower_rate}
        
        # store the current borrower's results, in the borrower results map
        borrower_results[borrower_id] = borrower_result
        
    # store the borrowers' results in the results map
    results["borrowers"] = borrower_results

    return results

def utility_function(parameters, results):
    # calculate the overall member rate margin    
    lender_rates_margin = calculate_lender_rates_margin(parameters, results)
    borrower_rates_margin = calculate_borrower_rates_margin(parameters, results)
    
    member_rates_margin = lender_rates_margin + borrower_rates_margin

    # calculate the overall member amount margin
    lender_amounts_margin = calculate_lender_amounts_margin(parameters, results)
    borrower_amounts_margin = calculate_borrower_amounts_margin(parameters, results)

    member_amounts_margin = lender_amounts_margin + borrower_amounts_margin 

    # calculate the total capital amount traded
    total_traded_amount = calculate_total_traded_amount(parameters, results)
    
    # calculate the tightness/fairness of the results
    tightness = calculate_tightness(parameters, results)

    # compute the composite utility of the specified solution results
    utility = member_rates_margin + member_amounts_margin + total_traded_amount + tightness     

    return utility

def calculate_lender_rates_margin(parameters, results):  
    lender_parameters = parameters["lenders"]
    lender_results = results["lenders"]

    lender_rate_margin = 0

    for lender_id in lender_parameters:
        minimum_rate = lender_parameters[lender_id]["minimum_rate"]
        effective_rate = lender_results[lender_id]["rate"]
        
        # the lender margin is the difference between the rate that the matches actually yielded
        # and the minimum rate it was willing to accept
        lender_rate_margin += effective_rate - minimum_rate

    return lender_rate_margin


def calculate_borrower_rates_margin(parameters, results):    
    borrower_parameters = parameters["borrowers"]
    borrower_results = results["borrowers"]

    borrower_rate_margin = 0

    for borrower_id in borrower_parameters:
        maximum_rate = borrower_parameters[borrower_id]["maximum_rate"]
        effective_rate = borrower_results[borrower_id]["rate"]
        
        # the borrower margin is the difference between the maximum rate it would have accepted
        # and the rate that the matches actually yielded
        borrower_rate_margin += maximum_rate - effective_rate

    return borrower_rate_margin

def calculate_lender_amounts_margin(parameters, results):
    return 0

def calculate_borrower_amounts_margin(parameters, results):
    return 0

def calculate_total_traded_amount(parameters, results):
    return 0

def calculate_tightness(parameters, results):
    return 0

def run():
    """ The main entry point """
    parameters = {"lenders" : {"1": {"minimum_rate" : 0.05, "minimum_amount" : 10, "maximum_amount" : 100},
                               "2": {"minimum_rate" : 0.04, "minimum_amount" : 20, "maximum_amount" : 200},
                               "3": {"minimum_rate" : 0.03, "minimum_amount" : 30, "maximum_amount" : 300}},
                  "borrowers" : {"4": {"maximum_rate" : 0.15, "minimum_amount" : 10, "maximum_amount" : 100},
                                 "5": {"maximum_rate" : 0.10, "minimum_amount" : 20, "maximum_amount" : 200},
                                 "6": {"maximum_rate" : 0.05, "minimum_amount" : 30, "maximum_amount" : 300}}}
    
    solution = solve(parameters)

# run the solver
run()