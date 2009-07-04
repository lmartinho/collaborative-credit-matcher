import time
import random
import logging
import statlib.stats

import matcher_optimization
import optimization


def generate_scenario(parameters):
    scenario = {}

    # generate lenders
    lenders = {}
    for lender in range(parameters["number_lenders"]):
        lender_parameters = {}

        lender_parameters["minimum_rate"] = round(random.gauss(parameters["mean_lender_rate"], parameters["lender_rate_standard_deviation"]), 2)
        lender_parameters["minimum_amount"] = int(round(random.gauss(parameters["mean_lender_amount"], parameters["lender_amount_standard_deviation"])))
        lender_parameters["maximum_amount"] = lender_parameters["minimum_amount"] * 2

        lenders[lender] = lender_parameters
    scenario["lenders"] = lenders

    # generate borrowers
    borrowers = {}
    for borrower in range(parameters["number_borrowers"]):
        borrower_parameters = {}

        borrower_parameters["maximum_rate"] = round(random.gauss(parameters["mean_borrower_rate"], parameters["borrower_rate_standard_deviation"]), 2)
        borrower_parameters["minimum_amount"] = int(round(random.gauss(parameters["mean_borrower_amount"], parameters["borrower_amount_standard_deviation"])))
        borrower_parameters["maximum_amount"] = borrower_parameters["minimum_amount"] * 2

        borrowers[borrower] = borrower_parameters
    scenario["borrowers"]= borrowers

    return scenario


def export_csv(experiment_results):
    # print header
    print "Iterations;",
    for optimizer_class in experiment_results:
        print "%s," % optimizer_class.__name__,
    print ""

    # print by row
    sample_iterations_budgets = experiment_results[optimizer_class].keys()
    sample_iterations_budgets.sort()
    for iterations_budget in sample_iterations_budgets:
        # row number
        print "%d;" % iterations_budget,

        # results for each column
        for optimizer_class in experiment_results:
            score = experiment_results[optimizer_class][iterations_budget]
            print "%f;" % score,
        print ""

def run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget=None, iterations_budget=None):
    if time_budget:
        optimizer.set_time_budget(time_budget)
    if iterations_budget:
        optimizer.set_iterations_budget(iterations_budget)

    # run the metaheuristic
    solution = optimizer.optimize()

    # evaluate the final solution
    utility = solution_evaluator.evaluate(parameters, solution)

    # display the results
    #solution_visualizer.debug(parameters, solution, utility)

    # display the execution stats from the optimizer
    logging.info("iterations: %s" % optimizer.get_last_run_iterations())
    logging.info("elapsed time: %ss" % optimizer.get_last_run_duration())
    logging.info("--")

    return utility["score"]
