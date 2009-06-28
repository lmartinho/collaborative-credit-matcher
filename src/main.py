import time
import random
import logging

import matcher_optimization
import optimization
import queens_optimization

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

def compare_optimizers():

    members_scale = [10, 50, 100]
    time_budgets = [60, 180, 600]
    iterations_budget = [100, 1000, 10000]

    # homogeneous mean, larger deviation
    scenario1_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.10,
                  "lender_rate_standard_deviation" : 0.05,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.10,
                  "borrower_rate_standard_deviation" : 0.05,
                  "number_lenders" : 10,
                  "number_borrowers" : 10}

    # conservative lenders, aggressive borrowers
    scenario2_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.20,
                  "lender_rate_standard_deviation" : 0.01,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.05,
                  "borrower_rate_standard_deviation" : 0.01,
                  "number_lenders" : 10,
                  "number_borrowers" : 10}

    # aggressive lenders, conservative borrowers
    scenario3_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.05,
                  "lender_rate_standard_deviation" : 0.01,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.20,
                  "borrower_rate_standard_deviation" : 0.01,
                  "number_lenders" : 10,
                  "number_borrowers" : 10}

    # homogeneous mean, smaller deviation
    scenario4_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.05,
                  "lender_rate_standard_deviation" : 0.01,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.20,
                  "borrower_rate_standard_deviation" : 0.01,
                  "number_lenders" : 5,
                  "number_borrowers" : 5}

    #scenario_parameters_list = [scenario1_parameters, scenario2_parameters, scenario3_parameters, scenario4_parameters]
    scenario_parameters_list = [scenario4_parameters]
    optimizer_classes = [#optimization.RandomSearchOptimizer,
                         #optimization.HillClimbingOptimizer,
                         #optimization.SimulatedAnnealingOptimizer,
                         optimization.GeneticAlgorithmOptimizer
                         #optimization.ParticleSwarmOptimizer
                         ]
    time_budget = 60
    iterations_budget = None
    number_runs = 1

    for scenario_parameters in scenario_parameters_list:
        parameters = generate_scenario(scenario_parameters)

        for optimizer_class in optimizer_classes:
            for run in range(number_runs):
                # create the generator
                solution_generator = matcher_optimization.MatcherSolutionGenerator(parameters)
                # create the evaluator, using the tight margin utility function
                solution_evaluator = matcher_optimization.MatcherSolutionEvaluator(matcher_optimization.MatcherSolutionEvaluator.tight_margin_utility)
                # create the visualizer
                solution_visualizer = matcher_optimization.MatcherSolutionVisualizer()

                # create the coordinator, injecting the created objects
                optimizer = optimizer_class(solution_generator, solution_evaluator, solution_visualizer)

                # run the optimizer
                logging.info("Run number %d of optimizer %s" % (run, optimizer))
                run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget, iterations_budget)

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
    solution_visualizer.display(parameters, solution, utility)

    # display the execution stats from the optimizer
    logging.info("iterations: %s" % optimizer.get_last_run_iterations())
    logging.info("elapsed time: %ss" % optimizer.get_last_run_duration())
    logging.info("--")

def run_queens():
    parameters = {"n_queens" : 9}
    # create the generator
    solution_generator = queens_optimization.QueensSolutionGenerator(parameters)
    # create the evaluator, using the tight margin utility function
    solution_evaluator = queens_optimization.QueensSolutionEvaluator()
    # create the visualizer
    solution_visualizer = queens_optimization.QueensSolutionVisualizer()

    solutions = solution_generator.get_solutions()

    for solution in solutions:
        solution_visualizer.display(parameters, solution)
        neighbor_solutions = solution_generator.get_neighborhood(solution)

        for neighbor_solution in neighbor_solutions:
            solution_visualizer.display(parameters, neighbor_solution)
        print "---------------------------"

def run_matcher():
    # homogeneous mean, smaller deviation
    scenario4_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.05,
                  "lender_rate_standard_deviation" : 0.01,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.20,
                  "borrower_rate_standard_deviation" : 0.01,
                  "number_lenders" : 5,
                  "number_borrowers" : 5}

    #scenario_parameters_list = [scenario1_parameters, scenario2_parameters, scenario3_parameters, scenario4_parameters]
    scenario_parameters_list = [scenario4_parameters]

    for scenario_parameters in scenario_parameters_list:
        parameters = generate_scenario(scenario_parameters)

        # create the generator
        solution_generator = matcher_optimization.MatcherSolutionGenerator(parameters)
        # create the evaluator, using the tight margin utility function
        solution_evaluator = matcher_optimization.MatcherSolutionEvaluator(matcher_optimization.MatcherSolutionEvaluator.tight_margin_utility)
        # create the visualizer
        solution_visualizer = matcher_optimization.MatcherSolutionVisualizer()

        for i in range(2):
            solution = solution_generator.get_solution()
            solution_visualizer.display_solution(parameters, solution)

        solution[solution.keys()[0]] = solution[solution.keys()[0]] + 1

        for i in range(10):
            closest_valid_solution = solution_generator.get_closest_valid_solution(solution)
            solution_visualizer.display_solution(parameters, closest_valid_solution)

        neighbor_solutions = solution_generator.get_neighborhood(solution)
        for neighbor_solution in neighbor_solutions:
            solution_visualizer.display_solution(parameters, neighbor_solution)

#LOG_FILENAME = '/tmp/logging_example.out'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.debug('This message should go to the log file')
#compare_optimizers()
#run_queens()
#run_matcher()

import cProfile
#cProfile.run("run_matcher()")
cProfile.run("compare_optimizers()")
