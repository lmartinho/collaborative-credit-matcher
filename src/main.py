import time
import random
import logging

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

def measured_run1():

#    # homogeneous mean, larger deviation
#    parameters = {"mean_lender_amount" : 1000,
#                  "lender_amount_standard_deviation" : 100,
#                  "mean_lender_rate" : 0.10,
#                  "lender_rate_standard_deviation" : 0.05,
#                  "mean_borrower_amount" : 1000,
#                  "borrower_amount_standard_deviation" : 100,
#                  "mean_borrower_rate" : 0.10,
#                  "borrower_rate_standard_deviation" : 0.05,
#                  "number_lenders" : 10,
#                  "number_borrowers" : 10}
#
#    # conservative lenders, aggressive borrowers
#    parameters = {"mean_lender_amount" : 1000,
#                  "lender_amount_standard_deviation" : 100,
#                  "mean_lender_rate" : 0.20,
#                  "lender_rate_standard_deviation" : 0.01,
#                  "mean_borrower_amount" : 1000,
#                  "borrower_amount_standard_deviation" : 100,
#                  "mean_borrower_rate" : 0.05,
#                  "borrower_rate_standard_deviation" : 0.01,
#                  "number_lenders" : 10,
#                  "number_borrowers" : 10}
#
#    # aggressive lenders, conservative borrowers
#    parameters = {"mean_lender_amount" : 1000,
#                  "lender_amount_standard_deviation" : 100,
#                  "mean_lender_rate" : 0.05,
#                  "lender_rate_standard_deviation" : 0.01,
#                  "mean_borrower_amount" : 1000,
#                  "borrower_amount_standard_deviation" : 100,
#                  "mean_borrower_rate" : 0.20,
#                  "borrower_rate_standard_deviation" : 0.01,
#                  "number_lenders" : 10,
#                  "number_borrowers" : 10}

    # homogeneous mean, smaller deviation
    scenario_generator_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.05,
                  "lender_rate_standard_deviation" : 0.01,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.20,
                  "borrower_rate_standard_deviation" : 0.01,
                  "number_lenders" : 5,
                  "number_borrowers" : 5}

    parameters = generate_scenario(scenario_generator_parameters)

    # create the generator
    solution_generator = matcher_optimization.MatcherSolutionGenerator(parameters)

    # create the evaluator, using the tight margin utility function
    solution_evaluator = matcher_optimization.MatcherSolutionEvaluator(matcher_optimization.MatcherSolutionEvaluator.tight_margin_utility)

    # create the visualizer
    solution_visualizer = matcher_optimization.MatcherSolutionVisualizer()

    # create the coordinator, injecting the created objects
    random_search_optimizer = optimization.RandomSearchOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    hill_climbing_optimizer = optimization.HillClimbingOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    simulated_annealing_optimizer = optimization.SimulatedAnnealingOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    particle_swarm_optimizer = optimization.ParticleSwarmOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    genetic_algorithm_optimizer = optimization.GeneticAlgorithmOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    genetic_algorithm_optimizer.set_maximum_trait_value(10000)

    time_budget = 40
    iterations_budget = None

    #optimizers = [random_search_optimizer, hill_climbing_optimizer, simulated_annealing_optimizer, particle_swarm_optimizer]
    optimizers = [hill_climbing_optimizer]
    #optimizers = [hill_climbing_optimizer]

    for optimizer in optimizers:
        run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget, iterations_budget)

def run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget=None, iterations_budget=None):
    logging.info("Running %s" % optimizer)

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


#LOG_FILENAME = '/tmp/logging_example.out'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logging.debug('This message should go to the log file')
measured_run1()
#import cProfile
#cProfile.run("measured_run1()")
