import time
import random
import logging
import numpy as np

import matcher_optimization
import optimization
import queens_optimization

from experiments import *

FORMAT = "%(asctime)-15s %(message)s"
""" The format for debugger messages """

LOGGING_LEVEL = logging.DEBUG
""" The configured logging level """

#LOG_FILENAME = '/tmp/logging_example.out'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)
logging.basicConfig(format=FORMAT, level=LOGGING_LEVEL)
""" The basic logging configuration """

def analyze_metaheuristics():
    # defining experimental setup
    #sampling_points = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    sampling_points = range(1000)
    sampling_points.remove(0)

    # Scenario 1: Highly competitive, tight market
    scenario1_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.10,
                  "lender_rate_standard_deviation" : 0.05,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.10,
                  "borrower_rate_standard_deviation" : 0.05,
                  "number_lenders" : 5,
                  "number_borrowers" : 5}

    optimizer_classes = [optimization.RandomSearchOptimizer,
                         optimization.HillClimbingOptimizer,
                         optimization.SimulatedAnnealingOptimizer,
                         optimization.GeneticAlgorithmOptimizer,
                         optimization.ParticleSwarmOptimizer
                         ]

    # generate the environment parameters
    parameters = generate_scenario(scenario1_parameters)

    experiment_results = {}
    number_runs = 1

    time_budget = None
    iterations_budget = 10

    for optimizer_class in optimizer_classes:
        run_results_list = []
        optimizer_results = {}
        experiment_results[optimizer_class] = {}

        for run in range(number_runs):
            # create the generator
            solution_generator = matcher_optimization.MatcherSolutionGenerator(parameters)
            # create the evaluator, using the tight margin utility function
            solution_evaluator = matcher_optimization.MatcherSolutionEvaluator(matcher_optimization.MatcherSolutionEvaluator.tight_margin_utility)
            # create the visualizer
            solution_visualizer = matcher_optimization.MatcherSolutionVisualizer()

            # create the coordinator, injecting the created objects
            optimizer = optimizer_class(solution_generator, solution_evaluator, solution_visualizer)

            # set the sampling points in the optimizer
            optimizer.set_sampling_points(sampling_points)

            # run the optimizer
            logging.info("Run number %d of optimizer %s" % (run, optimizer))
            score = run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget, iterations_budget)

            run_results = optimizer.get_results()
            run_results_list.append(run_results)

        # @todo: calculate the mean score for each sampling point
        iterations_list = run_results.keys()
        for number_iterations in iterations_list:
            results = [run_results[number_iterations] for run_results in run_results_list]
            optimizer_results[number_iterations] = np.mean(results)

        experiment_results[optimizer_class] = optimizer_results

    #print experiment_results
    export_csv(experiment_results)

def compare_optimizers():

    # number of times to run, to compute average
    #number_runs = 3
    number_runs = 2
    # number of sampling points for iterations
    #iterations_budgets = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    #iterations_budgets = [1, 5, 10, 50, 100, 500, 1000]
    iterations_budgets = [1, 100, 1000]

    # @todo: sync these scenarios with the dissertation text
    # aggressive lenders, conservative borrowers
    scenario1_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.05,
                  "lender_rate_standard_deviation" : 0.01,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.20,
                  "borrower_rate_standard_deviation" : 0.01,
                  "number_lenders" : 5,
                  "number_borrowers" : 5}
    # @todo: sync these scenarios with the dissertation text
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
    # @todo: sync these scenarios with the dissertation text
    # homogeneous mean, larger deviation
    scenario3_parameters = {"mean_lender_amount" : 1000,
                  "lender_amount_standard_deviation" : 100,
                  "mean_lender_rate" : 0.10,
                  "lender_rate_standard_deviation" : 0.10,
                  "mean_borrower_amount" : 1000,
                  "borrower_amount_standard_deviation" : 100,
                  "mean_borrower_rate" : 0.10,
                  "borrower_rate_standard_deviation" : 0.10,
                  "number_lenders" : 5,
                  "number_borrowers" : 5}
    # @todo: sync these scenarios with the dissertation text
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
    # @todo: sync these scenarios with the dissertation text
    # aggressive lenders, conservative borrowers, larger deviation
    scenario5_parameters = {"mean_lender_amount" : 1000,
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
    scenario_parameters_list = [scenario5_parameters]
    optimizer_classes = [#optimization.RandomSearchOptimizer,
                         #optimization.HillClimbingOptimizer,
                         optimization.SimulatedAnnealingOptimizer#,
                         #optimization.GeneticAlgorithmOptimizer,
                         #optimization.ParticleSwarmOptimizer
                         ]
    time_budget = None

#    optimizer_parameters = {optimization.SimulatedAnnealingOptimizer : {"initial_energy" : [10, 100, 1000]}}

    for scenario_parameters in scenario_parameters_list:
        parameters = generate_scenario(scenario_parameters)
        experiment_results = {}

        for optimizer_class in optimizer_classes:
            experiment_results[optimizer_class] = {}
            for iterations_budget in iterations_budgets:
                scores = []
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
                    score = run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget, iterations_budget)
                    scores.append(score)
                mean_score = np.mean(scores)
                experiment_results[optimizer_class][iterations_budget] = mean_score

    #print experiment_results
    export_csv(experiment_results)

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

#compare_optimizers()
analyze_metaheuristics()
#run_queens()
#run_matcher()

import cProfile
#cProfile.run("run_matcher()")
#cProfile.run("compare_optimizers()")
