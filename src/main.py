import time

import matcher_lib
import optimization_lib

def measured_run1():
    parameters = {"lenders" : {"1": {"minimum_rate" : 0.05, "minimum_amount" : 10, "maximum_amount" : 100},
                               "2": {"minimum_rate" : 0.04, "minimum_amount" : 20, "maximum_amount" : 200},
                               "3": {"minimum_rate" : 0.03, "minimum_amount" : 30, "maximum_amount" : 300}},
                  "borrowers" : {"4": {"maximum_rate" : 0.15, "minimum_amount" : 10, "maximum_amount" : 300},
                                 "5": {"maximum_rate" : 0.10, "minimum_amount" : 20, "maximum_amount" : 300},
                                 "6": {"maximum_rate" : 0.01, "minimum_amount" : 30, "maximum_amount" : 300}}}

#    parameters = {"lenders" : {"1": {"minimum_rate" : 0.05, "minimum_amount" : 10, "maximum_amount" : 100},
#                               "2": {"minimum_rate" : 0.04, "minimum_amount" : 20, "maximum_amount" : 200}},
#                  "borrowers" : {"4": {"maximum_rate" : 0.15, "minimum_amount" : 10, "maximum_amount" : 100},
#                                 "5": {"maximum_rate" : 0.10, "minimum_amount" : 20, "maximum_amount" : 100}}}

    # create the generator
    solution_generator = matcher_lib.MatcherSolutionGenerator(parameters)

    # create the evaluator, using the tight margin utility function
    solution_evaluator = matcher_lib.MatcherSolutionEvaluator(matcher_lib.MatcherSolutionEvaluator.tight_margin_utility)

    # create the visualizer
    solution_visualizer = matcher_lib.MatcherSolutionVisualizer()

    # create the coordinator, injecting the created objects
    random_search_optimizer = optimization_lib.RandomSearchOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    hill_climbing_optimizer = optimization_lib.HillClimbingOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    simulated_annealing_optimizer = optimization_lib.SimulatedAnnealingOptimizer(solution_generator, solution_evaluator, solution_visualizer)
    particle_swarm_optimizer = optimization_lib.ParticleSwarmOptimizer(solution_generator, solution_evaluator, solution_visualizer)

    time_budget = 10
    iterations_budget = None

    #optimizers = [random_search_optimizer, hill_climbing_optimizer, simulated_annealing_optimizer, particle_swarm_optimizer]
    optimizers = [hill_climbing_optimizer]

    for optimizer in optimizers:
        run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget, iterations_budget)

def run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget=None, iterations_budget=None):
    print optimizer

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
    print "iterations: %s" % optimizer.get_last_run_iterations()
    print "elapsed time: %ss" % optimizer.get_last_run_duration()
    print "--"

measured_run1()
