import time

import matcher_lib
import optimization_lib

def measured_run1():
    parameters = {"lenders" : {"1": {"minimum_rate" : 0.05, "minimum_amount" : 10, "maximum_amount" : 100},
                               "2": {"minimum_rate" : 0.04, "minimum_amount" : 20, "maximum_amount" : 200},
                               "3": {"minimum_rate" : 0.03, "minimum_amount" : 30, "maximum_amount" : 300}},
                  "borrowers" : {"4": {"maximum_rate" : 0.15, "minimum_amount" : 10, "maximum_amount" : 100},
                                 "5": {"maximum_rate" : 0.10, "minimum_amount" : 20, "maximum_amount" : 200},
                                 "6": {"maximum_rate" : 0.01, "minimum_amount" : 30, "maximum_amount" : 300}}}

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

    print "RandomSearchOptimizer"
    run_optimizer(parameters, random_search_optimizer, solution_evaluator, solution_visualizer, time_budget)

    print "HillClimbingOptimizer"
    run_optimizer(parameters, hill_climbing_optimizer, solution_evaluator, solution_visualizer, time_budget)

    print "SimulatedAnnealingOptimizer"
    run_optimizer(parameters, simulated_annealing_optimizer, solution_evaluator, solution_visualizer, time_budget)

    print "ParticleSwarmOptimizer"
    run_optimizer(parameters, particle_swarm_optimizer, solution_evaluator, solution_visualizer, time_budget)

def run_optimizer(parameters, optimizer, solution_evaluator, solution_visualizer, time_budget):
    optimizer.set_time_budget(time_budget)

    # run the metaheuristic
    start_time = time.time()
    solution = optimizer.optimize()
    end_time = time.time()

    duration = end_time - start_time
    iterations = optimizer.get_iterations()

    # evaluate the final solution
    utility = solution_evaluator.evaluate(parameters, solution)

    # display the results
    solution_visualizer.display(parameters, solution, utility)
    print "iterations: %s" % iterations
    print "elapsed time: %ss" % duration
    print "--"

measured_run1()
