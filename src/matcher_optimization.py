import constraint
import itertools
import logging
import statlib.stats
import time

import matcher_constraint
import matcher_utils

MAX_RATE = 10000

class MatcherSolutionGenerator(object):
    """
    Generates matcher solutions for the configured problem,
    according to a standard parameter specification for credit matching problems.
    Uses the matcher constraint library as the key solution generation mechanism.

    @capability: solution_generator(set_problem, get_solution, get_solution_iterator)
    """

    def __init__(self, parameters=None):
        if parameters:
            self.set_parameters(parameters)

    def set_parameters(self, parameters):
        self.parameters = parameters

        self.initialize_generator()

    def initialize_generator(self):
        self.solver = matcher_constraint.NeighborhoodBacktrackingSolver()

        self.problem = matcher_constraint.MatcherProblem(self.solver, [self.increment_variable, self.decrement_variable])

        self.create_variables()

        self.create_constraints()

        # @TODO: Shouldn't this be done Just In Time?
        self.solution_iterator = self.problem.getSolutionIter()

    def create_variables(self):
        logging.debug("creating variables")
        lenders = self.parameters["lenders"]
        borrowers = self.parameters["borrowers"]

        # each rate can range between 0.00 and 100.00%
        rate_range = range(1, MAX_RATE)

        for lender_id, lender in lenders.items():
            # each match amount is capped by the lender's availability
            lender_amount_range = range(0, lender["maximum_amount"])

            for borrower_id in borrowers:
                # add the rate match variable for the lender, borrower pair
                rate_variable = "rate_%s_%s" % (lender_id, borrower_id)
                self.problem.addVariable(rate_variable, rate_range)

                amount_variable = "amount_%s_%s" % (lender_id, borrower_id)
                # add the amount match variable for the lender, borrower pair
                self.problem.addVariable(amount_variable, lender_amount_range)

    def create_constraints(self):
        logging.debug("creating constraints")
        lenders = self.parameters["lenders"]
        borrowers = self.parameters["borrowers"]

        # add lender constraints
        for lender_id, lender in lenders.items():
            # build a list of all the amount variables for the current lender,
            lender_amounts = ["amount_%s_%s" % (lender_id, borrower_id) for borrower_id in borrowers]

            # constraint the amount variables' sum to the lender's maximum and minimum amounts
            lender_maximum_amount_constraint = constraint.MaxSumConstraint(lender["maximum_amount"])
            lender_minimum_amount_constraint = constraint.MinSumConstraint(lender["minimum_amount"])

            # apply the constraints to the problem
            self.problem.addConstraint(lender_maximum_amount_constraint, lender_amounts)
            self.problem.addConstraint(lender_minimum_amount_constraint, lender_amounts)

            # build a list of all the rate variables for the current lender
            lender_rates = ["rate_%s_%s" % (lender_id, borrower_id) for borrower_id in borrowers]
            # build a list of amount, rate pairs to use as input for the weighted average rate
            lender_amounts_rates = matcher_utils.flatten(zip(lender_amounts, lender_rates))

            # constraint the average rate (weighed by the amount) to the lender's minimum accepted rate
            lender_minimum_rate = round(lender["minimum_rate"] * MAX_RATE)
            lender_minimum_rate_constraint = matcher_constraint.MinWeightedAverageOrDefaultConstraint(lender_minimum_rate, 0)

            # apply the constraint to the problem
            self.problem.addConstraint(lender_minimum_rate_constraint, lender_amounts_rates)

        # add borrower constraints
        for borrower_id, borrower in borrowers.items():
            # build a list of all the amount variables for the current borrower
            borrower_amounts = ["amount_%s_%s" % (lender_id, borrower_id) for lender_id in lenders]

            # constraint the amount variables' sum to the lender's maximum and minimum amounts
            borrower_maximum_amount_constraint = constraint.MaxSumConstraint(borrower["maximum_amount"])
            borrower_minimum_amount_constraint = constraint.MinSumConstraint(borrower["minimum_amount"])

            # apply the constraints to the problem
            self.problem.addConstraint(borrower_maximum_amount_constraint, borrower_amounts)
            self.problem.addConstraint(borrower_minimum_amount_constraint, borrower_amounts)

            # build a list of all the rate variables for the current borrower
            borrower_rates = ["rate_%s_%s" % (lender_id, borrower_id) for lender_id in lenders]
            # build a list of amount, rates pairs to use as input for the weighted average rate
            borrower_amounts_rates = matcher_utils.flatten(zip(borrower_amounts, borrower_rates))

            # constraint the average rate (weighed by the amount) to the borrower's maximum offered rate
            borrower_maximum_rate = round(borrower["maximum_rate"] * MAX_RATE)
            borrower_maximum_rate_constraint = matcher_constraint.MaxWeightedAverageConstraint(borrower_maximum_rate)

            # apply the constraint to the problem
            self.problem.addConstraint(borrower_maximum_rate_constraint, borrower_amounts_rates)

    def increment_variable(self, variable, solution, domains):
        new_solution = solution.copy()

        # apply the operator to the specified variable
        new_solution[variable] = solution[variable] + 1

        # if the new variable value belongs to the domain
        if new_solution[variable] in domains[variable]:
            # return the computed solution
            return new_solution
        # if the resulting solution from applying the operator is
        # not in the variable's domain, return None
        else:
            return None

    def decrement_variable(self, variable, solution, domains):
        new_solution = solution.copy()

        new_solution[variable] = solution[variable] - 1

        # if the new variable value belongs to the domain
        if new_solution[variable] in domains[variable]:
            # return the computed solution
            return new_solution
        # if the resulting solution from applying the operator is
        # not in the variable's domain, return None
        else:
            return None

    def get_solution(self):
        logging.debug("solution requested")
        # @todo: return the solution affected by MAX_RATE to standardize the generator API (always 1=100%)
        solution = self.solution_iterator.next()
        logging.debug("solution retrieved")

        return solution

    def get_solution_iterator(self):
        return self.solution_iterator

    def get_parameters(self):
        return self.parameters

    def get_neighborhood(self, solution):
        logging.debug("neighborhood requested")
        neighborhood = self.problem.getNeighborhood(solution)
        logging.debug("neighborhood retrieved")

        return neighborhood

    def get_closest_valid_solution(self, candidate_solution):
        logging.debug("closest solution requested")
        solution = self.problem.getClosestValidSolution(candidate_solution)
        logging.debug("closest solution retrieved")

        return solution

    def get_variables(self):
        return self.problem.getVariables()

class MatcherSolutionEvaluator(object):
    """
    Evaluates a solution according to a selected utility function.
    """

    def __init__(self, utility_function):
        self.set_utility_function(utility_function)

    def set_utility_function(self, utility_function):
        self.utility_function = utility_function

    def evaluate(self, parameters, solution):
        """
        Aggregates the standard format solution into a result map, similar to the problem parameters.
        Compute the utility function for the specified solution.
        """
        # compute the aggregate results, for each participant
        results = self.calculate_aggregate_results(parameters, solution)

        # evaluate the utility function for the specified parameters and
        # the results computed from the specified solution
        return self.utility_function.__call__(self, parameters, results)

    def calculate_aggregate_results(self, parameters, solution):
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
                try:
                    lender_borrower_match_amount = solution[lender_borrower_match_amount_variable]
                except:
                    raise
                lender_amount += lender_borrower_match_amount

                lender_borrower_match_rate_variable = "rate_%s_%s" % (lender_id, borrower_id)
                lender_borrower_match_rate = solution[lender_borrower_match_rate_variable]
                # @todo: MAX_RATE is due to the FD-CLP, it should not be necessary elsewhere
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

    # @TODO: refactor the utility functions: use fitness_map as the return, and store the function value at the "fitness" index
    def tight_margin_utility(self, parameters, results):
        # calculate the overall member rate margin
        lender_rates_margin = self.calculate_lender_rates_margin(parameters, results)
        borrower_rates_margin = self.calculate_borrower_rates_margin(parameters, results)

        member_rates_margin = lender_rates_margin + borrower_rates_margin

        # calculate the total amount offered by lenders
        total_offered_amount = self.calculate_total_offered_amount(parameters, results)
        # calculate the total amount requested by borrowers
        total_requested_amount = self.calculate_total_requested_amount(parameters, results)
        # calculate the total capital amount successfully matched
        total_matched_amount = self.calculate_total_matched_amount(parameters, results)

        # calculate the part of the overall amount requests and offered that were successfully matched
        fulfillment_rate = self.calculate_fulfillment_rate(total_offered_amount, total_requested_amount, total_matched_amount)

        # calculate the tightness/fairness of the results
        tightness = self.calculate_tightness(parameters, results)

        # compute the composite utility of the specified solution results
        utility = {}

        utility["results"] = results
        utility["borrower_rates_margin"] = borrower_rates_margin
        utility["lender_rates_margin"] = lender_rates_margin
        utility["member_rates_margin"] = member_rates_margin
        utility["total_offered_amount"] = total_offered_amount
        utility["total_requested_amount"] = total_requested_amount
        utility["total_matched_amount"] = total_matched_amount
        utility["fulfillment_rate"] = fulfillment_rate
        utility["tightness"] = tightness
        utility["score"] = member_rates_margin * tightness * fulfillment_rate

        return utility

    def calculate_lender_rates_margin(self, parameters, results):
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


    def calculate_borrower_rates_margin(self, parameters, results):
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

    def calculate_total_offered_amount(self, parameters, results):
        lender_parameters = parameters["lenders"]

        total_offered_amount = 0
        for lender_id, lender_parameter in lender_parameters.items():
            total_offered_amount += lender_parameter["maximum_amount"]

        return total_offered_amount

    def calculate_total_requested_amount(self, parameters, results):
        borrower_parameters = parameters["borrowers"]

        total_requested_amount = 0
        for borrower_id, borrower_parameter in borrower_parameters.items():
            total_requested_amount += borrower_parameter["maximum_amount"]

        return total_requested_amount

    def calculate_total_matched_amount(self, parameters, results):
        borrower_results = results["borrowers"]

        total_matched_amount = 0
        for borrower_id, borrower_result in borrower_results.items():
            total_matched_amount += borrower_result["amount"]

        return total_matched_amount

    def calculate_fulfillment_rate(self, total_offered_amount, total_requested_amount, total_matched_amount):
        """ Calculates the relevance of the matched amount in relation to the overall initial values. """
        return total_matched_amount * 2 / float(total_offered_amount + total_requested_amount)

    def calculate_tightness(self, parameters, results):
        # calculate the overall member rate margin
        lender_rates_margins = self.calculate_lender_rates_margins(parameters, results)
        borrower_rates_margins = self.calculate_borrower_rates_margins(parameters, results)

        member_rate_margins = lender_rates_margins + borrower_rates_margins

        member_rate_margins_standard_deviation = statlib.stats.stdev(member_rate_margins)

        # the less the results vary, the tighter is the solution
        tightness = 1 / (1 + member_rate_margins_standard_deviation)

        return tightness

    def calculate_lender_rates_margins(self, parameters, results):
        lender_parameters = parameters["lenders"]
        lender_results = results["lenders"]

        lender_rate_margins = []

        for lender_id in lender_parameters:
            minimum_rate = lender_parameters[lender_id]["minimum_rate"]
            effective_rate = lender_results[lender_id]["rate"]

            # the lender margin is the difference between the rate that the matches actually yielded
            # and the minimum rate it was willing to accept
            lender_rate_margin = effective_rate - minimum_rate
            lender_rate_margins.append(lender_rate_margin)

        return lender_rate_margins

    def calculate_borrower_rates_margins(self, parameters, results):
        borrower_parameters = parameters["borrowers"]
        borrower_results = results["borrowers"]

        borrower_rate_margins = []

        for borrower_id in borrower_parameters:
            maximum_rate = borrower_parameters[borrower_id]["maximum_rate"]
            effective_rate = borrower_results[borrower_id]["rate"]

            # the borrower margin is the difference between the maximum rate it would have accepted
            # and the rate that the matches actually yielded
            borrower_rate_margin = maximum_rate - effective_rate
            borrower_rate_margins.append(borrower_rate_margin)

        return borrower_rate_margins

class MatcherSolutionVisualizer(object):
    def __init__(self, refresh_display_buffer=0):
        # initialize the double buffering control values
        self.refresh_display_buffer = refresh_display_buffer
        self.next_display_time = None

    def display_solution(self, parameters, solution):
        matcher_utils.print_solution(parameters, solution)

    def display_utility(self, utility):
        matcher_utils.print_utility(utility)

    def display(self, parameters, solution, utility):

        if time.time() > self.next_display_time:
            # update the next buffer refresh time
            self.next_display_time = time.time() + self.refresh_display_buffer

            # show the current status
            print
            self.display_solution(parameters, solution)
            self.display_utility(utility)
            print
