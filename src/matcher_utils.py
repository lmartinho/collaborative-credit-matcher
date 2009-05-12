import itertools

MAX_RATE = 10000

def grouper(n, iterable, fillvalue=None):
    """
    grouper(3, 'ABCDEFG', 'x') yields ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def flatten(l):
    return list(itertools.chain(*l))

def weighted_average_args(*args):
    # group the list into 2 value tuples
    grouped_arguments_iterator = grouper(2, args, 1)

    return weighted_average(grouped_arguments_iterator)

def weighted_average(iterable):  
    
    weighted_total = 0
    total_weight = 0
    for weight, value in iterable:
        weighted_total += weight * value
        total_weight += weight

    # if the total weight is zero, return 0 to prevent ZeroDivisionError
    if(total_weight == 0):
        return 0

    # compute the weighted average
    weighted_average = weighted_total / total_weight
    
    return weighted_average

def print_solution(parameters, solution):
    solution_list = build_solution_list(parameters, solution)
    print_solution_list(solution_list)

def build_solution_list(parameters, solution):
    solutions = []

    lenders = parameters["lenders"]
    borrowers = parameters["borrowers"]

    for lender_id in lenders:
        lender_match_list = []

        for borrower_id in borrowers:
            rate = solution["rate_%s_%s" % (lender_id, borrower_id)]
            rate = float(rate) / MAX_RATE * 100

            amount = solution["amount_%s_%s" % (lender_id, borrower_id)]

            solution_tuple = (amount, rate)

            lender_match_list.append(solution_tuple)
        
        # append the match list for the current borrower to the overall match list
        solutions.append(lender_match_list)
    
    return solutions

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

def print_match(match):
    amount, rate = match

    print "(%7.2fEUR" % amount,
    print("@"),
    print "%5.2f%%)" % rate,
