import itertools

MAX_RATE = 10000

def grouper(n, iterable, fillvalue=None):
    """
    grouper(3, 'ABCDEFG', 'x') yields ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def unique(s):
    """Return a list of the elements in s, but without duplicates.

    For example, unique([1,2,3,1,2,3]) is some permutation of [1,2,3],
    unique("abcabc") some permutation of ["a", "b", "c"], and
    unique(([1, 2], [2, 3], [1, 2])) some permutation of
    [[2, 3], [1, 2]].

    For best speed, all sequence elements should be hashable.  Then
    unique() will usually work in linear time.

    If not possible, the sequence elements should enjoy a total
    ordering, and if list(s).sort() doesn't raise TypeError it's
    assumed that they do enjoy a total ordering.  Then unique() will
    usually work in O(N*log2(N)) time.

    If that's not possible either, the sequence elements must support
    equality-testing.  Then unique() will usually work in quadratic
    time.

    @credits: Tim Peters
    @url: http://code.activestate.com/recipes/52560/
    """

    n = len(s)
    if n == 0:
        return []

    # Try using a dict first, as that's the fastest and will usually
    # work.  If it doesn't work, it will usually fail quickly, so it
    # usually doesn't cost much to *try* it.  It requires that all the
    # sequence elements be hashable, and support equality comparison.
    u = {}
    try:
        for x in s:
            u[x] = 1
    except TypeError:
        del u  # move on to the next method
    else:
        return u.keys()

    # We can't hash all the elements.  Second fastest is to sort,
    # which brings the equal elements together; then duplicates are
    # easy to weed out in a single pass.
    # NOTE:  Python's list.sort() was designed to be efficient in the
    # presence of many duplicate elements.  This isn't true of all
    # sort functions in all languages or libraries, so this approach
    # is more effective in Python than it may be elsewhere.
    try:
        t = list(s)
        t.sort()
    except TypeError:
        del t  # move on to the next method
    else:
        assert n > 0
        last = t[0]
        lasti = i = 1
        while i < n:
            if t[i] != last:
                t[lasti] = last = t[i]
                lasti += 1
            i += 1
        return t[:lasti]

    # Brute force is all that's left.
    u = []
    for x in s:
        if x not in u:
            u.append(x)
    return u


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
    solutions_map = build_solutions_map(parameters, solution)
    print_solutions_map(parameters, solutions_map)

def build_solutions_map(parameters, solution):
    solutions = {}

    lenders = parameters["lenders"]
    borrowers = parameters["borrowers"]

    for lender_id in lenders:
        lender_solutions = {}

        for borrower_id in borrowers:
            rate = solution["rate_%s_%s" % (lender_id, borrower_id)]
            rate = float(rate) / MAX_RATE * 100

            amount = solution["amount_%s_%s" % (lender_id, borrower_id)]

            solution_tuple = (amount, rate)

            lender_solutions[borrower_id] = solution_tuple

        # append the match list for the current lender to the overall match list
        solutions[lender_id] = lender_solutions

    return solutions

def print_match(amount, rate):
    print "(%6.2f@%5.2f%%)" % (amount, rate),

def print_utility(utility):
    print "Utility:"
    print "borrower_rates_margin: ", utility["borrower_rates_margin"]
    print "lender_rates_margin: ", utility["lender_rates_margin"]
    print "member_rates_margin", utility["member_rates_margin"]
    print "total_offered_amount", utility["total_offered_amount"]
    print "total_requested_amount", utility["total_requested_amount"]
    print "total_matched_amount", utility["total_matched_amount"]
    print "fulfillment_rate", utility["fulfillment_rate"]
    print "tightness", utility["tightness"]
    print "individual results: ", utility["results"]
    print "*score*: ", utility["score"]

def print_solutions_map(parameters, solutions_map):
    lenders = parameters["lenders"]
    borrowers = parameters["borrowers"]

    # print the header
    print "                   |",
    for borrower_id, borrower in borrowers.items():
        amount = borrower["maximum_amount"]
        rate = borrower["maximum_rate"] * 100
        print "B%s" % borrower_id,
        print_match(amount, rate)
        print "  ",

    print
    print ""

    # print the body
    i = 0
    for lender_id, lender_solutions in solutions_map.items():
        lender = lenders[lender_id]
        amount = lender["maximum_amount"]
        rate = lender["minimum_rate"] * 100
        print "L%s" % lender_id,
        print_match(amount, rate)
        print "|  ",

        for borrower_solution_amount, borrower_solution_rate in lender_solutions.values():
            print_match(borrower_solution_amount, borrower_solution_rate),
            print "  ",
        print ""
        i += 1
    print "   -------------------------------------------------------------------"
