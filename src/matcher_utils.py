import itertools

def grouper(n, iterable, fillvalue=None):
    """
    grouper(3, 'ABCDEFG', 'x') yields ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def flatten(l):
    return list(itertools.chain(*l))

def weighted_average(*args):  
    # group the list into 2 value tuples
    grouped_arguments_iterator = grouper(2, args, 1)
    
    weighted_total = 0
    total_weight = 0
    for weight, value in grouped_arguments_iterator:
        print "weight: %s, value : %s" % (weight, value)
        weighted_total += weight * value
        total_weight += weight
    
    # compute the weighted average
    weighted_average = weighted_total / total_weight
    
    return weighted_average

