#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    cnt = len(ages)
    errors = abs(predictions - net_worths)
    bar = sorted(errors)[round(0.9 * cnt)][0]
    
    for i, e in enumerate(errors):
        if e < bar:
            cleaned_data.append((ages[i][0], net_worths[i][0], errors[i][0]))
    
    return cleaned_data

