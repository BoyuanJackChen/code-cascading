def median(l: list):
    """Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """
    n = len(l)
    if n % 2 == 0:
        # If the list has an even number of elements, the median is the average of the two middle elements
        return (l[n//2 - 1] + l[n//2]) / 2
    else:
        # If the list has an odd number of elements, the median is the middle element
        return l[n//2]

print(median([-10, 4, 6, 1000, 10, 20]))
print(median([3, 1, 2, 4, 5]))

def median(l: list):
    """Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """
    n = len(l)
    if n % 2 == 0:
        return (l[n//2 - 1] + l[n//2]) / 2
    else:
        return l[n//2]

print(median([-10, 4, 6, 1000, 10, 20]))
print(median([3, 1, 2, 4, 5]))