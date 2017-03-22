""" quiz materials for feature scaling clustering """


### FYI, the most straightforward implementation might
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    data = []
    for i in arr:
        sc = (i - arr[0]) / float((arr[2] - arr[0]))
        data.append(sc)
    return data


data = [115, 140, 175]
print featureScaling(data)

"""Something to think about: What if x_max and x_min
are the same? For example, suppose the list of input features is
[10, 10, 10]--the denominator will be zero. Our suggestion would
be in general to assign each new feature to 0.5 (halfway between 0.0
and 1.0), but it's really your call. The main point is that this exact
formula can be broken."""