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

