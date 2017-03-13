import math

#sum all the fractions from all the classes, in this case there are 2 classes
#pi = fraction of samples  belonging to class 1
entropy = -0.5*math.log(0.5, 2) - 0.5*math.log(0.5, 2)
print(entropy)

#entropy = 1.0 Maximally impure sample

#information gain = entropy(parent) - [weigthed average]entropy(children)
#decision tree algorithm: Maximize information gain

#entropy child
entropy2 = -0.6667*math.log(0.6667, 2) - 0.3333*math.log(0.3333, 2)
print(entropy2)
#entropy 2 = 0.918262497114

#information gain
#entropy(children) = 3/4(0.9184) + 1/4(0)

entropy_children = (0.75)*entropy2 + (0.25)*0
info_gain = entropy - entropy_children

print(info_gain)

