from sklearn.preprocessing import MinMaxScaler
import numpy as np

weights = np.array([[115.], [140.], [175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights) # requires float

print(rescaled_weight)

"""Quiz: Quiz On Algorithms Requiring Rescaling:
Decision Trees use vertical and horizontal lines so there is no trade off.

 In linear regression, the coefficient and the feature always go together.

Algorithms in which two dimensions affect the  outcome will be affected by rescaling.

SVM with kernel RBF
K-Means Clustering need rescaling
"""