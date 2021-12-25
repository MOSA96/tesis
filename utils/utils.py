import numpy as np

#Define your function here
def test(x):
    #if x > 1.0:
    #    return x**2
   # elif x== 1:
       # return 0
    return np.sin(x) + np.sin((10.0 / 3.0) * x)

def function(individual):
    x = individual[0]
    f = test(x)
    return f,

def ind(low=-20, high=20):
    return [np.random.uniform(low=low, high=high)]
