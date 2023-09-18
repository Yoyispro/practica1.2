# import math library
from math import *
from decimal import Decimal
 
# Function distance between two points
# and calculate distance value to given
# root value(p is root value)
def p_root(value, root):
     
    root_value = 1 / float(root)
    return round (Decimal(value) **
             Decimal(root_value), 3)
 
def minkowski_distance(x, y, p_value):
     
    # pass the p_root function to calculate
    # all the value of vector parallelly
    return (p_root(sum(pow(abs(a-b), p_value)
            for a, b in zip(x, y)), p_value))
 
# Driver Code
vector1 = [0, 2, 3, 4]
vector2 = [2, 4, 3, 7]
p = 3
print(minkowski_distance(vector1, vector2, p))
