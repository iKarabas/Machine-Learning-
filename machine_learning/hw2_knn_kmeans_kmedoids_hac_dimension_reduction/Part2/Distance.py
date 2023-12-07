import numpy as np
import math

class Distance:
    @staticmethod
    def calculateCosineDistance(x, y, temp=None):
        dot_product = np.dot(x,y)
        x_len = np.sqrt(np.sum(x**2))
        y_len = np.sqrt(np.sum(y**2))
        return 1 - (dot_product/(x_len*y_len))
        
    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        return np.power(np.sum(np.abs(x - y) ** p), 1/p)
        
    @staticmethod
    def calculateMahalanobisDistance(x,y, S_minus_1):
        x_minus_y = x - y 
        first_calc = np.matmul(x_minus_y.T, S_minus_1)
        return np.sqrt(np.matmul(first_calc,x_minus_y) )

