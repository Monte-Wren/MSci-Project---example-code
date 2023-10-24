# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:35:49 2023

@author: mw2119
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#%% JITTED

@njit
def distance(x, y):
    return np.sqrt(x*x+y*y)

@njit
def jitCircleBound(radius):
    circle = []
    for x in range(-int(radius), int(radius)+1):
        y_max = int(np.sqrt(radius**2-x**2))
        for y in range(-y_max, y_max):
            circle.append((x, y))
    return circle

@njit
def jitRandPoint(circle):
    return circle[np.random.randint(0, len(circle))]

@njit
def jitGetNeighbours(radius, point):
    x,y = point
    NN = [(x-1, y+1), (x, y+1), (x+1, y+1),
          (x-1, y),             (x+1, y),
          (x-1, y-1), (x, y-1), (x+1, y-1)]
    
    return [(x,y) for x, y in NN if int(x*x+y*y) <= radius**2 - 1]


def jitCheckStick(points, adj_points):
    return any(n in adj_points for n in points)

@njit
def jitMovePoint(adj_points):
    new_pos = adj_points[np.random.randint(len(adj_points))]
    return new_pos

 
def jitGrow(points, boundary, radius, dr, no_points = 1):
    ticker = 0
    while ticker < no_points:
        spawn = jitRandPoint(boundary)
        adj_points = jitGetNeighbours(radius, spawn)
        while not jitCheckStick(points, adj_points):
            spawn = jitMovePoint(adj_points)
            if spawn[0]**2 + spawn[1]**2 > (radius + dr)**2:
                spawn = jitRandPoint(boundary)
            adj_points = jitGetNeighbours(radius, spawn)
        points.append(spawn)
        ticker += 1
        
        if distance(*spawn) + dr > radius:
            radius = int(distance(*spawn)) + dr
            boundary = jitCircleBound(radius)
        
    return points, radius, boundary

#%%

class DLA:
    def __init__(self, dr = 15):
        self.dr = dr
        self.radius = 1+dr
        self.points = [(0,0)]
        self.boundary = []
        for x in range(-self.radius, self.radius+1):
            for y in range(-self.radius, self.radius+1):
                if (int(x*x + y*y) <= self.radius**2 and (x, y) != (0,0)):
                    self.boundary.append((x, y))
    
    def distance(self, x, y):
        return np.sqrt(x*x+y*y)
    
    def circleBound(self):
        return jitCircleBound(self.radius)
    
    def randPoint(self, circle):
        return jitRandPoint(circle)
        
    def getNeighbours(self, point):        
        return jitGetNeighbours(self.radius, point)
    
    def checkStick(self, adj_points):
        return jitCheckStick(self.points, adj_points)
    
    def movePoint(self, adj_points):
        return jitMovePoint(adj_points)
        
        
    def grow(self, no_points = 1):
        ticker = 0
        while ticker < no_points:
            spawn = self.randPoint(self.boundary)
            adj_points = self.getNeighbours(spawn)
            while not self.checkStick(adj_points):
                spawn = self.movePoint(adj_points)
                if spawn[0]**2 + spawn[1]**2 > (self.radius + self.dr)**2:
                    spawn = self.randPoint(self.boundary)
                adj_points = self.getNeighbours(spawn)
                
            self.points.append(spawn)
            ticker += 1
            
            if self.distance(*spawn) + self.dr > self.radius:
                self.radius = int(self.distance(*spawn)) + self.dr
                self.boundary = self.circleBound()
            
    def plot(self, size):
        grid = np.ones((size, size))
        for x, y in self.points:
            grid[x+int(size/2), y+int(size/2)] = 0
        
        plt.imshow(grid, cmap = 'gray')
        plt.show()
            
            
            
        
        
        
    
        
        
        
    