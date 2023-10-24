# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:35:49 2023

@author: mw2119
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#%% JITTED: Functions jitted and called within DLA class in order to speed up code.

@njit
def distance(x, y):
    '''
    Calculates distance from the origin (0,0) to point (x, y)
    Input: 
    x: int
    y: int

    Return: float
    '''
    return np.sqrt(x*x+y*y)

@njit
def jitCircleBound(radius):
    '''
    Initialise circle at a specified radius via a list of integer points
    via equation x**2 + y**2 = radius**2
    Input:
    radius: float

    Return: 
    Circle: list(int)
    '''
    circle = []
    for x in range(-int(radius), int(radius)+1):
        y_max = int(np.sqrt(radius**2-x**2))
        for y in range(-y_max, y_max):
            circle.append((x, y))
    return circle

@njit
def jitRandPoint(circle):
    '''
    Return randomly chosen point from the circle perimeter.

    Input:
    Circle: list(int)

    Return: tuple(int)
    '''
    return circle[np.random.randint(0, len(circle))]

@njit
def jitGetNeighbours(radius, point):
    '''
    Return list of neighbouring points to specified points as long as points are
    not outside the initialised circle of given radius.
    
    Input:
    point: tuple(int)
    radius: int

    Return:
    list(tuple(int))
    '''
    x,y = point
    NN = [(x-1, y+1), (x, y+1), (x+1, y+1),
          (x-1, y),             (x+1, y),
          (x-1, y-1), (x, y-1), (x+1, y-1)]
    
    return [(x,y) for x, y in NN if int(x*x+y*y) <= radius**2 - 1]


def jitCheckStick(points, adj_points):
    '''
    Check if any points in points, i.e the cluster, share any points in adj_points. 
    This determines whether the randomly walking particle sticks to the cluster.

    Input:
    points: list(tuple(int))
    adj_points: list(tuple(int))

    Return: Bool
    '''
    return any(n in adj_points for n in points)

@njit
def jitMovePoint(adj_points):
    '''
    Move randomly walking particle by picking a random point in its neighbouring points

    Input:
    adj_points: list(tuple(int))

    Return:
    new_pos: tuple(int)
    '''
    new_pos = adj_points[np.random.randint(len(adj_points))]
    return new_pos

 
def jitGrow(points, boundary, radius, dr, no_points = 1):
    '''
    Grow cluster by picking a random point on the perimeter of the initial circle.
    Then move point to one of its neighbours (i.e an integer step) until one of its
    neighbouring points includes one of the points in the cluster.
    If the randomly walking point exits the initial circle + a small step dr, 
    it is respawned at another random point on the circle.
    Once the particle is registered as being stuck to the cluster, the list of points
    of the cluster is updated, the radius of the spawn circle expanded and the list
    of points making the spawn circle recalculated.
    no_points determines the number of points you wish to add to the cluster.

    Input:
    points: list(tuple(int))
    boundary: list(tuple(int))
    radius: float
    dr: int
    no_points: int, default set to 1

    Return:
    points: list(tuple(int))
    radius: float
    boundary: list(tuple(int))
    '''
    ticker = 0
    while ticker < no_points: #Break loop once no_points have been added to cluster
        spawn = jitRandPoint(boundary) #spawn random point on boundary of spawn circle
        adj_points = jitGetNeighbours(radius, spawn) # calculate neighbouring points
        while not jitCheckStick(points, adj_points): #Make sure none of the spawn point's neighbouring points are in the cluster
            spawn = jitMovePoint(adj_points) #If False, move point to one of its neighbouring points
            if spawn[0]**2 + spawn[1]**2 > (radius + dr)**2: #If exit the circle of radius radius + dr, relaunch particle on boundary 
                spawn = jitRandPoint(boundary)
            adj_points = jitGetNeighbours(radius, spawn)
        points.append(spawn) #Once Exiting the loop, append point to list of points within the cluster
        ticker += 1 #Update number of points added
        
        if distance(*spawn) + dr > radius: #If the newly added point is too close to the radius of the circle, update spawn circle and radius
            radius = int(distance(*spawn)) + dr
            boundary = jitCircleBound(radius)
        
    return points, radius, boundary

#%%

class DLA:
    def __init__(self, dr = 15):
        self.dr = dr
        self.radius = 1+dr
        self.points = [(0,0)] #Initialise with default single seed particle at origin
        self.boundary = []
        for x in range(-self.radius, self.radius+1): #Initialise spawn circle boundary 
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
            
            
            
        
        
        
    
        
        
        
    
