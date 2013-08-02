#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parameters.py - graphene.py's parameter file
Defines parameters of the script.

Defines Graphene Lattice Constants, Units, Lattice Options (width, length, etc.),
Antidot Options (size, position, etc)


Trevin Gandhi

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301, USA.
"""

#
# Graphene Lattice Constants
#

ARMCHAIR = True      # True if armchair orientation, false if zigzag
CHAIN = False         # Is it a chain?
BUILD_HOR = True     # True = upwards then horizontal, False = horizontal then upwards
A = 1.42             # Carbon-carbon bond length in Angstroms
# (1.42 is an average of carbon single bonds (C-C) and carbon double bonds (C=C))
T = 2.7              # Tight-binding nearest-neighbor hopping parameter

if ARMCHAIR:
    W_LEG = 0.71      # Horizontal leg: Hypotenuse (1.42 Angstroms) / 2 because it's a 30-60-90 triangle
    DW_LEG = 2.84     # W_LEG * 4 (width of hexagon)
    H_LEG = 1.2306    # Vertical leg: W_LEG * sqrt(3) (for the other leg of the triangle)
    DH_LEG = 2.4612   # H_LEG * 2  (height of hexagon)
    X_DIST = 2.13     # 3a/2
    Y_DIST = H_LEG    # (sqrt(3)*a)/2
    Z_DIST = A        # Distance between two points in the unit cell
    if CHAIN:
        X_DIST = 1.42
        Y_DIST = 1.42
else:
    H_LEG = 0.71      # Vertical LEG: Hypotenuse (1.42 Angstroms) / 2 because it's a 30-60-90 triangle
    DH_LEG = 2.84     # H_LEG * 4 (width of hexagon)
    W_LEG = 1.2306    # Horizontal LEG: H_LEG * sqrt(3) (for the other LEG of the triangle)
    DW_LEG = 2.4612   # W_LEG * 2  (height of hexagon)

#
# Units
#

DISTANCE = True     # True if x and y are distances, False if they are numbers of atoms
NANOMETERS = True   # True if parameter units are in nanometers, false if in Angstroms

#
# General Lattice Parameters
#

WIDTH = 20             # Width of the unit cell
HEIGHT = 20              # Height of the unit cell
NUM_X_TRANS = 0    # Number of times to translate unit cell along the x-axis
NUM_Y_TRANS = 0    # Number of times to translate unit cell along the y-axis
CUT_TYPE = 1       # 0 if no antidots, 1 if rectangular

#
# Rectangular Antidot Parameters
#

ANTIDOT_X_NUM = 2   # Number of antidots along x
ANTIDOT_Y_NUM = 2   # Number of antidots along y
RECT_X = 4        # x-coordinate of the bottom left corner of the antidot
RECT_Y = 4        # y-coordinate of the bottom left corner of the antidot
RECT_H = 4        # Height of the antidot
RECT_W = 3        # Width of the antidot
BTW_X_DIST = 6      # Horizontal distance between antidots
BTW_Y_DIST = 5      # Horizontal distance between antidots

#
# Data Options
#

# Note: Must have COORD2_CREATION as True for PLOT_OPTION to work
PLOT_OPTION = True          # Plot the graphene sheet
COORD2_CREATION = True     # Create the coord2 array?


#
# Convert nanometers to angstroms
#

if NANOMETERS and DISTANCE:
    WIDTH *= 10
    HEIGHT *= 10
    RECT_X *= 10
    RECT_Y *= 10
    RECT_H *= 10
    RECT_W *= 10
    BTW_X_DIST *= 10
    BTW_Y_DIST *= 10