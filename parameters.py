#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parameters.py - graphene.py's parameter file
Defines parameters of the script.

Defines Graphene Lattice Constants, Units, Lattice Options (width, length, etc.),
Antidot Options (size, position, etc)


Trevin Gandhi, based off work by Frank Tseng

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

armchair = True      # True if armchair orientation, false if zigzag
chain = False         # Is it a chain?
build_hor = True     # True = upwards then horizontal, False = horizontal then upwards
a = 1.42             # Carbon-carbon bond length in Angstroms
# (1.42 is an average of carbon single bonds (C-C) and carbon double bonds (C=C))

if armchair:
    w_leg = 0.71      # Horizontal leg: Hypotenuse (1.42 Angstroms) / 2 because it's a 30-60-90 triangle
    dw_leg = 2.84     # w_leg * 4 (width of hexagon)
    h_leg = 1.2306    # Vertical leg: w_leg * sqrt(3) (for the other leg of the triangle)
    dh_leg = 2.4612   # h_leg * 2  (height of hexagon)
    x_dist = 2.13     # 3a/2
    y_dist = h_leg    # (sqrt(3)*a)/2
    z_dist = a        # Distance between two points in the unit cell
    if chain:
        x_dist = 1.42
        y_dist = 1.42
else:
    h_leg = 0.71      # Vertical leg: Hypotenuse (1.42 Angstroms) / 2 because it's a 30-60-90 triangle
    dh_leg = 2.84     # h_leg * 4 (width of hexagon)
    w_leg = 1.2306    # Horizontal leg: h_leg * sqrt(3) (for the other leg of the triangle)
    dw_leg = 2.4612   # w_leg * 2  (height of hexagon)

#
# Units
#

distance = True     # True if x and y are distances, False if they are numbers of atoms
nanometers = True   # True if parameter units are in nanometers, false if in Angstroms

#
# General Lattice Parameters
#

width = 20             # Width of the unit cell
height = 20              # Height of the unit cell
num_x_trans = 0    # Number of times to translate unit cell along the x-axis
num_y_trans = 0    # Number of times to translate unit cell along the y-axis
cut_type = 0       # 0 if no antidots, 1 if rectangular

#
# Rectangular Antidot Parameters
#

rect_x = 1        # x-coordinate of the bottom left corner of the antidot
rect_y = 1        # y-coordinate of the bottom left corner of the antidot
rect_h = 5        # Height of the antidot
rect_w = 5        # Width of the antidot

#
# Data Options
#

plot_option = False # Plot the graphene sheet
coord2_creation = False # Create the coord2 array?