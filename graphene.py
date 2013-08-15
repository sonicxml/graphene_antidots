#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
graphene.py - Atomic Coordinate Generator, Hamiltonian Generator, and
Density of States Calculator for graphene lattices

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


from __future__ import division     # Use Python3 division
import gc as garcol                 # Garbage collection
import matplotlib.pyplot as plt     # matplotlib
import numpy as np                  # NumPy
import scipy.sparse as sparse       # Used for sparse matrices
from scipy import linalg            # Linear Algebra Functions
import scipy.io as sio              # Used for saving arrays as MATLAB files
import scipy.sparse.linalg as spla  # Sparse Matrix Linear Algebra Functions
from progressbar import Bar, Counter, ETA, Percentage, ProgressBar, Timer
from parameters import *
np.use_fastnumpy = True  # Enthought Canopy comes with "fastnumpy", so enable it

#
# Getting Started
#

# Before using this program, you must install:
#   Python 2.7.x (If using Python 3.x, you must change xrange() to range())
#   NumPy Python module
#   SciPy Python module
#   matplotlib Python module
# To run, in a python shell type %run graphene.py (for Windows machines)
# To change parameters of the graphene lattice,
# open the parameters.py module and change away


# To-Do list:
# TODO: Add periodic boundary conditions in Hamiltonian calculation - STARTED
# TODO: Clean up code according to Python standards - STARTED
# TODO: LOWER PRIORITY: Use unit vectors for coordinate generation - NOT STARTED
# TODO: LOW PRIORITY: Define X_DIST, Y_DIST, and Z_DIST
#       for zigzag orientation && check zigzag generation - NS
# TODO: LOW PRIORITY: Finish translator() - NOT STARTED

# Module options to possibly add: numexpr, Theano, pytables, cython, pysparse, numba
# http://technicaldiscovery.blogspot.com/2011/06/speeding-up-python-numpy-cython-and.html
# http://www.physics.udel.edu/~bnikolic/teaching/phys824/MATLAB/
# http://deeplearning.net/software/theano/
# http://lamp.tu-graz.ac.at/~hadley/ss1/bands/tightbinding/tightbinding.php
# http://www2.physics.ox.ac.uk/sites/default/files/BandMT_04.pdf
# http://www.physics.ucdavis.edu/Classes/Physics243A/TightBinding.Basics.Omar.pdf

def coord_generator():
    """
    Corrects for Units, Calculates Parameters,
    and Calls the Coordinate Generator Function
    """

    x_diff, y_diff = None, None

    if DISTANCE:
        # Limits for the loops
        # x_limit
        if WIDTH > DW_LEG:
            x_diff = (WIDTH % DW_LEG)
        x_limit = WIDTH

        # y_limit
        if HEIGHT > DH_LEG:
            y_diff = (HEIGHT % DH_LEG)
        y_limit = HEIGHT

        x_times = round(x_limit // X_DIST)
        y_times = round(y_limit / Y_DIST)

        print("Parameters calculated:")
        print("x_limit: %s" % str(x_limit))
        print("y_limit: %s" % str(y_limit))
        if x_diff:
            print("x_diff: %s" % str(x_diff))
        if y_diff:
            print("y_diff: %s" % str(y_diff))
    else:
        x_times, y_times = WIDTH, HEIGHT
        print("Parameters calculated:")

    print("x_times: %s" % str(x_times))
    print("y_times: %s" % str(y_times))
    print("Beginning coordinate generation")
    if COORD2_CREATION:
        (coord, coord2, x_atoms, y_atoms) = v_generator(x_times, y_times)
        print("Coordinate generation finished")
        if PLOT_OPTION:
            plot_graphene(coord2)
            print("Graphene plot drawn")
            del coord2
    else:
        (coord, x_atoms, y_atoms) = v_generator(x_times, y_times)
        print("Coordinate generation finished")

    return coord, x_atoms, y_atoms


def v_generator(x_times, y_times):
    """
    Creates a 3-dimensional array (n x 1 x 3) 'coord'
    Where n is the number of atoms in the sheet
    Defined by the unit cell consisting of two atoms,
    one a horizontal translation of the other
    Each atomic coordinate is defined by 3 numbers:
      A x-value (coord_x): As the unit cell is translated horizontally,
                           the x-value increments by 1
      A y-value (coord_y): As the unit cell is translated vertically,
                           the y-value increments by 1
      A unit cell value (coord_u): Declares atom's sub-lattice
          0 = the atom on the left (A sub-lattice)
          1 = the atom on the right (B sub-lattice)
    To convert these numbers into xyz coordinates,
      If a = 1.42
      ((coord_x, coord_y, coord_u).((3a/2), 0, a),
       (coord_x, coord_y, coord_u).(0, (sqrt(3)*a)/2, 0))
      Where . represents the dot product
    """

    if CHAIN:
        out_inc, in_inc = 1, 1
    elif BUILD_HOR:
        out_inc, in_inc = 1, 2
    else:
        out_inc, in_inc = 2, 1

    #
    # Coordinate Generator
    # TODO: Make it work with odd numbers of atoms on x-axis
    #

    j = np.arange(0, int(x_times if BUILD_HOR else y_times), out_inc)
    i = np.arange(0, int(y_times if BUILD_HOR else x_times), in_inc)
    j_shape = j.shape[0]
    i_shape = i.shape[0]
    j = np.repeat(j, i_shape)
    i = np.tile(i, j_shape)
    j = np.reshape(j, (j.shape[0], 1))
    i = np.reshape(i, (i.shape[0], 1))
    k = np.zeros_like(i)
    coord = np.dstack((j, i, k))
    del j_shape, i_shape, j, i, k
    if not CHAIN:
        coord = np.repeat(coord, 2, axis=0)
        coord[1::2][:, 0, 2] += 1
        coord[:, 0, 1][np.where(coord[:, 0, 0] % 2 != 0)] += 1
    x_atoms = (np.amax(coord[:, 0, 0]) + 1) * (2 if not CHAIN else 1)
    y_atoms = np.amax(coord[:, 0, 1]) + 1

    if TRIM_EDGES:
        x_max = np.amax(coord[:, 0, 0])
        idx = ((coord[:, 0, 0] == 0) & (coord[:, 0, 2] == 0)) & \
              (coord[:, 0, 1] % 2 == 0)
        idx2 = ((coord[:, 0, 0] == x_max) & (coord[:, 0, 2] == 1)) & \
               (coord[:, 0, 1] % 2 == (0 if x_times % 2 == 1 else 1))
        idx = ~(idx | idx2)
        coord = coord[idx]

    print(np.shape(coord))

    print("Number of atoms along the x-axis: %s" % str(x_atoms))
    print("Number of atoms along the y-axis: %s" % str(y_atoms))

    #
    # Antidot Generator
    #

    if COORD2_CREATION or CUT_TYPE:
        for ii in xrange(ANTIDOT_X_NUM):
            for jj in xrange(ANTIDOT_Y_NUM):
                coord_x = ((coord[:, 0, 0] * X_DIST +
                            coord[:, 0, 2] * Z_DIST).reshape(coord.shape[0], 1))
                coord_y = (coord[:, 0, 1] * Y_DIST).reshape(coord.shape[0], 1)

                if CUT_TYPE:
                    # Get bottom left x and y values
                    rect_x2 = RECT_X + ii * (BTW_X_DIST + RECT_W)
                    rect_y2 = RECT_Y + jj * (BTW_Y_DIST + RECT_H)

                    # Get top left y value and bottom right x value of rectangle
                    opp_x = (RECT_X + RECT_W) + ii * (BTW_X_DIST + RECT_W)
                    opp_y = (RECT_Y + RECT_H) + jj * (BTW_Y_DIST + RECT_H)

                    idx = ((coord_x <= rect_x2) | (coord_x >= opp_x)) | \
                          ((coord_y <= rect_y2) | (coord_y >= opp_y))
                    n = np.count_nonzero(idx)
                    coord = coord[idx].reshape(n, 1, 3)
                    if COORD2_CREATION:
                        coord2 = np.hstack((coord_x[idx].reshape(n, 1),
                                            coord_y[idx].reshape(n, 1)))
                elif COORD2_CREATION:   # and not CUT_TYPE is assumed
                    coord2 = np.hstack((coord_x, coord_y))
                    return coord, coord2, x_atoms, y_atoms

        # Save as a MATLAB file for easy viewing
        # sio.savemat('coord.mat', {'coord': coord}, oned_as='column')

        # Save xyz coordinates to graphenecoordinates.txt
        np.savetxt('graphenecoordinates.txt', coord2, delimiter='\t', fmt='%f')
        if COORD2_CREATION:
            return coord, coord2, x_atoms, y_atoms

    return coord, x_atoms, y_atoms


def plot_graphene(coord2):
    """
    Plot the graphene sheet
    """
    # Plot xy coordinates
    plt.figure(1)
    font = {'size': 22}
    plt.rc('font', **font)
    # Use plot() = line graph, scatter() = point graph
    plt.scatter(coord2[:, 0], coord2[:, 1], marker='o')
    plt.grid(True)
    plt.xlabel('Length (Angstroms)', fontsize=24)
    plt.ylabel('Width (Angstroms)', fontsize=24)
    plt.title('Graphene Antidot Lattice', fontsize=40)
    plt.draw()


def dos_calculator(coord, x_atoms, y_atoms):
    length = np.amax(coord[:, 0, 0]) * X_DIST + Z_DIST
    width = np.amax(coord[:, 0, 1]) * Y_DIST
    # Generate Hamiltonian
    (H, atoms) = v_hamiltonian(coord, x_atoms, y_atoms)
    print(H)
    del coord
    np.save('H', H)
    del H

    if BINNING:
        # Number of bins
        Nb = 40
    else:
        Nb = (6 * T) / atoms

    # Min and Max Energy Values
    E_min = -3 * T
    E_max = 3 * T

    if BINNING:
        # Energy Levels to bin density of states at
        E = np.linspace(E_min, E_max, Nb)
    else:
        # Energy Levels to calculate density of states at
        E = np.arange(E_min, E_max + Nb, Nb)

    if PERIODIC_BOUNDARY:
        eigenvalues = np.array([])
        marker = 0

        # k vectors
        k_min = 0
        k_max = (2 * np.pi) / A
        k_points = 50
        x = np.linspace(k_min, k_max, k_points)
        y = np.linspace(k_min, k_max, k_points)
        kx, ky = np.meshgrid(x, y)

        # length and width of the graphene sheet
        dists = np.array([length, width])

        # Progress Bar
        widgets = [Percentage(), ' ', Bar('>'), ' K vectors done: ', Counter(),
                   ' ', ETA(), ' ', Timer()]
        pbar = ProgressBar(widgets=widgets, maxval=(k_points ** 2)).start()

        for k in np.array(zip(kx.ravel(), ky.ravel())):
            r = k * dists
            left_phase_factor = T * np.e ** (-1j * np.dot(k, r))
            right_phase_factor = T * np.e ** (1j * np.dot(k, r))
            H = np.load('H.npy')
            H = H.astype(np.complex, copy=False)
            iu = np.triu_indices(H.shape[0])
            H[iu] = np.dot(H[iu], right_phase_factor)
            del iu
            il = np.tril_indices(H.shape[0])
            H[il] = np.dot(H[il], left_phase_factor)
            del il
            eigenvalues = np.append(eigenvalues, linalg.eigvals(H))
            # if not marker:
            #     N = dos_eig(H, E, E_min, E_max, Nb, atoms)
            #     marker += 1
            # else:
            #     N += dos_eig(H, E, E_min, E_max, Nb, atoms)
            #     marker += 1
            marker += 1
            pbar.update(marker)
        pbar.finish()
        N = dos_eig(E, E_min, E_max, Nb, eigenvalues)
    else:
        # Calculate Density of States
        N = dos_eig(H, E, E_min, E_max, Nb, atoms)

    sio.savemat('Hphase.mat', {'H': H}, oned_as='column')
    print(k)
    print(r)
    print(left_phase_factor, right_phase_factor)
    print(left_phase_factor * T, right_phase_factor * T)
    N = N / atoms
    # data = np.column_stack((E, N))
    # np.savetxt('pythonEigenvalues.txt', eigenvalues, delimiter='\t', fmt='%f')
    # np.savetxt('pythonDoSData-Eig.txt', data, delimiter='\t', fmt='%f')
    # np.savetxt('pythonDoSBroad.txt', N, delimiter='\t', fmt='%f')
    np.save('Ndata', N)

    # dos()
    plt.figure(2)
    font = {'size': 22}
    plt.rc('font', **font)
    plt.plot(E, N)
    plt.grid(True)
    plt.xlabel('Energy (eV)', fontsize=24)
    plt.ylabel('Density of States', fontsize=24)
    plt.title('Density of States vs Energy',
              horizontalalignment='center', fontsize=40)
    # plt.figtext(0, .01, 'Eigenvalues: %s, Data Points: %s, '
    #                     'Gamma: %s\n Size: %s x %s angstroms'
    #                     % (Ne, E.shape[0] - 1, gamma, WIDTH, HEIGHT))
    plt.draw()

    print("DoS calculation complete")


def v_hamiltonian(coord, x_atoms, y_atoms):
    """
    Generates the Hamiltonian of the sheet
    Uses t = -2.7 eV as the interaction (hopping) parameter
    Only does nearest-neighbor calculations
    """

    print("Start Ham calc")
    max_size = 16000     # Keep arrays at max size of around 2 GB.
                         # Also limits lattice to ~1969 nm.
    diff = y_atoms if BUILD_HOR else x_atoms
    num = coord.shape[0]    # Number of atoms in the lattice
    print("num = " + str(num))

    if num / y_atoms <= max_size:
        chunks = 1
    else:
        chunks = np.floor(num / max_size + 1)

    if ON_SITE:
        # To try to translate delta function at 0,
        # add on-site energies to not fully bonded atoms (edge atoms)
        left_edge = np.nonzero(coord[:, 0, 0] == 0)[0]
        left_edge = left_edge[::2]  # All border atoms on left edge
        right_edge = np.nonzero(coord[:, 0, 0] == np.amax(coord[:, 0, 0]))[0]
        right_edge = right_edge[1::2]   # All border atoms on right edge

    for i in xrange(int(chunks)):
        bound1 = i * max_size - i * diff
        bound2 = (i + 1) * max_size - i * diff
        idx = ((np.abs(coord[bound1:bound2, 0, 0] -
                       coord[bound1:bound2, 0, 0, None]) <= 2) &
               (np.abs(coord[bound1:bound2, 0, 1] -
                       coord[bound1:bound2, 0, 1, None]) <= 1))
        rows, cols = np.nonzero(idx)
        x_arr = ((coord[rows, 0, 0] - coord[cols, 0, 0]) * X_DIST +
                 (coord[rows, 0, 2] - coord[cols, 0, 2]) * Z_DIST)
        y_arr = (coord[rows, 0, 1] - coord[cols, 0, 1]) * Y_DIST
        r2 = x_arr * x_arr + y_arr * y_arr

        idx = ((A - 0.5) ** 2 <= r2) & (r2 <= (A + 0.5) ** 2)

        rows, cols = rows[idx], cols[idx]

        try:
            row_data = np.hstack((row_data, rows + bound1))
            col_data = np.hstack((col_data, cols + bound1))
        except NameError:
            row_data = rows + bound1
            col_data = cols + bound1
    del idx, rows, cols, r2

    if PERIODIC_BOUNDARY:
        # Horizontal border
        x_max = np.amax(coord[:, 0, 0])
        coord2 = coord.copy()
        coord2[:-y_atoms, 0, 0] += x_max + 1
        idx = ((np.abs(coord2[:, 0, 0] - coord2[:, 0, 0, None]) <= 2) &
               (np.abs(coord2[:, 0, 1] - coord2[:, 0, 1, None]) <= 1))
        rows, cols = np.nonzero(idx)
        x_arr = ((coord2[rows, 0, 0] - coord2[cols, 0, 0]) * X_DIST +
                 (coord2[rows, 0, 2] - coord2[cols, 0, 2]) * Z_DIST)
        y_arr = (coord2[rows, 0, 1] - coord2[cols, 0, 1]) * Y_DIST
        r2 = x_arr * x_arr + y_arr * y_arr

        idx = ((A - 0.5) ** 2 <= r2) & (r2 <= (A + 0.5) ** 2)

        rows, cols = rows[idx], cols[idx]

        row_data = np.hstack((row_data, rows))
        col_data = np.hstack((col_data, cols))

        # Vertical border
        y_max = np.amax(coord[:, 0, 1])
        coord2 = coord.copy()
        coord2[:, 0, 1][np.where(coord[:, 0, 1] != y_max)] += y_max + 1
        idx = ((np.abs(coord2[:, 0, 0] - coord2[:, 0, 0, None]) <= 2) &
               (np.abs(coord2[:, 0, 1] - coord2[:, 0, 1, None]) <= 1))
        rows, cols = np.nonzero(idx)
        x_arr = ((coord2[rows, 0, 0] - coord2[cols, 0, 0]) * X_DIST +
                 (coord2[rows, 0, 2] - coord2[cols, 0, 2]) * Z_DIST)
        y_arr = (coord2[rows, 0, 1] - coord2[cols, 0, 1]) * Y_DIST
        r2 = x_arr * x_arr + y_arr * y_arr

        idx = ((A - 0.5) ** 2 <= r2) & (r2 <= (A + 0.5) ** 2)

        rows, cols = rows[idx], cols[idx]

        row_data = np.hstack((row_data, rows))
        col_data = np.hstack((col_data, cols))

    if ON_SITE:
        row_data = np.concatenate((row_data, left_edge, right_edge))
        col_data = np.concatenate((col_data, left_edge, right_edge))
        coords = np.hstack((row_data[:, None], col_data[:, None]))
        coords = np.array(list(set(tuple(c) for c in coords)))
        data = np.concatenate((np.repeat(-2.7, coords.shape[0]),
                               np.repeat(-2.7, left_edge.shape[0]),
                               np.repeat(2.7, right_edge.shape[0])))
    else:
        coords = np.hstack((row_data[:, None], col_data[:, None]))
        coords = np.array(list(set(tuple(c) for c in coords)))
        data = np.repeat(-T, coords.shape[0])
        # When simulating periodic boundary conditions,
        # corner atoms become double bonded to each other
        idx = [i for i in xrange(rows.size)
               if ((rows[i] == 0 and cols[i] == coord.shape[0] - 1) or
                   (cols[i] == 0 and rows[i] == coord.shape[0] - 1))]
        data[idx] *= 2
    H = sparse.coo_matrix((data, (coords[:, 0], coords[:, 1])),
                          shape=(num, num)).tocsc()

    H = H.todense()

    # if ON_SITE:
    count = 0
    for i in xrange(num):
        if np.nonzero(H[i, :])[0].shape[1] < 3:
            if np.nonzero(H[:, i])[0].shape[1] < 3:
                count += 1
                print(i)

    # Save as a MATLAB file for easy viewing
    sio.savemat('H.mat', {'H': H}, oned_as='column')

    # Help save memory
    garcol.collect()

    print("End Ham calc")

    return H, num


def dos():
    """
    Calculate the theoretical density of states of graphene
    using the "binning" method
    """

    Nb = 101                # Number of bins
    Nkx = 186               # Number of k vectors on x
    Nky = 164               # Number of k vectors on y

    DoS = np.zeros(Nb)      # Initialize DoS

    # Min and Max Energy Values
    E_min = -10
    E_max = 10

    a = 2.461

    # k vectors
    kx = np.linspace((-4 * np.pi) / (2 * a * np.sqrt(3)),
                     (4 * np.pi) / (2 * a * np.sqrt(3)), num=Nkx)
    ky = np.linspace((-4 * np.pi * np.sqrt(3)) / (2 * a * np.sqrt(3)),
                     (4 * np.pi * np.sqrt(3)) / (2 * a * np.sqrt(3)), num=Nky)
    if BINNING:
        E = np.linspace(E_min, E_max, num=Nb)

        # Energy increment
        inc = (E_max - E_min) / Nb
    # else:
    #     Nb = (6 * T) / atoms
    for i in kx:
        for j in ky:
            # Energy Dispersion - Calculate positive and negative
            e = T * np.sqrt(1 + 4 * np.cos((np.sqrt(3) * i * a) / 2) *
                            np.cos((j * a) / 2) + 4 *
                            (np.cos((j * a) / 2)) ** 2)
            e2 = -e
            if BINNING:
                # Find bins
                b = np.floor((e - E_min) / inc)
                b2 = np.floor((e2 - E_min) / inc)

                # Tally bins
                try:
                    DoS[b] += 1
                except IndexError:
                    pass
                try:
                    DoS[b2] += 1
                except IndexError:
                    pass

    # data = np.column_stack((E, DoS))
    # np.savetxt('DataTxt/pythonDoSData-Theo.txt', data,
               # delimiter='\t', fmt='%f')

    plt.figure(2)
    font = {'family': 'normal',
            'size': 22}
    plt.rc('font', **font)
    plt.plot(E, DoS)
    plt.grid(True)
    plt.xlabel('Energy (eV)', fontsize=24)
    plt.ylabel('Density of States', fontsize=24)
    plt.title('Density of States vs Energy', fontsize=40)
    plt.show()


def dos_eig(E, E_min, E_max, Nb, eigenvalues):
    """
    Calculate the density of states across the graphene sheet
    by solving Schrodinger's equation
    """

    Ne = np.size(eigenvalues)

    N = np.empty(E.shape[0])                     # Initialize N

    # Disregard imaginary part - eigenvalues are 'a + 0i' form
    # eigenvalues = np.real(eigenvalues)

    if BINNING:
        # Binning Method
        # Energy increment
        inc = (E_max - E_min) / Nb
        for e in eigenvalues:
            e2 = -e

            # Find bins
            b = np.floor((e - E_min) / inc)
            b2 = np.floor((e2 - E_min) / inc)

            # Tally bins (-1 because Python indexing starts at 0)
            try:
                N[b] += 1
            except IndexError:
                pass
            try:
                N[b2] += 1
            except IndexError:
                pass

    else:
        # Broadening Method
        gamma = 3 * Nb
        for e in xrange(E.shape[0]):
            # Lorentzian (broadening) function -
            # gamma changes the broadening amount
            # Used to broaden the delta functions
            # Also normalizes the density of states
            N[e] = np.sum((1 / np.pi) * (gamma / (((E[e] - eigenvalues) ** 2)
                                                  + (gamma ** 2))))

    return N


def main():
    # Generate Coordinates
    (coord, x_atoms, y_atoms) = coord_generator()

    # Calculate Hamiltonian and Density of States
    dos_calculator(coord, x_atoms, y_atoms)

    plt.show()


if __name__ == '__main__':
    main()
