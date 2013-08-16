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
from progressbar import Bar, Counter, ETA, Percentage, ProgressBar, Timer

import parameters as p
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

# Get rid of TO-DO list warnings
# pylint: disable=W0511

# False positive from np.load()
# pylint: disable=E1103

# IT'S NOT MAGIC, IT'S A LANGUAGE FEATURE PYLINT!!
# (W0142: Used * or ** magic (star-args))
# pylint: disable=W0142

# To-Do list:
# TODO: Add periodic boundary conditions in Hamiltonian calculation - STARTED
# TODO: Clean up code according to Python standards - STARTED
# TODO: LOWER PRIORITY: Use unit vectors for coordinate generation - NOT STARTED
# TODO: LOW PRIORITY: Define X_DIST, Y_DIST, and Z_DIST
#       for zigzag orientation && check zigzag generation - NS
# TODO: LOW PRIORITY: Finish translator() - NOT STARTED
# TODO: Break up dos_calculator() and v_hamiltonian()
# TODO: Rename dos_calculator(), dos(), dos_eig(), v_hamiltonian()

# Module options to possibly add: numexpr, Theano, pytables,
#                                 cython, pysparse, numba
# technicaldiscovery.blogspot.com/2011/06/speeding-up-python-numpy-cython-and.html
# www.physics.udel.edu/~bnikolic/teaching/phys824/MATLAB/
# deeplearning.net/software/theano/
# lamp.tu-graz.ac.at/~hadley/ss1/bands/tightbinding/tightbinding.php
# www2.physics.ox.ac.uk/sites/default/files/BandMT_04.pdf
# www.physics.ucdavis.edu/Classes/Physics243A/TightBinding.Basics.Omar.pdf


def generator_wrapper():
    """
    Corrects for Units, Calculates Parameters,
    and Calls the Coordinate Generator Function
    """

    if p.DISTANCE:
        x_times = round(p.WIDTH // p.X_DIST)
        y_times = round(p.HEIGHT / p.Y_DIST)
    else:
        x_times, y_times = p.WIDTH, p.HEIGHT

    print("Parameters calculated:")
    print("x_times: %s" % str(x_times))
    print("y_times: %s" % str(y_times))
    print("Beginning coordinate generation")
    if p.XY_COORD_CREATION:
        (vector_coord,
         num_x_atoms, num_y_atoms) = vector_coord_generator(x_times, y_times)
        vector_coord, xy_coord = antidot_generator(vector_coord)
        print("Coordinate generation finished")
        if p.PLOT_OPTION:
            plot_graphene_lattice(xy_coord)
            print("Graphene plot drawn")
            del xy_coord
    else:
        (vector_coord, num_x_atoms, num_y_atoms) = \
            vector_coord_generator(x_times, y_times)
        vector_coord = antidot_generator(vector_coord)
        print("Coordinate generation finished")

    return vector_coord, num_x_atoms, num_y_atoms


def vector_coord_generator(x_times, y_times):
    """
    Creates a 3-dimensional array (n x 1 x 3) 'vector_coord'
    Where n is the number of atoms in the sheet
    Defined by the unit cell consisting of two atoms,
    one a horizontal translation of the other
    :param y_times:
    :param x_times:
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

    if p.CHAIN:
        out_inc, in_inc = 1, 1
    elif p.BUILD_HOR:
        out_inc, in_inc = 1, 2
    else:
        out_inc, in_inc = 2, 1

    #
    # Coordinate Generator
    # TODO: Make it work with odd numbers of atoms on x-axis
    # TODO: Fix BUILD_HOR, rename j, i, k
    #

    j = np.arange(0, int(x_times if p.BUILD_HOR else y_times), out_inc)
    i = np.arange(0, int(y_times if p.BUILD_HOR else x_times), in_inc)
    j_shape = j.shape[0]
    i_shape = i.shape[0]
    j = np.repeat(j, i_shape)
    i = np.tile(i, j_shape)
    j = np.reshape(j, (j.shape[0], 1))
    i = np.reshape(i, (i.shape[0], 1))
    k = np.zeros_like(i)
    vector_coord = np.dstack((j, i, k))
    del j_shape, i_shape, j, i, k

    if not p.CHAIN:
        # Create 'B' sub-lattice
        vector_coord = np.repeat(vector_coord, 2, axis=0)
        vector_coord[1::2][:, 0, 2] += 1
        vector_coord[:, 0, 1][np.where(vector_coord[:, 0, 0] % 2 != 0)] += 1

    num_x_atoms = (np.amax(vector_coord[:, 0, 0]) + 1) * \
                  (2 if not p.CHAIN else 1)
    num_y_atoms = np.amax(vector_coord[:, 0, 1]) + 1

    if p.TRIM_EDGES:
        x_max = np.amax(vector_coord[:, 0, 0])
        idx = (((vector_coord[:, 0, 0] == 0) & (vector_coord[:, 0, 2] == 0)) &
              (vector_coord[:, 0, 1] % 2 == 0))
        idx2 = (((vector_coord[:, 0, 0] == x_max) &
                 (vector_coord[:, 0, 2] == 1)) &
                (vector_coord[:, 0, 1] % 2 == (0 if x_times % 2 == 1 else 1)))
        idx = ~(idx | idx2)
        vector_coord = vector_coord[idx]

    print(np.shape(vector_coord))

    print("Number of atoms along the x-axis: %s" % str(num_x_atoms))
    print("Number of atoms along the y-axis: %s" % str(num_y_atoms))

    return vector_coord, num_x_atoms, num_y_atoms


def antidot_generator(vector_coord):
    #
    # Antidot Generator
    #

    """

    :param vector_coord:
    :return:
    """
    if p.XY_COORD_CREATION or p.CUT_TYPE:
        for x_antidot_num in xrange(p.ANTIDOT_X_NUM):
            for y_antidot_num in xrange(p.ANTIDOT_Y_NUM):
                coord_x = ((vector_coord[:, 0, 0] * p.X_DIST +
                            vector_coord[:, 0, 2] * p.Z_DIST).
                           reshape(vector_coord.shape[0], 1))
                coord_y = (vector_coord[:, 0, 1] * p.Y_DIST).\
                    reshape(vector_coord.shape[0], 1)

                if p.CUT_TYPE:
                    # Get bottom left x and y values
                    rect_x2 = p.RECT_X + x_antidot_num * \
                              (p.BTW_X_DIST + p.RECT_W)
                    rect_y2 = p.RECT_Y + y_antidot_num * \
                              (p.BTW_Y_DIST + p.RECT_H)

                    # Get top left y value and bottom right x value of rectangle
                    opp_x = (p.RECT_X + p.RECT_W) + x_antidot_num * \
                            (p.BTW_X_DIST + p.RECT_W)
                    opp_y = (p.RECT_Y + p.RECT_H) + y_antidot_num * \
                            (p.BTW_Y_DIST + p.RECT_H)

                    idx = ((coord_x <= rect_x2) | (coord_x >= opp_x)) | \
                          ((coord_y <= rect_y2) | (coord_y >= opp_y))
                    num_atoms = np.count_nonzero(idx)
                    vector_coord = vector_coord[idx].reshape(num_atoms, 1, 3)
                    if p.XY_COORD_CREATION:
                        xy_coord = np.hstack((coord_x[idx].
                                              reshape(num_atoms, 1),
                                              coord_y[idx].
                                              reshape(num_atoms, 1)))
                elif p.XY_COORD_CREATION:   # and not CUT_TYPE is assumed
                    xy_coord = np.hstack((coord_x, coord_y))
                    return vector_coord, xy_coord

        # Save as a MATLAB file for easy viewing
        # sio.savemat('vector_coord.mat', {'vector_coord': vector_coord},
        # oned_as='column')

        # Save xyz coordinates to graphenecoordinates.txt
        np.savetxt('graphenecoordinates.txt', xy_coord,
                   delimiter='\t', fmt='%f')
        if p.XY_COORD_CREATION:
            return vector_coord, xy_coord

    return vector_coord


def plot_graphene_lattice(coord2):
    """
    Plot the graphene sheet
    :param coord2:
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
    """

    :param coord:
    :param x_atoms:
    :param y_atoms:
    """
    length = np.amax(coord[:, 0, 0]) * p.X_DIST + p.Z_DIST
    width = np.amax(coord[:, 0, 1]) * p.Y_DIST
    # Generate Hamiltonian
    (hamiltonian, atoms) = v_hamiltonian(coord, x_atoms, y_atoms)
    print(hamiltonian)
    del coord
    np.save('hamiltonian', hamiltonian)
    del hamiltonian

    if p.BINNING:
        # Number of bins
        num_data_points = 40
    else:
        # Number of data points
        num_data_points = (6 * p.T) / atoms

    # Min and Max Energy Values
    min_energy = -3 * p.T
    max_energy = 3 * p.T

    if p.BINNING:
        # Energy Levels to bin density of states at
        energy_levels = np.linspace(min_energy, max_energy, num_data_points)
    else:
        # Energy Levels to calculate density of states at
        energy_levels = np.arange(min_energy,
                                  max_energy + num_data_points, num_data_points)

    if p.PERIODIC_BOUNDARY:
        eigenvalues = np.array([])
        marker = 0

        # k vectors
        k_min = 0
        k_max = (2 * np.pi) / p.A
        num_k_points = 50
        x_points = np.linspace(k_min, k_max, num_k_points)
        y_points = np.linspace(k_min, k_max, num_k_points)
        kx_points, ky_points = np.meshgrid(x_points, y_points)

        # length and width of the graphene sheet
        dists = np.array([length, width])

        # Progress Bar
        widgets = [Percentage(), ' ', Bar('>'), ' K vectors done: ', Counter(),
                   ' ', ETA(), ' ', Timer()]
        pbar = ProgressBar(widgets=widgets, maxval=(num_k_points ** 2)).start()

        k_vector = None
        for k_vector in np.array(zip(kx_points.ravel(), ky_points.ravel())):
            r_vector = k_vector * dists

            left_phase_factor = p.T * np.e ** (-1j * np.dot(k_vector, r_vector))
            right_phase_factor = p.T * np.e ** (1j * np.dot(k_vector, r_vector))

            hamiltonian = np.load('hamiltonian.npy')
            hamiltonian = hamiltonian.astype(np.complex, copy=False)

            upper_triangle_index = np.triu_indices(hamiltonian.shape[0])

            hamiltonian[upper_triangle_index] = np.dot(
                hamiltonian[upper_triangle_index], right_phase_factor)
            del upper_triangle_index

            lower_triangle_index = np.tril_indices(hamiltonian.shape[0])
            hamiltonian[lower_triangle_index] = np.dot(
                hamiltonian[lower_triangle_index], left_phase_factor)
            del lower_triangle_index

            eigenvalues = np.append(eigenvalues, linalg.eigvals(hamiltonian))
            # if not marker:
            #     density_of_states = dos_eig(hamiltonian, energy_levels,
            # min_energy, max_energy, num_data_points)
            #     marker += 1
            # else:
            #     density_of_states += dos_eig(hamiltonian, energy_levels,
            # min_energy, max_energy, num_data_points)
            #     marker += 1
            marker += 1
            pbar.update(marker)
        pbar.finish()
        density_of_states = dos_eig(energy_levels, min_energy, max_energy,
                    num_data_points, eigenvalues)
    else:
        # Calculate Density of States
        eigenvalues = linalg.eigvals(hamiltonian)
        density_of_states = dos_eig(energy_levels, min_energy,
                    max_energy, num_data_points, eigenvalues)

    sio.savemat('Hphase.mat', {'hamiltonian': hamiltonian}, oned_as='column')
    print(k_vector)
    print(r_vector)
    print(left_phase_factor, right_phase_factor)
    print(left_phase_factor * p.T, right_phase_factor * p.T)
    density_of_states /= (atoms * num_k_points)
    # data = np.column_stack((energy_levels, density_of_states))
    # np.savetxt('pythonEigenvalues.txt', eigenvalues, delimiter='\t', fmt='%f')
    # np.savetxt('pythonDoSData-Eig.txt', data, delimiter='\t', fmt='%f')
    # np.savetxt('pythonDoSBroad.txt', density_of_states, delimiter='\t',
    # fmt='%f')
    np.save('Ndata', density_of_states)

    # dos()
    plt.figure(2)
    font = {'size': 22}
    plt.rc('font', **font)
    plt.plot(energy_levels, density_of_states)
    plt.grid(True)
    plt.xlabel('Energy (eV)', fontsize=24)
    plt.ylabel('Density of States', fontsize=24)
    plt.title('Density of States vs Energy',
              horizontalalignment='center', fontsize=40)
    # plt.figtext(0, .01, 'Eigenvalues: %s, Data Points: %s, '
    #                     'Gamma: %s\n Size: %s x_points %s angstroms'
    #                     % (Ne, energy_levels.shape[0] - 1, gamma, WIDTH,
    #                        HEIGHT))
    plt.draw()

    print("DoS calculation complete")


def v_hamiltonian(coord, x_atoms, y_atoms):
    """
    Generates the Hamiltonian of the sheet
    Uses t = -2.7 eV as the interaction (hopping) parameter
    Only does nearest-neighbor calculations
    :param y_atoms:
    :param x_atoms:
    :param coord:
    """

    print("Start Ham calc")
    max_size = 16000     # Keep idx array at max size of around 2 GB.
                         # Also limits lattice to ~1969 nm.

    # All nearest-neighbor atoms exist within diff atoms of an atom
    diff = y_atoms if p.BUILD_HOR else x_atoms
    num = coord.shape[0]    # Number of atoms in the lattice
    print("num = " + str(num))

    if num / y_atoms <= max_size:
        chunks = 1
    else:
        chunks = np.floor(num / max_size + 1)

    # To try to translate delta function at 0,
    # add on-site energies to not fully bonded atoms (edge atoms)
    left_edge = np.nonzero(coord[:, 0, 0] == 0)[0]
    left_edge = left_edge[::2]  # All border atoms on left edge
    right_edge = np.nonzero(coord[:, 0, 0] == np.amax(coord[:, 0, 0]))[0]
    right_edge = right_edge[1::2]   # All border atoms on right edge
    top_edge = np.nonzero(coord[:, 0, 1] == y_atoms - 1)[0]
    bottom_edge = np.nonzero(coord[:, 0, 1] == 0)[0]

    row_data = np.array([])
    col_data = np.array([])
    for i in xrange(int(chunks)):
        bound1 = i * max_size - i * diff
        bound2 = (i + 1) * max_size - i * diff

        # Following code snippet credit Jaime - StackOverflow
        # stackoverflow.com/questions/17792077/vectorizing-for-loops-numpy/17793222
        idx = ((np.abs(coord[bound1:bound2, 0, 0] -
                       coord[bound1:bound2, 0, 0, None]) <= 2) &
               (np.abs(coord[bound1:bound2, 0, 1] -
                       coord[bound1:bound2, 0, 1, None]) <= 1))
        rows, cols = np.nonzero(idx)
        x_arr = ((coord[rows, 0, 0] - coord[cols, 0, 0]) * p.X_DIST +
                 (coord[rows, 0, 2] - coord[cols, 0, 2]) * p.Z_DIST)
        y_arr = (coord[rows, 0, 1] - coord[cols, 0, 1]) * p.Y_DIST
        dist_squared = x_arr * x_arr + y_arr * y_arr

        idx = (((p.A - 0.5) ** 2 <= dist_squared) &
               (dist_squared <= (p.A + 0.5) ** 2))

        rows, cols = rows[idx], cols[idx]

        row_data = np.hstack((row_data, rows + bound1))
        col_data = np.hstack((col_data, cols + bound1))

    del idx, rows, cols, dist_squared

    if p.PERIODIC_BOUNDARY:
        # Horizontal border
        x_max = np.amax(coord[:, 0, 0])
        coord2 = coord.copy()
        coord2[:-y_atoms, 0, 0] += x_max + 1
        idx = ((np.abs(coord2[:, 0, 0] - coord2[:, 0, 0, None]) <= 2) &
               (np.abs(coord2[:, 0, 1] - coord2[:, 0, 1, None]) <= 1))
        rows, cols = np.nonzero(idx)
        x_arr = ((coord2[rows, 0, 0] - coord2[cols, 0, 0]) * p.X_DIST +
                 (coord2[rows, 0, 2] - coord2[cols, 0, 2]) * p.Z_DIST)
        y_arr = (coord2[rows, 0, 1] - coord2[cols, 0, 1]) * p.Y_DIST
        dist_squared = x_arr * x_arr + y_arr * y_arr

        idx = ((p.A - 0.5) ** 2 <= dist_squared) & \
              (dist_squared <= (p.A + 0.5) ** 2)

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
        x_arr = ((coord2[rows, 0, 0] - coord2[cols, 0, 0]) * p.X_DIST +
                 (coord2[rows, 0, 2] - coord2[cols, 0, 2]) * p.Z_DIST)
        y_arr = (coord2[rows, 0, 1] - coord2[cols, 0, 1]) * p.Y_DIST
        dist_squared = x_arr * x_arr + y_arr * y_arr

        idx = ((p.A - 0.5) ** 2 <= dist_squared) & \
              (dist_squared <= (p.A + 0.5) ** 2)

        rows, cols = rows[idx], cols[idx]

        row_data = np.hstack((row_data, rows))
        col_data = np.hstack((col_data, cols))

    if p.ON_SITE:
        row_data = np.concatenate((row_data, left_edge, right_edge))
        col_data = np.concatenate((col_data, left_edge, right_edge))
        sparse_coords = np.hstack((row_data[:, None], col_data[:, None]))
        sparse_coords = np.array(list(set(tuple(c) for c in sparse_coords)))
        data = np.concatenate((np.repeat(-2.7, sparse_coords.shape[0]),
                               np.repeat(-2.7, left_edge.shape[0]),
                               np.repeat(2.7, right_edge.shape[0])))
    else:
        sparse_coords = np.hstack((row_data[:, None], col_data[:, None]))
        print(sparse_coords)
        idx = [(np.any(sparse_coords[i, 0] == bottom_edge) and
               np.any(sparse_coords[i, 1] == top_edge)) or
               (np.any(sparse_coords[i, 0] == top_edge) and
               np.any(sparse_coords[i, 1] == bottom_edge))
               for i in xrange(sparse_coords.shape[0])]
        idx = np.array(idx)
        idx2 = [(np.any(sparse_coords[i, 0] == left_edge) and
                np.any(sparse_coords[i, 1] == right_edge)) or
                (np.any(sparse_coords[i, 0] == right_edge) and
                np.any(sparse_coords[i, 1] == left_edge))
                for i in xrange(sparse_coords.shape[0])]
        idx = idx | idx2
        idx = sparse_coords[idx].copy()
        idx = np.array(list(set(tuple(c) for c in idx)))
        sparse_coords = np.array(list(set(tuple(c) for c in sparse_coords)))
        sparse_coords = np.append(sparse_coords, idx, axis=0)
        data = np.repeat(-p.T, sparse_coords.shape[0])
        # When simulating periodic boundary conditions,
        # corner atoms become double bonded to each other
        # idx = [i for i in xrange(sparse_coords.shape[0])
        # if ((sparse_coords[i, 0] == 0 and
        #       sparse_coords[i, 1] == coord.shape[0] - 1) or
        #     (sparse_coords[i, 1] == 0 and
        #       sparse_coords[i, 0] == coord.shape[0] - 1))]
        # data[idx] *= 2
    hamiltonian = sparse.coo_matrix((data, (sparse_coords[:, 0],
                                            sparse_coords[:, 1])),
                                    shape=(num, num)).tocsc()

    hamiltonian = hamiltonian.todense()

    # if ON_SITE:
    for i in xrange(num):
        if np.nonzero(hamiltonian[i, :])[0].shape[1] < 3:
            if np.nonzero(hamiltonian[:, i])[0].shape[1] < 3:
                print(i)

    # Save as a MATLAB file for easy viewing
    sio.savemat('hamiltonian.mat', {'hamiltonian': hamiltonian},
                oned_as='column')

    # Help save memory
    garcol.collect()

    print("End Ham calc")

    return hamiltonian, num


def dos():
    """
    Calculate the theoretical density of states of graphene
    using the "binning" method
    """

    num_bins = 101
    num_kx_points = 186               # Number of k vectors on x
    num_ky_points = 164               # Number of k vectors on y

    density_of_states = np.zeros(num_bins)      # Initialize DoS

    # Min and Max Energy Values
    min_energy = -10
    max_energy = 10

    lattice_const = 2.461

    # k vectors
    k_x_component = np.linspace((-4 * np.pi) / (2 * lattice_const * np.sqrt(3)),
                                (4 * np.pi) / (2 * lattice_const * np.sqrt(3)),
                                num=num_kx_points)
    k_y_component = np.linspace((-4 * np.pi * np.sqrt(3)) /
                                (2 * lattice_const * np.sqrt(3)),
                                (4 * np.pi * np.sqrt(3)) /
                                (2 * lattice_const * np.sqrt(3)),
                                num=num_ky_points)
    if p.BINNING:
        energy_levels = np.linspace(min_energy, max_energy, num=num_bins)

        # Energy increment
        increment = (max_energy - min_energy) / num_bins
    # else:
    #     num_bins = (6 * T) / atoms
    for i in k_x_component:
        for j in k_y_component:
            # Energy Dispersion - Calculate positive and negative
            energy = p.T * np.sqrt(1 + 4 *
                                   np.cos((np.sqrt(3) * i *
                                          lattice_const) / 2) *
                                   np.cos((j * lattice_const) / 2) +
                                   4 * (np.cos((j * lattice_const) / 2)) ** 2)
            energy2 = -energy
            if p.BINNING:
                # Find bins
                bin1 = np.floor((energy - min_energy) / increment)
                bin2 = np.floor((energy2 - min_energy) / increment)

                # Tally bins
                try:
                    density_of_states[bin1] += 1
                except IndexError:
                    pass
                try:
                    density_of_states[bin2] += 1
                except IndexError:
                    pass

    # data = np.column_stack((energy_levels, density_of_states))
    # np.savetxt('DataTxt/pythonDoSData-Theo.txt', data,
               # delimiter='\t', fmt='%f')

    plt.figure(2)
    font = {'family': 'normal',
            'size': 22}
    plt.rc('font', **font)
    plt.plot(energy_levels, density_of_states)
    plt.grid(True)
    plt.xlabel('Energy (eV)', fontsize=24)
    plt.ylabel('Density of States', fontsize=24)
    plt.title('Density of States vs Energy', fontsize=40)
    plt.show()


def dos_eig(energy_levels, min_energy, max_energy, num_bins, eigenvalues):
    """
    Calculate the density of states across the graphene sheet
    by solving Schrödinger's equation
    :param eigenvalues:
    :param num_bins:
    :param max_energy:
    :param min_energy:
    :param energy_levels:
    """

    density_of_states = np.empty(energy_levels.shape[0])

    # Disregard imaginary part - eigenvalues are 'a + 0i' form
    # eigenvalues = np.real(eigenvalues)

    if p.BINNING:
        # Binning Method
        # Energy increment
        inc = (max_energy - min_energy) / num_bins
        for energy in eigenvalues:
            energy2 = -energy

            # Find bins
            bin1 = np.floor((energy - min_energy) / inc)
            bin2 = np.floor((energy2 - min_energy) / inc)

            # Tally bins (-1 because Python indexing starts at 0)
            try:
                density_of_states[bin1] += 1
            except IndexError:
                pass
            try:
                density_of_states[bin2] += 1
            except IndexError:
                pass

    else:
        # Broadening Method
        gamma = 3 * num_bins
        for energy in xrange(energy_levels.shape[0]):
            # Lorentzian (broadening) function -
            # gamma changes the broadening amount
            # Used to broaden the delta functions
            # Also normalizes the density of states
            density_of_states[energy] = np.sum((1 / np.pi) *
                                               (gamma / (((energy_levels[energy]
                                                           - eigenvalues) ** 2)
                                                         + (gamma ** 2))))

    return density_of_states


def main():
    # Generate Coordinates
    """

    TODO: Fill this out
    """
    (vector_coord, num_x_atoms, num_y_atoms) = generator_wrapper()

    # Calculate Hamiltonian and Density of States
    dos_calculator(vector_coord, num_x_atoms, num_y_atoms)

    plt.show()


if __name__ == '__main__':
    main()
