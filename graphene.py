#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
graphene.py - Atomic Coordinate Generator, Hamiltonian Generator, and Transmission Calculator for graphene lattices

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

from __future__ import division, print_function     # Use Python3 division
import gc as garcol                 # Used for garbage collection to help save memory

import matplotlib.pyplot as plt     # Matplotlib
import numpy as np                  # NumPy
import scipy.sparse as sparse       # Used for sparse matrices
from scipy import linalg            # Linear Algebra Functions
import scipy.sparse.linalg as spla
from numba.decorators import autojit
from progressbar import ProgressBar, Percentage, Bar

from parameters import *
np.use_fastnumpy = True

# import scipy.io as sio              # Used for saving arrays as MATLAB files
# from pysparse.sparse import spmatrix
# from pysparse.itsolvers import krylov
# from pysparse.eigen import jdsym


#
# Getting Started
#

# Before using this program, you must install:
#   Python 2.7.x
#   Numpy python module
#   Scipy python module
#   Matplotlib python module
#   IPython (optional)
#   IDE: PyCharm
# To run, in a python shell type %run graphene.py (for Windows machines)
# To change parameters of the graphene lattice, open the parameters.py module and change away


# To-Do list:
# TODO: Clean up code according to Python standards - STARTED
# TODO: Vectorize coordinate generator - NOT STARTED
# TODO: Possibly use Cython or Numba to optimize calculations and improve calculation speed - STARTED
# TODO: LOW PRIORITY: Define X_DIST, Y_DIST, and Z_DIST for zigzag orientation && check zigzag generation - NS
# TODO: LOW PRIORITY: Finish translator() - NOT STARTED
# TODO: Could just use floor() instead of ceil() in dos()

# Module options to possibly add: PyPy, numexpr, Theano, pytables, cython, pysparse, numba, vectorization
# http://technicaldiscovery.blogspot.com/2011/06/speeding-up-python-numpy-cython-and.html
# http://www.physics.udel.edu/~bnikolic/teaching/phys824/MATLAB/
# http://deeplearning.net/software/theano/
# http://lamp.tu-graz.ac.at/~hadley/ss1/bands/tightbinding/tightbinding.php
# http://www2.physics.ox.ac.uk/sites/default/files/BandMT_04.pdf
# http://www.physics.ucdavis.edu/Classes/Physics243A/TightBinding.Basics.Omar.pdf

def main_generator():
    """
    Corrects for Units, Calculates Parameters, and Calls the Coordinate Generator Function
    """

    x_diff, y_diff = None, None

    if DISTANCE:
        # Limits for the loops
        # x_limit
        if WIDTH > DW_LEG:
            x_diff = (WIDTH % DW_LEG)
        x_limit = WIDTH    # - x_diff

        # y_limit
        if HEIGHT > DH_LEG:
            y_diff = (HEIGHT % DH_LEG)
        y_limit = HEIGHT    # - y_diff

        x_times = round(x_limit // X_DIST)
        y_times = round(y_limit / Y_DIST)

        print("x_limit: %s" % str(x_limit))
        print("y_limit: %s" % str(y_limit))
        if x_diff:
            print("x_diff: %s" % str(x_diff))
        if y_diff:
            print("y_diff: %s" % str(y_diff))
    else:
        x_times, y_times = WIDTH, HEIGHT

    print("x_times: %s" % str(x_times))
    print("y_times: %s" % str(y_times))
    print("Beginning Coordinate Generation")
    if COORD2_CREATION:
        (coord, coord2, x_atoms, y_atoms) = v_generator(x_times, y_times)
        return coord, coord2, x_atoms, y_atoms
    else:
        (coord, x_atoms, y_atoms) = v_generator(x_times, y_times)
        return coord, x_atoms, y_atoms


def plot_graphene(coord2):
    """
    Plot the graphene sheet
    """
    # Plot xy coordinates
    plt.figure(1)
    plt.scatter(coord2[:, 0], coord2[:, 1], marker='o')     # plot() = line graph, scatter() = point graph
    plt.grid(True)
    plt.xlabel('Length (Angstroms)')
    plt.ylabel('Width (Angstroms)')
    plt.title('Graphene Lattice')
    plt.draw()


def transmission(H, atoms):
    """
    Calculate the transmission across the graphene sheet using a recursive Non-Equilibrium Green's Function
    """

    atomsh = atoms // 2
    if atoms % 2 != 0:
        atoms -= 1
    print(str(atoms))
    print(str(atomsh))

    # Make Hon and Hoff each half of H
    H = np.asmatrix(H)                        # Convert H to a matrix
    Hon = sparse.dia_matrix(H[0:atomsh, 0:atomsh])
    Hoff = sparse.dia_matrix(H[0:atomsh, atomsh:atoms])
    Hoffd = sparse.dia_matrix(Hoff.H)      # Conjugate Transpose of Hoff
    del H

    eta = 0.003
    # eta_original = 0.001

    I = np.eye(Hon.shape[0])

    Ne = 5        # Number of data points

    E = np.linspace(-2, 2, Ne)     # Energy Levels to calculate transmission at

    T = [None] * Ne     # Initialize T

    def grmagnus_2(alpha, beta, betad, kp):
        """
        grmagnus with sparse matrices
        From J. Phys. F Vol 14, 1984, 1205, M P Lopez Sancho, J. Rubio
        20-50 % faster than gravik
        From Huckel IV Simulator
        """

        tmp = linalg.inv(alpha.todense())           # Inverse part of Eq. 8
        t = (-1 * tmp) * betad.todense()            # Eq. 8 (t0)
        tt = (-1 * tmp) * beta.todense()            # Eq. 8 (t0 tilde)
        T = t.copy()                                # First term in Eq. 16
        Toldt = I.copy()                            # Product of tilde t in subsequent terms in Eq. 16
        change = 1                                  # Convergence measure
        counter = 0                                 # Just to make sure no infinite loop

        etag = 0.000001     # 1E-6
        etan = 0.0000000000001      # 1E-13
        while linalg.norm(change) > etag and counter < 100:
            counter += 1
            Toldt = Toldt * tt      # Product of tilde t in subsequent terms in Eq. 16
                                    # Don't use Toldt *= tt because
                                    # "ComplexWarning: Casting complex values to real discards the imaginary part"
                                    # - ruins results
            tmp = I - t * tt - tt * t
            if (1 / (np.linalg.cond(tmp))) < etan:
                g = 0
                print("1: tmp NaN or Inf occurred, return forced. Kp: " + str(kp))
                return g

            tmp = linalg.inv(tmp)                   # Inverse part of Eq. 12
            t = (tmp * t * t)                       # Eq. 12 (t_i)
            tt = (tmp * tt * tt)                    # Eq. 12 (t_i tilde)
            change = Toldt * t                      # Next term of Eq. 16
            T += change                             # Add it to T, Eq. 16

            if np.isnan(change).sum() or np.isinf(change).sum():
                g = 0
                print("2: tmp NaN or Inf occurred, return forced. Kp: " + str(kp))
                return g

        g = (alpha + beta * T)

        if (1 / (np.linalg.cond(g))) < etan:
            g = 0
            print("3: tmp NaN or Inf occured, return forced. Kp: " + str(kp))
            return g

        g = linalg.inv(g)
        gn = (abs(g - linalg.inv(alpha - beta * g * betad)))

        if gn.max() > 0.001 or counter > 99:
            g = 0
            print("4: Attention! not correct sgf. Kp: " + str(kp))
            return g

        # Help save memory
        del tmp, t, tt, T, Toldt, change, counter, etag, etan, gn

        return g

    def gravik(alpha, beta, betad):
        """
        From J. Phys. F Vol 14, 1984, 1205, M P Lopez Sancho, J. Rubio
        From Huckel IV Simulator
        """
        ginit = alpha.I
        g = ginit.copy()
        eps = 1
        it = 1
        while eps > 0.000001:
            it += 1
            S = g.copy()
            g = alpha - beta * S * betad
            try:
                g = g.I
            except linalg.LinAlgError:
                pass

            g = g * 0.5 + S * 0.5
            eps = (abs(g - S).sum()) / (abs(g + S).sum())
            if it > 200:
                #if eps > 0.01:
                #    debug_here()
                eps = -eps
        return g

    for kp in xrange(Ne):
        EE = E[kp]
        print(str(EE))

        alpha = sparse.coo_matrix((EE + 1j * eta) * I - Hon)
        beta = sparse.coo_matrix((EE + 1j * eta) * I - Hoff)
        betad = sparse.coo_matrix((EE + 1j * eta) * I - Hoffd)

        # Use grmagnus
        g1 = grmagnus_2(alpha, betad, beta, E[kp])
        g2 = grmagnus_2(alpha, beta, betad, E[kp])

        # Use gravik
        # g1 = gravik(alpha,betad,beta)
        # g2 = gravik(alpha,beta,betad)

        #
        # Equations Used
        #

        # Non-Equilibrium Green's Function: G = [EI - H - Sig1 - Sig2]^-1
        #   EI = 0.003i
        # Transmission: T = Trace[Gam1*G*Gam2*G.H]
        # Gam1 (Broadening Function of lead 1) = i(Sig1 - Sig1.H)
        # Gam2 (Broadening Function of lead 2) = i(Sig2 - Sig2.H)

        sig1 = betad * g1 * beta
        sig2 = beta * g2 * betad

        # Help save memory
        del alpha, beta, betad, g1, g2

        gam1 = (1j * (sig1 - sig1.H))
        gam2 = (1j * (sig2 - sig2.H))

        G = linalg.inv((EE - 1j * eta) * I - Hon - sig1 - sig2)

        # Help save memory
        del sig1, sig2

        T[kp] = np.trace(gam1 * G * gam2 * G.H).real

        # Help save memory
        del gam1, gam2, G

    data = np.column_stack((E, T))
    np.savetxt('DataTxt/pythonTransmissionData.txt', data, delimiter='\t', fmt='%f')
    plt.plot(E, T)
    plt.grid(True)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    plt.title('Transmission vs Energy')
    plt.show()


def dos():
    """
    Calculate the theoretical density of states of graphene using the "binning" method
    """

    t = 2.7

    Nb = 101                # Number of bins
    Nk = 1001               # Number of k vectors

    DoS = np.zeros(Nb)      # Initialize DoS

    # Min and Max Energy Values
    E_min = -2
    E_max = 2

    # k vectors
    kx = np.linspace((-4 * np.pi) / (2 * A * np.sqrt(3)), (4 * np.pi) / (2 * A * np.sqrt(3)), num=Nk)
    ky = np.linspace((-4 * np.pi * np.sqrt(3)) / (2 * A * np.sqrt(3)),
                     (4 * np.pi * np.sqrt(3)) / (2 * A * np.sqrt(3)), num=Nk)

    E = np.linspace(E_min, E_max, num=Nb)

    # Energy increment
    inc = (E_max - E_min) / Nb

    for i in kx:
        for j in ky:
            # Energy Dispersion - Calculate positive and negative
            e = t * np.sqrt(1 + 4 * np.cos((np.sqrt(3) * i * A) / 2) *
                    np.cos((j * A) / 2) + 4 * (np.cos((j * A) / 2)) ** 2)
            e2 = -e

            # Find bins
            b = np.ceil((e - E_min) / inc)
            b2 = np.ceil((e2 - E_min) / inc)

            # Tally bins (-1 because Python indexing starts at 0)
            try:
                DoS[b - 1] += 1
            except IndexError:
                pass
            try:
                DoS[b2 - 1] += 1
            except IndexError:
                pass
                
    data = np.column_stack((E, DoS))
    np.savetxt('DataTxt/pythonDoSData-Theo.txt', data, delimiter='\t', fmt='%f')

    plt.plot(E, DoS)
    plt.grid(True)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States')
    plt.title('Density of States vs Energy')
    plt.show()


def dos_negf(H, atoms):
    """
    Calculate the density of states across the graphene sheet using a recursive Non-Equilibrium Green's Function
    """

    # H = sparse.dia_matrix(H)             # Convert H to a matrix

    eta = 1e-3

    I = sparse.eye(atoms)

    # Min and Max Energy Values
    E_min = -10
    E_max = 10

    Ne = 101                             # Number of Data Points

    E = np.linspace(E_min, E_max, Ne)    # Energy Levels to calculate transmission at

    N = np.zeros(Ne)                     # Initialize N

    print(np.array_equal(H, H.H))         # Check if Hamiltonian is a Hermitian matrix

    for kp in xrange(Ne):
        EE = E[kp]
        print(str(EE))

        G = np.asmatrix(spla.inv((EE + 1j * eta) * I - H))

        N[kp] = (-1 / np.pi) * np.imag(np.trace(G))

        # Help save memory
        del G

    data = np.column_stack((E, N))
    np.savetxt('DataTxt/pythonDoSData-NEGF.txt', data, delimiter='\t', fmt='%f')

    plt.clf()
    plt.plot(E, N)
    plt.grid(True)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States')
    plt.title('Density of States vs Energy')
    plt.show()


def dos_eig(H, atoms):
    """
    Calculate the density of states across the graphene sheet using an eigenvalue solver
    """

    # H = sparse.dia_matrix(H)             # Convert H to a matrix - would it save mem to have non sparse H only?

    # Check if Hamiltonian is a Hermitian matrix
    #  print "Is H Hermitian? %s" % str(np.array_equal(H.todense(), H.todense().H))

    # s = spla.svds(H, )[1]
    # rank = np.sum(s > 1e-12)

    t = 2.7

    print("Calculating eigenvalues")
    # vals = spla.eigsh(H, 500, which='BE', return_eigenvectors=False)
    vals = linalg.eigvals(H.todense())
    # vals = jdsym.jdsym(H, None, None, atoms, 1.2, 1e-12, atoms, krylov.qmrs)[1]
    del H
    Ne = np.size(vals)
    print("Ne = %s" % str(Ne))
    print("Eigenvalues calculated")
    # Number of bins
    Nb = 40

    # Min and Max Energy Values
    E_min = -3 * t
    E_max = 3 * t

    E = np.linspace(E_min, E_max, Nb)    # Energy Levels to calculate transmission at

    N = np.zeros(Nb)                     # Initialize N

    vals = np.real(vals)                 # Disregard imaginary part - eigenvalues are 'a + 0i' form

    # Binning Method
    # Energy increment
    # inc = (E_max - E_min) / Nb
    # for e in vals:
    #     e2 = -e
    #
    #     # Find bins
    #     b = np.ceil((e - E_min) / inc)
    #     b2 = np.ceil((e2 - E_min) / inc)
    #
    #     # Tally bins (-1 because Python indexing starts at 0)
    #     try:
    #         N[b - 1] += 1
    #     except IndexError:
    #         pass
    #     try:
    #         N[b2 - 1] += 1
    #     except IndexError:
    #         pass
    # N = N / atoms

    # Summation Method
    gamma = 0.05
    for e in E:
        N[e] = (1 / atoms) * np.sum(gamma / (((e - vals) ** 2) + (gamma ** 2)))

    # data = np.column_stack((E, N))
    # np.savetxt('DataTxt/pythonEigenvalues.txt', vals, delimiter='\t', fmt='%f')
    np.savetxt('pythonEigenvalues.txt', vals, delimiter='\t', fmt='%f')
    # np.savetxt('DataTxt/pythonDoSData-Eig.txt', data, delimiter='\t', fmt='%f')

    plt.figure(2)
    plt.plot(E, N)
    plt.grid(True)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States')
    plt.title('Density of States vs Energy \n Eigenvalues: %s, Bins: %s' % (Ne, Nb), horizontalalignment='center')
    plt.draw()


# @autojit
def v_generator(x_times, y_times):
    """
    Creates a 3-dimensional array (n x 0 x 2 - since the index starts at 0) 'coord'
    Where n is the number of atoms in the sheet
    Defined by the unit cell consisting of two atoms, one a horizontal translation of the other
    Each atomic coordinate is defined by 3 numbers:
      A x-value (coord_x): As the unit cell is translated horizontally, the x-value increments by 1
      A y-value (coord_y): As the unit cell is translated vertically, the y-value increments by 1
      A unit cell value (coord_u): Defines which point in the unit cell the atom is
          0 = the atom on the left
          1 = the atom on the right
    To convert these numbers into xyz coordinates,
      If a = 1.42
      ((coord_x, coord_y, coord_u).((3a/2), 0, a), (coord_x, coord_y, coord_u).(0, (sqrt(3)*a)/2, 0))
      Where . represents the dot product
    """

    x_atoms, y_atoms = 0, 0
    marker = 0
    if CHAIN:
        out_inc, in_inc = 1, 1
    elif BUILD_HOR:
        out_inc, in_inc = 1, 2
    else:
        out_inc, in_inc = 2, 1

    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=x_times).start()

    #
    # Coordinate Generator
    #

    for j in xrange(0, int(x_times if BUILD_HOR else y_times), out_inc):
        j2 = j
        for i in xrange(0, int(y_times if BUILD_HOR else x_times), in_inc):
            if not BUILD_HOR:
                j, i = i, j2

            if i == 0:
                x_atoms += 1
                if not CHAIN:
                    x_atoms += 1

            if j == 0:
                y_atoms += 1

            if marker:
                if j % 2 != 0:
                    if not CHAIN:
                        coord = np.vstack((coord, [[[j, i + 1, 0]]]))
                        coord = np.vstack((coord, [[[j, i + 1, 1]]]))
                    else:
                        pass
                else:
                    coord = np.vstack((coord, [[[j, i, 0]]]))
                    if x_times % 2 != 0 and j2 == (x_times - 1) and not DISTANCE:
                        pass
                    else:
                        coord = np.vstack((coord, [[[j, i, 1]]]))
            else:
                coord = [[[j, i, 0]]]
                coord = np.vstack((coord, [[[j, i, 1]]]))
                marker = 1
        pbar.update(j2 + 1)
    pbar.finish()

    print(np.shape(coord))

    print("Number of atoms along the x-axis: %s" % str(x_atoms))
    print("Number of atoms along the y-axis: %s" % str(y_atoms))

    #
    # Antidot Generator
    #

    if CUT_TYPE == 1:
        rect_x2 = RECT_X

        # Get upper left Y value and bottom right x value of rectangle
        opp_x = rect_x2 + RECT_W
        opp_y = RECT_Y + RECT_H

    if COORD2_CREATION or CUT_TYPE:
        for ii in xrange(ANTIDOT_NUM):
            rect_x2 += ii * (BTW_DIST + RECT_W)
            opp_x += ii * (BTW_DIST + RECT_W)

            coord_x = ((coord[:, 0, 0] * X_DIST + coord[:, 0, 2] * Z_DIST).reshape(coord.shape[0], 1))
            coord_y = (coord[:, 0, 1] * Y_DIST).reshape(coord.shape[0], 1)

            if CUT_TYPE:
                idx = ((coord_x <= rect_x2) | (coord_x >= opp_x)) & ((coord_y <= RECT_Y) | (coord_y >= opp_y))
                n = np.count_nonzero(idx)
                coord = coord[idx].reshape(n, 1, 3)
                if COORD2_CREATION:
                    coord2 = np.hstack((coord_x[idx].reshape(n, 1), coord_y[idx].reshape(n, 1)))
            elif COORD2_CREATION:   # and not CUT_TYPE is assumed
                coord2 = np.hstack((coord_x, coord_y))
                return coord, coord2, x_atoms, y_atoms

        # Save as a MATLAB file for easy viewing and to compare MATLAB results with Python results
        # sio.savemat('coord.mat', {'coord': coord}, oned_as='column')

        # Save xyz coordinates to graphenecoordinates.txt
        np.savetxt('graphenecoordinates.txt', coord2, delimiter='\t', fmt='%f')
        if COORD2_CREATION:
            return coord, coord2, x_atoms, y_atoms

    return coord, x_atoms, y_atoms


def v_hamiltonian(coord, x_atoms, y_atoms):
    """
    Generates the Hamiltonian of the sheet
    Uses t = -2.7 eV as the interaction (hopping) parameter
    Only does nearest-neighbor calculations
    """

    print("Start Ham calc")
    max_size = 16000     # Keep arrays at max size of around 2 GB. Also limits lattice to ~1969 nm.
    diff = y_atoms if BUILD_HOR else x_atoms
    # if diff == 0:
    #     diff += 1
    num = coord.shape[0]    # Number of atoms in the lattice
    print("num = " + str(num))

    if num / y_atoms <= max_size:
        chunks = 1
    else:
        chunks = np.floor(num / max_size + 1)

    for i in xrange(int(chunks)):
        bound1 = i * max_size - i * diff
        bound2 = (i + 1) * max_size - i * diff
        idx = ((np.abs(coord[bound1:bound2, 0, 0] - coord[bound1:bound2, 0, 0, None]) <= 2) &
               (np.abs(coord[bound1:bound2, 0, 1] - coord[bound1:bound2, 0, 1, None]) <= 1))

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
    data = np.repeat(-2.7, row_data.shape[0])
    H = sparse.coo_matrix((data, (row_data, col_data)), shape=(num, num)).tocsc()
    # H = spmatrix.ll_mat(num, num)
    # H.put(-2.7, row_data, col_data)

    # print H
    # Save as a MATLAB file for easy viewing and to compare MATLAB results with Python results
    # sio.savemat('H.mat', {'H': H.todense()}, oned_as='column')

    # Help save memory
    garcol.collect()

    print("End Ham calc")

    return H, num


def main():
    # Check to make sure garbage collection is enabled
    garcol_boolean = garcol.isenabled()
    print(garcol_boolean)

    # Generate Coordinates
    if COORD2_CREATION:
        (coord, coord2, x_atoms, y_atoms) = main_generator()
    else:
        (coord, x_atoms, y_atoms) = main_generator()
    print("main_generator() complete")

    # Plot graphene
    if COORD2_CREATION and PLOT_OPTION:
        plot_graphene(coord2)
        print("plot_graphene() complete")
        del coord2

    # Generate Hamiltonian
    (H, atoms) = v_hamiltonian(coord, x_atoms, y_atoms)
    del coord

    # print H
    # Calculate Transmission
    # transmission(H, atoms)
    # dos()
    dos_eig(H, atoms)
    print("DoS complete")

    plt.show()


if __name__ == '__main__':
    main()