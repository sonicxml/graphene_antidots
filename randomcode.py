# Random Imports
# import string # May or may not be here for a reason
# import sys    # General python import
# import timeit # Used for timing functions - see random_code()
# import scipy.sparse.linalg as spla  # Used for sparse matrix calculations
# from pysparse.sparse import spmatrix
# from pysparse.itsolvers import krylov
# from pysparse.eigen import jdsym
# from numba.decorators import autojit


def random_code_snippets():
    # Just random code that I want to keep that doesn't belong anywhere else

    # Coordinate plotting in MATLAB:
    #   plot(graphenecoordinates(:,1),graphenecoordinates(:,2),'o')'

    # Don't truncate printed arrays
    #   set_printoptions(threshold=nan)

    # DeMorgan's Law on a line in generator_loop()
    #   if not (not (cut_type == 1) or not (rect_y <= h_units <= opp_y) or not (rect_x <= w_units <= opp_x)):

    # Load coordinate file as array
    #   gc = loadtxt("graphenecoordinates.txt")
    #   gc.view('i8,i8,i8').sort(order=['f1'], axis=0) # For 64-bit systems - for 32-bit, change 'i8' to 'i4'

    # Hamiltonian Speed Testing
    #   gc = loadtxt("graphenecoordinates.txt")
    #   t1 = timeit.Timer(lambda: hamiltonian(gc))
    #   t2 = timeit.Timer(lambda: v_hamiltonian(coord))
    #   print t1.timeit(number=1)
    #   print t2.timeit(number=1)

    # For speed testing, in IPython type (after commenting out show plot of Transmission and graphene)
    #   %timeit -n 20 %run graphenex-xx.py
    #   Where 20 is the number of times you want it to loop / 3

    # Translator Remove Zeroes
    #   gc2 = filter(lambda a: a != 0, gc2) - Was used to remove zeroes, but adding (x_limit+1.24) solves that problem

    pass


def sign(x):
    # Returns 1 if x > 0, 0 if x == 0, -1 if x < 0
    """

    :param x:
    :return:
    """
    return x > 0 - x < 0


def vert_convert(x):
    # Convert a 1xn array to nx1
    """

    :param x:
    :return:
    """

    # COULD JUST USE A RESHAPE HERE
    x = np.atleast_2d(x)
    x = np.column_stack(x)     # column_stack((x)) may or may not have redundant parentheses
    return x


def faster_inverse(A):
    """
    stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
    nullege.com/codes/show/src%40n%40u%40numpy-refactor-HEAD%40numpy%40linalg%40linalg.py/18/numpy.core.zeros/python
    www.netlib.org/lapack/double/dgesv.f
    www.netlib.org/lapack/complex16/zgesv.f

    Even faster inverse
    numpy/scipy's linalg.inv(A) essentially does linalg.solve(A, identity(A.shape[0])
    Looking into linalg.solve(), one can see that there are many safeguards to ensure the correct input
    Removing those safeguards greatly speeds up the code

    :param A:
    :return: :raise:
    """
    global Ic
    # Slow faster version of faster_inverse

    # (Slow) Fast version
    #   lapack_routine = lapack_lite.zgesv
    #   print A.shape
    #   b = eye(A.shape[0], dtype=A.dtype)
    #   n_eq = A.shape[1]
    #   n_rhs = A.shape[0]
    #   pivots = zeros(n_eq, intc)
    #   identity = eye(n_eq)
    #   b = copy(identity)
    #   results = linalg.lapack_lite.zgesv(n_eq, n_rhs, A, n_eq, pivots, b, n_eq, 0)
    #   if results['info'] > 0:
    #       raise LinAlgError('Singular matrix')

    return np.asmatrix(linalg.lapack.zgesv(A, np.copy(Ic))[2])


def translator(gc, num_x_trans, num_y_trans):
    # WIP
    # Cannot yet translate in both x and y directions

    # Translate an array gc num_x_trans times horizontally and num_y_trans vertically

    global x_limit, y_limit

    gc1 = np.copy(gc[:, 0])
    gc2 = np.copy(gc[:, 1])
    gc3 = np.copy(gc[:, 2])

    for counter in xrange(4):
        for i in xrange(num_x_trans if (num_x_trans > num_y_trans) else num_y_trans):
            gc1trans = np.copy(gc1)
            if i < num_x_trans and (counter == 2 or counter == 3):
                gc1trans = [x + ((x_limit + 1.24) * (i + 1)) for x in gc1trans]
            gc1trans = vert_convert(gc1trans)

            gc2trans = np.copy(gc2)
            if i < num_y_trans and (counter == 1 or counter == 3):
                gc2trans = [y + ((y_limit + 1.24) * (i + 1)) for y in gc2trans]
            gc2trans = vert_convert(gc2trans)

            gc3trans = np.copy(gc3)
            gc3trans = vert_convert(gc3trans)

            gc1trans = np.column_stack((gc1trans, gc2trans))
            gc1trans = np.column_stack((gc1trans, gc3trans))

            gc = np.concatenate((gc, gc1trans), axis=0)

    return gc


def grmagnus(alpha, beta, betad, kp):
    """
    From J. Phys. F Vol 14, 1984, 1205, M P Lopez Sancho, J. Rubio
    20-50 % faster than gravik
    From Huckel IV Simulator
    """
    tmp = linalg.inv(alpha)                    # Inverse part of Eq. 8
    t = (-1 * tmp) * betad              # Eq. 8 (t0)
    tt = (-1 * tmp) * beta              # Eq. 8 (t0 tilde)
    T = t.copy()                        # First term in Eq. 16
    Id = np.eye(alpha.shape[0])            # Save the identity matrix
    Toldt = Id.copy()                   # Product of tilde t in subsequent terms in Eq. 16
    change = 1                          # Convergence measure
    counter = 0                         # Just to make sure no infinite loop

    etag = 0.000001
    etan = 0.0000000000001
    while linalg.norm(change) > etag and counter < 100:
        counter += 1
        Toldt = linalg.dot(Toldt, tt)    # Product of tilde t in subsequent terms in Eq. 16
        if (1 / (np.linalg.cond(Id - linalg.dot(t, tt) - linalg.dot(tt, t)))) < etan:
            g = 0
            print "1: tmp NaN or Inf occurred, return forced. Kp: " + str(kp)
            return g

        tmp = linalg.inv(Id - t.dot(tt) - tt.dot(t))    # Inverse part of Eq. 12

        t = tmp.dot(t).dot(t)       # Eq. 12 (t_i)
        tt = tmp.dot(tt).dot(tt)    # Eq. 12 (t_i tilde)
        change = Toldt.dot(t)       # Next term of Eq. 16
        T = T + change              # Add it to T, Eq. 16

        if np.isnan(change).sum() or np.isinf(change).sum():
            g = 0
            print "2: tmp NaN or Inf occurred, return forced. Kp: " + str(kp)
            return g

    if (1 / (np.linalg.cond(alpha + beta.dot(T)))) < etan:
        g = 0
        print "3: tmp NaN or Inf occurred, return forced. Kp: " + str(kp)
        return g

    g = linalg.inv(alpha + beta.dot(T))

    gn = abs(g - linalg.inv(alpha - beta.dot(g).dot(betad)))

    if gn.max() > 0.001 or counter > 99:
        g = 0
        print "4: Attention! not correct sgf. Kp: " + str(kp)
        return g

    # Help save memory
    del tmp, t, tt, T, Id, Toldt, change, counter, etag, etan, gn

    return g


def generator_loop():
    """
    DEPRECATED
    """

    global rect_x, rect_y, rect_h, rect_w
    pointy = 1
    holey = -1
    coord = linalg.array([])
    checker = 0
    h_units = 0

    if cut_type == 1:
        # Convert to Angstroms
        if nanometers:
            rect_x *= 10
            rect_y *= 10
            rect_h *= 10
            rect_w *= 10

        # Get upper left Y value and bottom right x value of rectangle
        opp_x = rect_x + rect_w
        opp_y = rect_y + rect_h

    while h_units <= y_limit:
        if pointy > 0:
            w_units = w_leg
        else:
            w_units = 0

        while w_units <= x_limit:
            if cut_type == 1 and rect_y <= h_units <= opp_y and rect_x <= w_units <= opp_x:
                cut = True
            else:
                cut = False

            if not cut:
                try:
                    coord = np.vstack((coord, [w_units, h_units, 0]))
                except NameError:
                    coord = [w_units, h_units, 0]

            if (checker == 0) and (pointy == 1):
                holey = -1
                checker = 1
            elif (checker == 0) and (pointy == -1):
                holey = 1
                checker = 1
            else:
                holey *= -1

            if holey:
                w_increment = w_leg * 4
            else:
                w_increment = w_leg * 2

            # In C code, had if (armchair) and else {w_increment = dw_leg} - haven't added that in yet
            w_units += w_increment

        h_increment = h_leg
        # In C code, had if (armchair) and else ifs - haven't added that yet

        if not h_units:
            pointy = -1
        else:
            pointy += sign(pointy)
            if abs(pointy) > 1:
                pointy = -sign(pointy)

        checker = 0

        h_units += h_increment

    np.savetxt('graphenecoordinates.txt', coord, delimiter='\t', fmt='%f')

    return coord


# noinspection PyArgumentList
def hamiltonian(coord):
    """
    DEPRECATED
    """
    Norb = coord.shape[0]
    numH = Norb

    x = np.copy(coord[:, 0])
    x = vert_convert(x)

    y = np.copy(coord[:, 1])
    y = vert_convert(y)

    z = np.copy(coord[:, 2])
    z = vert_convert(z)

    x = np.tile(x, Norb)
    y = np.tile(y, Norb)
    z = np.tile(z, Norb)

    x = np.square(x.T - x)
    y = np.square(y.T - y)
    z = np.square(z.T - z)
    Q = x + y + z

    # Help save memory
    del x, y, z

    R = np.sqrt(Q)

    # Help save memory
    del Q
    garcol.collect()

    H = np.zeros(shape=(numH, numH))
    for k in xrange(0, numH):
        for l in xrange(0, numH):
            if 1.38 <= R[k, l] <= 1.45:
                H[k, l] = -2.7
            else:
                H[k, l] = 0

    np.savetxt('pythonHam.txt', H, delimiter='\t', fmt='%f')
    atoms = H.shape[0]
    garcol.collect()
    return H, atoms


def dos_eigs_snippets():
    # H = sparse.dia_matrix(H)             # Convert H to a matrix - would it save mem to have non sparse H only?

    # Check if Hamiltonian is a Hermitian matrix
    #  print "Is H Hermitian? %s" % str(np.array_equal(H.todense(), H.todense().H))

    # s = spla.svds(H, )[1]
    # rank = np.sum(s > 1e-12)

    # Scipy Sparse Linalg
    # vals = spla.eigsh(H, 500, which='BE', return_eigenvectors=False)

    # PySparse
    # vals = jdsym.jdsym(H, None, None, atoms, 1.2, 1e-12, atoms, krylov.qmrs)[1]

    # Binning Method
    # Energy increment
    # inc = (E_max - E_min) / Nb
    # for e in vals:
    #     e2 = -e
    #
    #     # Find bins
    #     b = np.floor((e - E_min) / inc)
    #     b2 = np.floor((e2 - E_min) / inc)
    #
    #     # Tally bins (-1 because Python indexing starts at 0)
    #     try:
    #         N[b] += 1
    #     except IndexError:
    #         pass
    #     try:
    #         N[b2] += 1
    #     except IndexError:
    #         pass
    # N = N / atoms

    # np.savetxt('DataTxt/pythonEigenvalues.txt', vals, delimiter='\t', fmt='%f')

    pass


def v_hamiltonian_snippets():
    # H = spmatrix.ll_mat(num, num)
    # H.put(-2.7, row_data, col_data)

    pass


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
    # H = np.asmatrix(H)                        # Convert H to a matrix
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
        ginit = spla.inv(alpha)
        g = ginit.copy()
        eps = 1
        it = 1
        while eps > 0.000001:
            it += 1
            S = g.copy()
            g = alpha - beta * S * betad
            try:
                g = spla.inv(g)
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
        # g1 = grmagnus_2(alpha, betad, beta, E[kp])
        # g2 = grmagnus_2(alpha, beta, betad, E[kp])

        # Use gravik
        g1 = gravik(alpha,betad,beta)
        g2 = gravik(alpha,beta,betad)

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


def dos_negf(H, atoms):
    """
    Calculate the density of states across the graphene sheet using a
    recursive Non-Equilibrium Green's Function
    """

    # H = sparse.dia_matrix(H)             # Convert H to a matrix

    eta = 1e-3

    I = sparse.eye(atoms)

    # Min and Max Energy Values
    E_min = -10
    E_max = 10

    Ne = 101                             # Number of Data Points

    # Energy Levels to calculate density of states at
    E = np.linspace(E_min, E_max, Ne)

    N = np.zeros(Ne)

    print(
        np.array_equal(H, H.H))  # Check if Hamiltonian is a Hermitian matrix

    for kp in xrange(Ne):
        EE = E[kp]
        print(str(EE))

        G = np.asmatrix(spla.inv((EE + 1j * eta) * I - H))

        N[kp] = (-1 / np.pi) * np.imag(np.trace(G))

        # Help save memory
        del G

    data = np.column_stack((E, N))
    np.savetxt('DataTxt/pythonDoSData-NEGF.txt', data, delimiter='\t',
               fmt='%f')

    plt.clf()
    plt.plot(E, N)
    plt.grid(True)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States')
    plt.title('Density of States vs Energy')
    plt.show()

