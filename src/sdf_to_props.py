# sdf_to_props.py

__doc__ = """Calculates a set of chemical descriptors related to the overall
shape of molecules.
"""

###############################################################################
import numpy as np
import pybel
import openbabel
import optparse
import json
import sys
from user_predict import get_predicted_properties
###############################################################################


def main():
    usage = """
    usage: %prog [options]
    """

    parser = optparse.OptionParser(usage)
    parser.add_option("-s", "--smiles", dest="sdf_file",
                      default=None, help="SDF file of conformer ensemble")
    parser.add_option("-c", "--conf", dest="conf", default=False,
                      help="Option for calculating conformers [default: %default]")

    (options, args) = parser.parse_args()

    if options.sdf_file is None:
        parser.error("Please specify a SDF file")

    confs = list(pybel.readfile("sdf", options.sdf_file))
    globs = np.empty(len(confs))
    pbfs = np.empty(len(confs))
    for i in range(len(confs)):
        # calculate properties
        globs[i] = calc_glob(confs[i])
        pbfs[i] = calc_pbf(confs[i])
        

    data = {}
    data['molecule'] = {
        'form' : confs[0].formula,
        'molwt' : confs[0].molwt,
        'formal_chrg' : rotatable_bonds(confs[0]),
        'ppsa' : np.mean(globs),
        'hbdsa' : np.mean(pbfs)
    }
    json.dump(data, sys.stdout)


def calc_glob(mol):
    """
    Calculates the globularity (glob) of a molecule

    glob varies from 0 to 1 with completely flat molecules like benzene having a
    glob of 0 and spherical molecules like adamantane having a glob of 1

    Arguments:
        mol: pybel molecule object
    Returns:
        glob: globularity of molecule
    """
    points = get_atom_coords(mol, heavy_only = False)
    if points is None:
        return 0
    points = points.T

    # calculate covariance matrix
    cov_mat = np.cov([points[0,:],points[1,:],points[2,:]])

    # calculate eigenvalues of covariance matrix and sort
    vals, vecs = np.linalg.eig(cov_mat)
    vals = np.sort(vals)[::-1]

    # glob is ratio of last eigenvalue and first eigenvalue
    if vals[0] != 0:
        return vals[-1]/vals[0]
    else:
        return 0


def calc_pbf(mol):
    """
    Uses SVD to fit atoms in molecule to a plane then calculates the average
    distance to that plane.

    Args:
        mol: pybel molecule object
    Returns:
        pbf: average distance of all atoms to the best fit plane
    """
    points = get_atom_coords(mol)
    c, n = svd_fit(points)
    pbf = 12
    pbf = calc_avg_dist(points, c, n)
    return pbf


def rotatable_bonds(mol):
    """
    Calculates the number of rotatable bonds in a molecules. Rotors are defined
    as any non-terminal bond between heavy atoms, excluding amides

    Arg:
        mol: pybel molecule object
    Returns:
        rb: number of rotatable bonds
    """
    rb = 0
    for bond in openbabel.OBMolBondIter(mol.OBMol):
        if bond.IsRotor() and not bond.IsAmide():
            rb += 1
    return rb


def calc_avg_dist(points, C, N):
    """
    Calculates the average difference a given set of points is from a plane

    Args:
        points: numpy array of points
        C: centroid vector of plane
        N: normal vector of plane
    Returns:
        Average distance of each atom from the best-fit plane
    """
    sum = 0
    for xyz in points:
        sum += abs(distance(xyz, C, N))
    return sum / len(points)

def get_atom_coords(mol, heavy_only = False):
    """
    Retrieve the 3D coordinates of all atoms in a molecules

    Args:
        mol: pybel molecule object
    Returns:
        points: numpy array of coordinates
    """

    num_atoms = len(mol.atoms)
    pts = np.empty(shape = (num_atoms,3))

    for a in range(num_atoms):
        pts[a] = mol.atoms[a].coords

    return pts

def svd_fit(X):
    """
        Fitting algorithmn was obtained from https://gist.github.com/lambdalisue/7201028
    Find (n - 1) dimensional standard (e.g. line in 2 dimension, plane in 3
    dimension, hyperplane in n dimension) via solving Singular Value
    Decomposition.
    The idea was explained in the following references
    - http://www.caves.org/section/commelect/DUSI/openmag/pdf/SphereFitting.pdf
    - http://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
    - http://www.ime.unicamp.br/~marianar/MI602/material%20extra/svd-regression-analysis.pdf
    Example:
        >>> XY = [[0, 1], [3, 3]]
        >>> XY = np.array(XY)
        >>> C, N = svd_fit(XY)
        >>> C
        array([ 1.5,  2. ])
        >>> N
        array([-0.5547002 ,  0.83205029])
    Args:
        X: n x m dimensional matrix which n indicate the number of the dimension
            and m indicate the number of points
    Returns:
        [C, N] where C is a centroid vector and N is a normal vector
    """
    # Find the average of points (centroid) along the columns
    C = np.average(X, axis=0)
    # Create CX vector (centroid to point) matrix
    CX = X - C
    # Singular value decomposition
    U, S, V = np.linalg.svd(CX)
    # The last row of V matrix indicate the eigenvectors of
    # smallest eigenvalues (singular values).
    N = V[-1]
    return C, N


def distance(x, C, N):
    """
    Calculate an orthogonal distance between the points and the standard
    Args:
        x: n x m dimensional matrix
        C: n dimensional vector whicn indicate the centroid of the standard
        N: n dimensional vector which indicate the normal vector of the standard
    Returns:
        m dimensional vector which indicate the orthogonal distance. the value
        will be negative if the points beside opposite side of the normal vector
    """
    return np.dot(x - C, N)

if __name__=='__main__':
	main()
