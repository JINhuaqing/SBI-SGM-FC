# a set of functions to facilitate the operation of the points on sphere
from scipy.optimize import minimize
from easydict import EasyDict as edict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

from sklearn.utils import check_random_state

# plot a sphere on the ax
def plot_sphere(ax, radius=1, color='b', alpha=0.2):
    """
    args: 
        ax: the axis
        radius: the radius of the sphere
        color: the color of the sphere
        alpha: the transparency of the sphere
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

# convert the spherical coordinate to cartesian coordinate
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# convert the cartesian coordinate to spherical coordinate
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi



# find the point on the sphere with minimal sum of distances to all other points
def find_minimal_sum_point(points):
    """
    args: 
        points: the points on the sphere
        r: the radius of the sphere
    """
    # Convert points to numpy array
    points = np.array(points)
    r = np.linalg.norm(points, axis=-1).mean()
    
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Normalize the centroid to have unit length
    centroid /= np.linalg.norm(centroid)
    
    # Return the normalized centroid as the point with minimal sum of distances
    return centroid * r



def fibonacci_sphere(samples=1000):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)









def generate_spins(points_lh, points_rh=None, unique=False, n_rep=100,
                    random_state=None, surface_algorithm='FreeSurfer'):
    """ Generate rotational spins based on points that lie on a sphere.
    from https://brainspace.readthedocs.io/en/latest/_modules/brainspace/null_models/spin.html#SpinPermutations.randomize

    Parameters
    ----------
    points_lh : ndarray, shape = (n_lh, 3)
        Array of points in a sphere, where `n_lh` is the number of points.
    points_rh : ndarray, shape = (n_rh, 3), optional
        Array of points in a sphere, where `n_rh` is the number of points. If
        provided, rotations are derived from the rotations computed for
        `points_lh` by reflecting the rotation matrix across the Y-Z plane.
        Default is None.
    unique : bool, optional
        Whether to enforce a one-to-one correspondence between original points
        and rotated ones. If true, the Hungarian algorithm is used.
        Default is False.
    n_rep : int, optional
        Number of random rotations. Default is 100.
    surface_algorithm : {'FreeSurfer', 'CIVET'}
        For 'CIVET', no flip is required to generate the spins for the right
        hemisphere. Only used when ``points_rh is not None``.
        Default is 'FreeSurfer'.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    result : dict[str, ndarray]
        Spin indices for left points (and also right, if provided).

    References
    ----------
    * Alexander-Bloch A, Shou H, Liu S, Satterthwaite TD, Glahn DC,
      Shinohara RT, Vandekar SN and Raznahan A (2018). On testing for spatial
      correspondence between maps of human brain structure and function.
      NeuroImage, 178:540-51.
    * Blaser R and Fryzlewicz P (2016). Random Rotation Ensembles.
      Journal of Machine Learning Research, 17(4): 1–26.
    * https://netneurotools.readthedocs.io

    """

    # Handle if user provides spheres
    if not isinstance(points_lh, np.ndarray):
        points_lh = np.array(points_lh)

    if points_rh is not None:
        if not isinstance(points_rh, np.ndarray):
            points_rh = np.array(points_rh)

    pts = {'lh': points_lh}
    if points_rh is not None:
        pts['rh'] = points_rh

        # for reflecting across Y-Z plane
        reflect = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    idx = {k: np.arange(p.shape[0]) for k, p in pts.items()}
    spin = {k: np.empty((n_rep, p.shape[0]), dtype=int)
            for k, p in pts.items()}
    if not unique:
        # tree = {k: cKDTree(p, leafsize=20) for k, p in pts.items()}
        tree = {k: KDTree(p, leafsize=20) for k, p in pts.items()}

    rs = check_random_state(random_state)

    rot = {}
    for i in range(n_rep):

        # generate rotation for left
        rot['lh'], temp = np.linalg.qr(rs.normal(size=(3, 3)))
        rot['lh'] *= np.sign(np.diag(temp))
        rot['lh'][:, 0] *= np.sign(np.linalg.det(rot['lh']))

        # reflect the left rotation across Y-Z plane
        if 'rh' in pts:
            if surface_algorithm.lower() == 'freesurfer':
                rot['rh'] = reflect @ rot['lh'] @ reflect
            else:
                rot['rh'] = rot['lh']

        for k, p in pts.items():
            if unique:
                dist = cdist(p, p @ rot[k])
                row, col = linear_sum_assignment(dist)
                spin[k][i, idx[k]] = idx[k][col]
            else:
                _, spin[k][i] = tree[k].query(p @ rot[k], k=1)

    return spin


def get_mid_pts(vers_sp, labs):
    """
    Get the mid points of each region
    args:
        vers_sp: vertices on the sphere
        labs: labels of each vertex
    return:
        mid_pts: mid points of each region
    """
    uni_labs = np.sort(np.unique(labs));
    mid_pts = []
    for uni_lab in uni_labs:
        sub_pts = vers_sp[labs == uni_lab];
        mid_pt = find_minimal_sum_point(sub_pts);
        mid_pts.append(mid_pt);
    
    mid_pts = np.array(mid_pts);

    return mid_pts

def trans_spins(spins, sph_labs, my_labs):
    """trans spin index from sph_lab to mylab (order)
    args:
        spins (np.array): n x nroi
        sph_labs: the order of rois in sphere
        my_labs: the order of rois in my target vector

    """

    sph2my = []
    # for each lab in sph, find the lab index in my_labs 
    # if no, use inf
    for lab in sph_labs:
        idx = np.where(np.array(my_labs)==lab)[0]
        if len(idx) > 0:
            sph2my.append(idx[0])
        else:
            sph2my.append(np.inf)
    sph2my = np.array(sph2my);
    
    spins_my_raw = sph2my[spins];
    # remove inf 
    spins_my = spins_my_raw[spins_my_raw < np.inf].reshape(-1, len(my_labs))
    return spins_my