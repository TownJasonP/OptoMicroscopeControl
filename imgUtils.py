from skimage import measure, filters, morphology, io
import numpy as np
from scipy import spatial
from scipy.ndimage import distance_transform_edt, center_of_mass


def binarize_img(img, thresh, min_cell_size):

    smooth_img = filters.gaussian(img, 2, preserve_range = True)
    binary_img = smooth_img > thresh
    binary_img = morphology.remove_small_objects(binary_img, min_cell_size)
    binary_img = morphology.remove_small_holes(binary_img, min_cell_size)
    binary_img = morphology.binary_erosion(binary_img, selem = morphology.disk(4))

    return(binary_img)


def get_closest_cell(input_bin, x_coord, y_coord):
    '''
    take a binary mask, potentially with multiple objects and return a mask containing
    only the object closest to (x_coord, y_coord)
    '''
    if np.sum(input_bin) > 0:
        input_lbl = measure.label(input_bin)
        regions = measure.regionprops(input_lbl)

        label_ids = np.array([r.label for r in regions])
        dists = []

        for r in regions:
            yc, xc = r.centroid
            dy = yc - y_coord
            dx = xc - x_coord
            dists.append(np.hypot(dx, dy))

        dists = np.array(dists)

        closest_label_id = label_ids[np.argmin(dists)]

        output_bin = input_lbl == closest_label_id

        return(output_bin)
    else:
        return(input_bin)
    

def get_fluor_polar_angle(input_bin, input_fluor):
    '''
    Given a binary images representing the location of a single cell and the
    corresponding fluorescence image, calculate a polatity angle based on the difference
    between the weighted and unweighted centroids
    '''

    input_lbl = measure.label(input_bin)
    region = measure.regionprops(input_lbl, intensity_image = input_fluor)[0]
    y_c, x_c = region.centroid
    y_w, x_w = region.weighted_centroid

    theta = np.arctan2(y_w - y_c, x_w - x_c)

    return(theta)


def get_move_polar_angle(input_bin_initial, input_bin_final):
    '''
    Given two binary images representing the location of a single cell
    calculate a movement vector based on the relative locations of the centroids
    and locate the lateral sides of the cell relative to this movement vector
    '''

    # collect centroid info to define transformation below
    input_lbl_i = measure.label(input_bin_initial)
    region_i = measure.regionprops(input_lbl_i)[0]
    y_i, x_i = region_i.centroid

    input_lbl_f = measure.label(input_bin_final)
    region_f = measure.regionprops(input_lbl_f)[0]
    y_f, x_f = region_f.centroid

    theta = np.arctan2(y_f - y_i, x_f - x_i)

    return(theta)



def get_tform_mat(input_bin, theta):
    '''
    Given a binary image representing a single cell and an angle,
    generate the transformation matrix that allows for rotation about
    the center of mass of the binary input by theta
    '''

    input_lbl = measure.label(input_bin)
    region = measure.regionprops(input_lbl)[0]
    y_c, x_c = region.centroid

    c = np.cos(theta)
    s = np.sin(theta)
    t_y = -y_c
    t_x = -x_c

    # translation followed by rotation
    tsl_mat = np.array([[1, 0, 0], [0, 1, 0], [t_x, t_y, 1]])
    rot_mat = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])

    tfm_mat = np.matmul(tsl_mat, rot_mat)

    return(tfm_mat)


def get_lateral_points(input_bin, theta):
    '''
    Given a binary image representing the location of a single cell
    and the corresponding fluorescence image, calculate a polarity vector
    based on the fluorescence image and locate the lateral sides of the cell
    relative to the polarity vector
    '''

    tfm_mat = get_tform_mat(input_bin, theta)

    # collect coordinates and transform
    all_ys, all_xs = np.where(input_bin)
    all_zs = np.ones_like(all_ys)

    coords_initial = np.array([all_xs, all_ys, all_zs]).T
    coords_tformed = np.matmul(coords_initial, tfm_mat)

    x, y, z = coords_tformed.T

    # look for furthest lateral sides in transformed coordinates
    ymax = np.max(y[(x < 1) & (x > -1)])
    ymin = np.min(y[(x < 1) & (x > -1)])

    # apply inverse transformation to lateral coordinates
    x_R, y_R, placeholder = np.matmul(np.array([0, ymin, 1]), np.linalg.inv(tfm_mat))
    x_L, y_L, placeholder = np.matmul(np.array([0, ymax, 1]), np.linalg.inv(tfm_mat))

    return([x_R, y_R, x_L, y_L])



def get_back_point(input_bin, theta):
    '''
    Given a binary image representing the location of a single cell
    and the corresponding fluorescence image, calculate a polarity vector
    based on the fluorescence image and locate the lateral sides of the cell
    relative to the polarity vector
    '''

    tfm_mat = get_tform_mat(input_bin, theta - math.pi/2)

    # collect coordinates and transform
    all_ys, all_xs = np.where(input_bin)
    all_zs = np.ones_like(all_ys)

    coords_initial = np.array([all_xs, all_ys, all_zs]).T
    coords_tformed = np.matmul(coords_initial, tfm_mat)

    x, y, z = coords_tformed.T

    # look for furthest lateral sides in transformed coordinates
    ymin = np.min(y[(x < 1) & (x > -1)])

    # apply inverse transformation to lateral coordinates

    x_B, y_B, placeholder = np.matmul(np.array([0, ymin, 1]), np.linalg.inv(tfm_mat))

    return([x_B, y_B])
