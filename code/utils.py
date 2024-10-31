import numpy as np
from math import pi, sin, cos
from cv2 import warpPerspective, INTER_CUBIC
from imresize import imresize
from shutil import copy
from time import strftime, localtime
import os
import glob
from scipy.ndimage import measurements, interpolation
from scipy.io import loadmat


def random_augment(ims,
                   base_scales=None,
                   leave_as_is_probability=0.2,
                   no_interpolate_probability=0.3,
                   min_scale=0.5,
                   max_scale=1.0,
                   allow_rotation=True,
                   scale_diff_sigma=0.01,
                   shear_sigma=0.01,
                   crop_size=128):
    random_chooser = np.random.rand()
    if random_chooser < leave_as_is_probability:
        mode = 'leave_as_is'
    elif leave_as_is_probability < random_chooser < leave_as_is_probability + no_interpolate_probability:
        mode = 'no_interp'
    else:
        mode = 'affine'
    if base_scales is None:
        base_scales = [np.sqrt(np.prod(im.shape) / np.prod(ims[0].shape)) for im in ims]
    max_scale = np.min([max_scale])
    if mode == 'leave_as_is':
        scale = 1.0
    else:
        scale = np.random.rand() * (max_scale - min_scale) + min_scale
    scale_ind, base_scale = next((ind, np.min([base_scale])) for ind, base_scale in enumerate(base_scales)
                                 if np.min([base_scale]) > scale - 1.0e-6)
    im = ims[scale_ind]
    shift_to_center_mat = np.array([[1, 0, - im.shape[1] / 2.0],
                                    [0, 1, - im.shape[0] / 2.0],
                                    [0, 0, 1]])
    shift_back_from_center = np.array([[1, 0, im.shape[1] / 2.0],
                                       [0, 1, im.shape[0] / 2.0],
                                       [0, 0, 1]])
    if mode != 'affine':
        shift_to_center_mat = np.round(shift_to_center_mat)
        shift_back_from_center = np.round(shift_back_from_center)
    if mode == 'affine':
        scale /= base_scale
        scale_diff = np.random.randn() * scale_diff_sigma
    else:
        scale = 1.0
        scale_diff = 0.0
    if mode == 'leave_as_is' or not allow_rotation:
        reflect = 1
    else:
        reflect = np.sign(np.random.randn())
    scale_mat = np.array([[reflect * (scale + scale_diff / 2), 0, 0],
                          [0, scale - scale_diff / 2, 0],
                          [0, 0, 1]])
    shift_x = np.random.rand() * np.clip(scale * im.shape[1] - crop_size, 0, 9999)
    shift_y = np.random.rand() * np.clip(scale * im.shape[0] - crop_size, 0, 9999)
    shift_mat = np.array([[1, 0, - shift_x],
                          [0, 1, - shift_y],
                          [0, 0, 1]])
    if mode != 'affine':
        shift_mat = np.round(shift_mat)
    if mode == 'affine':
        theta = np.random.rand() * 2 * pi
    elif mode == 'no_interp':
        theta = np.random.randint(4) * pi / 2
    else:
        theta = 0
    if not allow_rotation:
        theta = 0
    rotation_mat = np.array([[cos(theta), sin(theta), 0],
                             [-sin(theta), cos(theta), 0],
                             [0, 0, 1]])
    if mode == 'affine':
        shear_x = np.random.randn() * shear_sigma
        shear_y = np.random.randn() * shear_sigma
    else:
        shear_x = shear_y = 0
    shear_mat = np.array([[1, shear_x, 0],
                          [shear_y, 1, 0],
                          [0, 0, 1]])
    transform_mat = (shift_back_from_center
                     .dot(shift_mat)
                     .dot(shear_mat)
                     .dot(rotation_mat)
                     .dot(scale_mat)
                     .dot(shift_to_center_mat))
    im_ds = []
    for i in range(im.shape[-1]):
        im_d = im[:,:,i]
        im_d = im_d.reshape(im_d.shape[0], im_d.shape[1], 1)
        im_d = np.clip(warpPerspective(im_d, transform_mat, (crop_size, crop_size), flags=INTER_CUBIC), 0, 1)
        im_ds.append(im_d)
    im_ds = np.asarray(im_ds)
    im_ds = im_ds.reshape(im_ds.shape[1], im_ds.shape[2], im_ds.shape[0])
    return im_ds


def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    y_sr += imresize(y_lr - imresize(y_sr,
                                     scale_factor=1.0/sf,
                                     output_shape=y_lr.shape,
                                     kernel=down_kernel),
                     scale_factor=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)

def preprocess_kernels(kernels, conf):
    if kernels is not None:
        return [kernel_shift(loadmat(kernel)['Kernel'], sf)
                for kernel, sf in zip(kernels, conf.scale_factors)]
    else:
        return [conf.downscale_method] * len(conf.scale_factors)

def kernel_shift(kernel, sf):
    current_center_of_mass = measurements.center_of_mass(kernel)
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    shift_vec = wanted_center_of_mass - current_center_of_mass
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    return interpolation.shift(kernel, shift_vec)

def prepare_result_dir(conf):
    if conf.create_result_dir:
        conf.result_path += '/' + conf.sample + strftime('_%b_%d_%H_%M_%S', localtime())
        os.makedirs(conf.result_path)
    if conf.create_code_copy:
        for py_file in glob.glob(os.path.dirname(__file__) + '/*.py'):
            copy(py_file, conf.result_path)

    return conf.result_path