"""
preprocessing.py

Provides preprocessing transformations conforming to the sklearn
Transformer API and auxiliary functions for preparing raster/image data for
machine learning operations.

Main high-level interfaces are currently training and inference pipelines for
semantic segmentation (see `get_image_training_pipeline`), along with
generators for image augmentation (see e.g. `get_fancy_aug_datagen`).
"""
import random
import logging
from typing import Union, Tuple
from functools import partial

import numpy as np
import dask.array as da
import rasterio
import timbermafia as tm
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline#, FeatureUnion#, make_pipeline
from sklearn.preprocessing import FunctionTransformer#, minmax_scale, scale, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from skimage import data
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.preprocessing import image



log = logging.getLogger(__name__)

#from ImageDataAugmentor.image_data_augmentor import *
from albumentations.core.transforms_interface import ImageOnlyTransform


def get_image_training_pipeline(
    patch_size:Tuple[int]=(256,256),
    seed:int=42,
    channels:Tuple[int]=(0,1,2,3),
    **kwargs
):
    """
    Factory returning a pipeline to preprocess raster data for training.

    Applies operations for selecting, tiling and shuffling dask arrays extracted 
    from rasters into a stack of patches ready for division into batches.

    Arguments
    ---------
    window_dims: array_like
        The desired row, col size of patches used for training. Together with
        the downstream batch_size, this is usually limited by GPU memory.
        For example, (256,256) might permit O(10) images per batch.
    seed: int
        Random seed for shuffling patches
    channels: list of innt
        Channel indices to use

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline encapsulating the preprocessing stages
    """
    steps = [
        ('channel_selector', ChannelSelector(channels=list(channels))),
        ('padder', ArrayPadder(*patch_size)),
        ('tiler', Tiler(patch_size))
    ]
    if seed is not None:
        steps.append(('synchronised_shuffler', SynchronisedShuffler(seed=seed)))
    pipeline = Pipeline(steps=steps)
    return pipeline


class Tiler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer batching a (H, W, C) image array into (B, h, w, C); h, w < H, W

    Convert a 3D image array into 4D array of tiles with shape window_dims
    (for use in e.g. ML inference) in the first two dimensions (w, h if channels
    last).

    repeating the necessary rows and columns back from the edge to form full
    tiles.

    Invertible.

    As a consequence of using this in ML inference, parts of the edges/corners
    will be slightly oversampled by some fraction of a window, ie the same edge
    region may appear twice in two different training tiles.

    Parameters
    ----------
    window_dims: array_like
        (h, w); the dimensions of the small tiles making up the batches

    Attributes
    ----------
    cropper_: WindowFitter
        WindowFitter transformer to crop evenly-dividing piece of the array
    stacker_: TileStacker
        TileStacker used to divide up the part of the array that fits neatly
        into an integer numbe rof tilers
    extra_rows_: array_like
        The excess rows at the end which don't fully form a tile
    extra_cols_: array_like
        The excess columns at the end which don't fully form a tile

    Notes
    -----
    Uses 3 stackers and croppers to tile the evenly-dividing piece of the array,
    and the extra rows/columns respectively, plus a fourth object to address
    the bottom right corner.

    See fit method implementation for full details of the intermediate stages
    """
    def __init__(self, window_dims):
        self.window_dims = window_dims

    def fit(self, X, y=None):
        # identify the part that when cropped will nicely divide into windows
        self.cropper_ = WindowFitter(window_dims=self.window_dims).fit(X)
        # prepare transformation of cropped area to stacked windows
        self.log.debug(f"Shape going into stacker: {X.shape}")
        self.stacker_ = TileStacker(self.window_dims).fit(
            X[:self.cropper_.row_max_, :self.cropper_.col_max_]
        )
        # rows and columns
        self.extra_rows_ = X[self.cropper_.row_max_:]
        self.extra_cols_ = X[:, self.cropper_.col_max_:]
        row_remainder = (X.shape[0] + self.extra_rows_.shape[0]) % self.window_dims[0]
        col_remainder = (X.shape[1] + self.extra_cols_.shape[0]) % self.window_dims[1]
        # if there are additional rows that don't fit cleanly into the window-cropped area
        if self.extra_rows_.size > 0:
            # grab the last possible set of rows forming full windows
            self.last_window_rows_ = X[-self.window_dims[0]:]
            self.last_rows_wf_ = WindowFitter(window_dims=self.window_dims)
            self.last_window_rows_ = self.last_rows_wf_.fit_transform(
                self.last_window_rows_
            )
            # figure out how to partition it into full windows
            self.r_stacker_ = TileStacker(self.window_dims).fit(self.last_window_rows_)
            # if the columns do not evenly divide into window breadths,
            # this should exclude a partial window at the (bottom) right
             # figure out indices of windows from privileged rows and columns at edge
            self.lr_max_ind_ = self.last_rows_wf_.n_windows_r_ * self.window_dims[0]
        # ditto for columns
        if self.extra_cols_.size > 0:
            # grab the last possible set of cols forming full windows
            self.last_window_cols_ = X[:, -self.window_dims[1]:]
            self.last_cols_wf_ = WindowFitter(window_dims=self.window_dims)
            self.last_window_cols_ = self.last_cols_wf_.fit_transform(
                self.last_window_cols_
            )
            # figure out how to partition it into full windows
            self.c_stacker_ = TileStacker(self.window_dims).fit(self.last_window_cols_)
            # if the rows do not evenly divide into window heights,
            # this should exclude a partial window at the bottom (right)
            # figure out indices of windows from privileged rows and columns at edge
            self.lc_max_ind_ = self.last_cols_wf_.n_windows_c_ * self.window_dims[1]
        # if neither dimension factorises into windows we need to treat the bottom
        # right corner separately
        if self.extra_rows_.size > 0 and self.extra_cols_.size > 0:
            self.br_corner_ = X[-self.window_dims[0]:, -self.window_dims[1]:]
        # we just treat this as one window on its own, with the bottom
        # right corner aligned with that of the full array
        # figure out the indices in the eventual 4D array which correspond to
        # each component
        # elements up to this index can be nicely retiled
        self.crop_max_ix_ = self.cropper_.n_windows_r_ * self.cropper_.n_windows_c_
        # figure out indices of windows from privileged rows and columns at edge
        #self.lr_max_ind_ = self.last_rows_wf_.n_windows_r_ * self.window_dims[0]
        #self.lc_max_ind_ = self.last_cols_wf_.n_windows_c_ * self.window_dims[1]
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)

        Returns
        -------
        dask.array.Array
            Tiled (B, h, w, C) Dask array
        """
        # anticipate output arrays
        X_out_components = []
        # first extract the nicely dividing part
        X_crop = self.cropper_.transform(X)
        # tile it
        self.X_crop_tiled = self.stacker_.transform(X_crop)
        # tile the extra rows
        X_out_components.append(self.X_crop_tiled)
        if self.extra_rows_.size > 0:
            X_out_components.append(
                self.r_stacker_.transform(self.last_window_rows_)
            )
        if self.extra_cols_.size > 0:
            X_out_components.append(
                self.c_stacker_.transform(self.last_window_cols_)
            )
        if self.extra_rows_.size > 0 and self.extra_cols_.size > 0:
            X_out_components.append(
                self.br_corner_.reshape((1, *self.br_corner_.shape))
            )
        if isinstance(X, da.Array):
            lib = da
        elif isinstance(X, np.ndarray):
            lib = np
        else:
            raise ValueError("array data type not understood")
        return lib.concatenate(X_out_components)

    def inverse_transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Tiled (B, h, w, C) Dask array

        Returns
        -------
        dask.array.Array
            Input (H, W, C) Dask array (upon which fit was called)
        """
        # determine whether to use numpy or dask
        if isinstance(X, da.Array):
            lib = da
        elif isinstance(X, np.ndarray):
            lib = np
        else:
            raise ValueError(
                f"array type {type(X)} not understood. expected np or dask array"
            )
        X_crop_tiled, X_rest_tiled = X[:self.crop_max_ix_], X[self.crop_max_ix_:]
        X_crop = self.stacker_.inverse_transform(X_crop_tiled)
        # stop here if array perfectly factorised into windows
        if self.extra_rows_.size == 0 and self.extra_cols_.size == 0:
            return X_crop
        # reconstruct the partial window of rows at the bottom edge
        if self.extra_rows_.size > 0:
            ix_lr = self.last_rows_wf_.n_windows_c_
            X_last_window_rows = self.r_stacker_.inverse_transform(
                X_rest_tiled[:ix_lr]
            )
            # get the non-duplicated part of the window to stitch back on
            X_last_window_rows = X_last_window_rows[
                -self.extra_rows_.shape[0]:
            ]
            # if there are no extra columns, concatenate and return here
            if self.extra_cols_.size == 0:
                return lib.concatenate([X_crop, X_last_window_rows], axis=0)
            X_rest_tiled = X_rest_tiled[ix_lr:]
        # reconstruct the partial window of columns at the rightmost edge
        if self.extra_cols_.size > 0:
            ix_lc = self.last_cols_wf_.n_windows_r_
            X_last_window_cols = self.c_stacker_.inverse_transform(
                X_rest_tiled[:ix_lc]
            )
            # get the non-duplicated part of the window to stitch back on
            X_last_window_cols = X_last_window_cols[
                :, -self.extra_cols_.shape[1]:
            ]
            # if there are no extra rows, concatenate and return here
            if self.extra_rows_.size == 0:
                return lib.concatenate([X_crop, X_last_window_cols], axis=1)
        # otherwise we need to stick the corner to a row/col first before conc.
        br_corner = X_rest_tiled[
            -1, -self.extra_rows_.shape[0]:, -self.extra_cols_.shape[1]:
        ]
        # workaround - block doesn't seem to work here? or im just a dafty
        c1=da.concatenate([X_crop, X_last_window_rows])
        c2=da.concatenate([X_last_window_cols, br_corner])
        return lib.concatenate([c1, c2], axis=1)


#combined_pipeline = FeatureUnion(transformer_list=[('pipeline_1', pl1 ), ('pipeline_2', pl2 )])
class WindowFitter(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer to crop an image-like array (along the first two axes)

    The output is the subset of the input which fits into an integer multiple
    of window_dims.

    Parameters
    ----------
    window_dims: array_like, optional
        h, w of eventual window/tile size

    Attributes
    ----------
    n_windows_r_: int
        number of full window breadths along the rows
    n_windows c_: int
        number of full window breadths down the columns
    row_max_: int
        max row index
    col_max_: int
        max col index
    """
    def __init__(self, window_dims=(224,224), behaviour='crop'):
        """ defaults to adding +/- 2 pixels in each dimension """
        self.window_dims = window_dims

    def fit(self, X, y=None):
        # check args
        self.n_rows_window, self.n_cols_window = self.window_dims
        assert self.n_rows_window == self.n_cols_window, (
            "Use square window dimensions!"
        )
        # calculate number of full window-breadths in each dimension
        self.n_windows_r_, self.n_windows_c_ = [
            int(n) for n in np.array(X.shape[:2])/self.n_rows_window
        ]
        # calc indices used to crop
        self.row_max_ = self.n_windows_r_*self.n_rows_window
        self.col_max_ = self.n_windows_c_*self.n_cols_window
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array or np.ndarray
            Input (H, W, C) array (upon which fit was called)

        Returns
        -------
        dask.array.Array or np.ndarray
            Cropped (H', W', C) array; H' <= H, W' <= W
        """
        self.log.debug("Transforming input data to match integer number of "
                  "windows...")
        # get the part of the array which divides evenly into windows
        X_ = X[:self.row_max_, :self.col_max_]
        return X_



class TileStacker(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer batching a (H, W, C) image array into (B, h, w, C); h, w < H, W

    Convert a 3D image array into 4D array of tiles with shape window_dims
    (for use in e.g. ML inference) in the first two dimensions (w, h if channels
    last).

    This method works only where the input array neatly divides into tiles.
    For a general version dealing with cases which don't nearly divide see Tiler.

    Invertible.

    See:
    https://stackoverflow.com/questions/42297115/
    numpy-split-cube-into-cubes/42298440#42298440

    Parameters
    ----------
    hypertile_shape: array_like
        (h, w); the dimensions of the small tiles making up the batches

    Attributes
    ----------
    X_hypertile_shape_: array_like
        The full shape in 3D of each tile/patch
    X_old_shape_: array_like
        Copy of the original shape of the array
    X_repeats_: array_like
        Array with the ratios of the size to the tile size in each dimension
    X_tmpshape_: array_like
        An intermediary higher-dimensional shape to facilitate the tiling
    X_order_: array_like
        The order of the higher-dimensional X_tmpshape_ axis to transpose by
        in the final step of the tiling operation

    To Do
    -----
    Normalise the attribute names for parity with the other related tiling
    methods

    See Also
    --------
    WindowFitter
    Tiler
    """
    def __init__(self, hypertile_shape:tuple=(224,224)):
        """
            Signature:
                hypertile_shape: a 2-tuple, list or iterable specifying the
                                 row, col shape of the desired hypertiles.
                                 for example, (224,224) (px)
                                 final dimension is assumed to be channels
                                 and is preserved (so 224x224x3 cubes will
                                 be produced for RGB).

            For example:
                8 x 8 array w/ new_shape (2,2) -> (16,2,2 array)

        """
        self.hypertile_shape = hypertile_shape

    def fit(self, X, y=None):
        # e.g. if given an image 9000 * 16000 * 3 and provide hypertile size (200, 200)
        # this will be (200, 200, 3). likewise if mask is 9000 * 16000 * 2 => (200, 200, 2)
        self.X_hypertile_shape_ = np.array(list(self.hypertile_shape) + [X.shape[-1]])
        self.X_old_shape_ = np.array(X.shape)
        self.X_repeats_ = (self.X_old_shape_ / self.X_hypertile_shape_).astype(int)
        self.X_tmpshape_ = np.stack((self.X_repeats_, self.X_hypertile_shape_), axis=1)
        self.X_tmpshape_ = self.X_tmpshape_.ravel()
        self.X_order_ = np.arange(len(self.X_tmpshape_))
        self.X_order_ = np.concatenate([self.X_order_[::2], self.X_order_[1::2]])
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array or numpy.ndarray
            Input (H, W, C) array (upon which fit was called)

        Returns
        -------
        dask.array.Array or numpy.ndarray
            Tiled (B, h, w, C) array
        """
        self.log.debug("Reshaping array into tiles...")
        # adapt to channel dimension of input
        X_tmpshp = (*self.X_tmpshape_[:-1], X.shape[-1])
        X_ht_shp = (*self.X_hypertile_shape_[:-1], X.shape[-1])
        # new_shape must divide old_shape evenly or else ValueError will be raised
        try:
            X_ = X.reshape(X_tmpshp).transpose(*self.X_order_).reshape(
                -1, *X_ht_shp
            )
        except Exception as e:
            self.log.error(e)
            self.log.debug(f"tmp: {self.X_tmpshape_}, ord: {self.X_order_}, "
                           f"x_ht_shp: {self.X_hypertile_shape_}")
            raise
        return X_

    def inverse_transform(self, X, y=None):
        """
        Parameters
        ----------
        X: dask.array.Array
            Tiled (B, h, w, C) array

        Returns
        -------
        dask.array.Array
            Input (H, W, C) array (upon which fit was called)
        """
        self.log.debug("Reshaping array into tiles...")
        X_N_, X_new_shape_ = X.shape[0], X.shape[1:]
        # update channel dim
        old_shp = np.array((*self.X_old_shape_[:-1], X.shape[-1]))
        X_repeats_ = (old_shp / X_new_shape_).astype(int)
        X_tmpshape_ = np.concatenate([X_repeats_, X_new_shape_])
        X_order_ = np.arange(len(X_tmpshape_)).reshape(2, -1).ravel(order='F')
        # adapt to channel dimension of input
        # transform
        X_ = X.reshape(X_tmpshape_).transpose(*X_order_).reshape(old_shp)
        return X_


class DimensionAdder(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Simple Transformer to add a channel dimension to an array if it's 2D

    For example, array with shape (H, W) -> (H, W, 1)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(X.shape) == 2:
            self.log.debug("Adding a third (channel) dimension to array...")
            return X.reshape(list(X.shape) + [1])
        elif len(X.shape) == 3:
            self.log.debug("Leaving array shape alone...")
            return X
        else:
            raise ValueError("X should have two (row/col) or three dimensions (+channel)")


class SimpleInputScaler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Simple Transformer wrapper for scaling an array

    Parameters
    ----------
    sf: float
        scale factor, typically 1/255. to map 8-bit RGBs -> [0, 1]
    """
    def __init__(self, sf=1/255.):
        self.sf = sf
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X * self.sf


class Float32er(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Simple Transformer wrapper for casting an array to 32-bit float
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.astype(np.float32)


class SynchronisedShuffler(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer to (un)shuffle array along axis 0 with a seeded permutation

    Works with Dask or Numpy arrays.

    Invertible.

    Parameters
    ----------
    seed: int
        random seed value

    Attributes
    ----------
    random_index_permutation_: array_like
        the permutation of indices specified by the random seed
    inverse_permutation_: array_like
        the inverse permutation of indices to unshuffle the array
    """
    def __init__(self, seed:int=42):
        self.seed = seed

    def fit(self, X, y=None):
        self.random_index_permutation_ = (
            np.random.RandomState(seed=self.seed).permutation(X.shape[0])
        )
        self.inverse_permutation_ = np.argsort(self.random_index_permutation_)
        return self

    def get_arr_perm(self, arr, perm):
        if isinstance(arr, np.ndarray):
            return arr[self.random_index_permutation_]
        elif isinstance(arr, da.Array):
            return da.slicing.shuffle_slice(arr, self.random_index_permutation_)

    def transform(self, X, y=None):
        X_ = self.get_arr_perm(X, self.random_index_permutation_)
        return X_

    def inverse_transform(self, X, y=None):
        X_ = self.get_arr_perm(X, self.inverse_permutation_)
        return X_


class ChunkBatchAligner(TransformerMixin, BaseEstimator, tm.Logged):
    """
    Transformer to rechunk an array along axis 0 so 1 chunk = M * batch_size

    Parameters
    ----------
    batch_size: int
        the batch size
    ideal_chunksize: int
        a precalculated "ideal" chunk size based on RAM considerations, for
        example this might be 60 with a batch_size of 7, which will result in
        9 batches * 7 samples = 63 samples per chunk, the closest integer

    Attributes
    ----------
    chk_cfg_: array_like
        the chunk shape of the reshaped array
    batches_per_chunk_: int
        the closest integer number of batches which will fit in a chunk of
        ideal_chunksize
    """
    def __init__(self, batch_size=None, ideal_chunksize=None):
        self.batch_size=batch_size
        self.ideal_chunksize = ideal_chunksize
    def fit(self, X, y=None):
        assert self.batch_size is not None
        # align specified ideal chunksize if provided, otherwise use current chunksize
        chunksize = X.chunksize[0] if self.ideal_chunksize is None else self.ideal_chunksize
        #self.divisible = True if chunksize % self.batch_size == 0 else False
        self.batches_per_chunk_ = round(chunksize / self.batch_size)
        self.chk_cfg_ = (self.batches_per_chunk_*self.batch_size, -1 , -1, -1)
        return self
    def transform(self, X, y=None):
        #if not self.divisible:
        return X.rechunk(self.chk_cfg_)
        #return X


class Rechunker(BaseEstimator, TransformerMixin, tm.Logged):
    """
    Transformer to adjust size of dask chunks between computations

    Parameters
    ----------
    chunks: array_like or str or int, optional
        see dask.array.Array.rechunk
    threshold, optional
        see dask.array.Array.rechunk
    block_size_limit: array_like or int, optional
        see dask.array.Array.rechunk
    """
    def __init__(self, chunks='auto', threshold=None, block_size_limit=None):
        self.chunks = chunks
        self.threshold = threshold
        self.block_size_limit = block_size_limit

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if isinstance(X, da.Array):
            return X.rechunk(chunks=self.chunks,
                             threshold=self.threshold,
                             block_size_limit=self.block_size_limit)
        return X


class ChannelSelector(TransformerMixin, BaseEstimator):
    """
    Transformer to select channels / project out slices along last axis of array

    Parameters
    ----------
    channels: array_like, optional
        Indices of channels to select
    """
    def __init__(self, channels=[0,1,2]):
        self.channels = channels
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[:,:,self.channels]


class ArrayPadder(TransformerMixin, BaseEstimator):
    """
    Transformer to pad an image array up to a certain shape with constant values

    Pad an array at the end of each of the first two dimensions with constant values
    such that each of these is at least min_rows, min_cols respectively

    Parameters
    ----------
    min_rows: int
        the minimum number of acceptable rows in the output
    max_cols: int
        the minimum number of acceptable cols in the output
    constant_values: int, optional
        the values to pad with

    Attributes
    ----------
    padding_rows_: int
        number of rows which are padded. requires fit.
    padding_cols_: int
        number of cols which are padded. requires fit.
    pad_width_: array_like
        the calculated pad_width shape parameter passed to np.pad

    Notes
    -----
    If used on an array from a raster, the choice of using the end of each dim
    will not shift the origin (top left), so the geotransform is unchanged
    """
    def __init__(self, min_rows, min_cols, constant_values=1):
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.constant_values = constant_values

    def fit(self, X, y=None):
        assert len(X.shape) == 3, (
            f"shape of X should be (rows, cols, channels). got {X.shape}!"
        )
        self.padding_rows_ = max(self.min_rows - X.shape[0], 0)
        self.padding_cols_ = max(self.min_cols - X.shape[1], 0)
        self.pad_width_ =(
           (0, self.padding_rows_), # start, end (both axes)
           (0, self.padding_cols_),
           (0, 0) # do not pad channels
        )
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            lib = np
        elif isinstance(X, da.Array):
            lib = da
        else:
            raise TypeError(f"Type {type(X)} of {X} not understood in pad."
                            " Should be a numpy/dask array.")
        return lib.pad(X,
                       pad_width=self.pad_width_,
                       mode='constant',
                       constant_values=self.constant_values)