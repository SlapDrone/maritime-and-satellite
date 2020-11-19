from typing import Union

import logging

import dask.array as da
import numpy as np


log = logging.getLogger(__name__)


def swapaxes_shuffle(
    arr:Union[da.Array, np.ndarray],
    n_classes:int=24,
    n_snr:int=12,
    samples_per_combination:int=4096
) -> Union[da.Array, np.ndarray]:
    """
    Expand, rotate and recollapse the axes of a multidimensional array where 
    the first axes traverses a ravelled set of value pairs (classes and snr).
    
    Has the effect of something like a shuffling operation on an out-of-core
    array without needing to compute and mix blocks. The returned array has 
    the data reordered so that the class/snr combinations are traversed first
    instead of the samples for each of these pairs.
    
    Parameters
    ----------
    arr: :obj:`np.ndarray` or :obj:`da.Array`
        An array where samples are distributed along the first axes, traversing
        sets of examples for pairs of "classes" and "snr" variables in order
    n_classes: int, optional
        Number of classes represented in array
    n_snr: int, optional
        Number of SNR values represented in array
    samples_per_combination: int, optional
        The number of examples appearing for each combination, for example the 
        data may contain 4096 samples for classification A and SNR 1, and so on.
        
    Returns
    -------
    :obj:`np.ndarray` or :obj:`dask.array.Array`
        An array of the same shape as the input, but reordered so that the 
        classification and SNR dimensions are traversed first and samples last,
        effectively putting the first sample for each SNR/classification combo
        next to each other before moving on to the second sample, etc.
    """
    last_axes = arr.shape[1:]
    exp_dims = arr.reshape(n_classes, n_snr, samples_per_combination, *last_axes)
    # traverse the classes and snr dimensions before moving down samples
    arr_T =  exp_dims.transpose(2, 1, 0, *tuple(range(3, len(exp_dims.shape))))
    # flatten out again with new ordering
    return arr_T.reshape(n_classes*n_snr*samples_per_combination, *last_axes)


def stack_interleave_flatten(arrs):
    """
    Performs folding and interleaving operation on a dask array.
    Chunks are assumed to be along axis 0 only (the samples dimension).
    This function folds a dask array on itself along axis 0, then
    interleaves the vertically adjacent elements (like shuffling a deck of cards
    by splitting it in two, placing one next to the other then interleaving both
    halves).
    Parameters
    ----------
    X : list of :obj:`dask.array.Array`
        A list of dask arrays each with at least two dimensions and multiple blocks
        along the first dimension
    Returns
    -------
    :obj:`dask.array.Array`
        The input array, with its elements now alternating between those of the
        first and second halves of the array
    See Also
    --------
    `fold_interleave`
    `fold_interleave_together`
    """
    assert all(len(a.shape) == len(arrs[0].shape) for a in arrs), (
        "arrays must have the same number of dimensions"
    )
    static_ax_ixs = range(2, len(arrs[0].shape) + 1)
    stk = da.stack(arrs, axis=0).transpose(1,0,*static_ax_ixs)
    rshp = (stk.shape[i] for i in static_ax_ixs)
    stk = stk.reshape((stk.shape[0]*stk.shape[1], *rshp))
    return stk

def shuffle_blocks(X):
    """
    Shuffles the order of the blocks present in a dask array
    Chunks are assumed to be along axis 0 only (the samples dimension).
    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0 only (typically the samples
        dimension)
    Returns
    -------
    :obj:`dask.array.Array`
        The input array with blocks reordered by a random permutation
    """
    blk_inds = list(np.ndindex(X.numblocks))
    log.debug("Shuffling blocks in dask array...")
    rand_perm = np.random.permutation(len(blk_inds))
    blk_inds = [blk_inds[p] for p in rand_perm]
    _X = da.concatenate(
        [X.blocks[blix] for blix in blk_inds], axis=0
    )
    return _X


def fold_interleave(X, repeats=4):
    """
    Performes iterative pseudo-shuffling operation on a dask array.
    Chunks are assumed to be along axis 0 only (the samples dimension).
    This function folds a dask array on itself along axis 0, then
    interleaves the vertically adjacent elements (like shuffling a deck of cards
    by splitting it in two, placing one next to the other then interleaving both
    halves), repeating `repeats` times.
    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension
    repeats: int
        The number of times to perform the folding shuffle operation
    Returns
    -------
    :obj:`dask.array.Array`
        The pseudo-shuffled input array
    See Also
    --------
    `stack_interleave_flatten`
    """
    n_blks = X.numblocks[0]
    n_fold = n_blks // 2
    for _ in range(repeats):
        p1, p2 = X.blocks[:n_fold], X.blocks[n_fold:2*n_fold]
        _X = stack_interleave_flatten([p1, p2])
        # if there's an odd number of blocks, put the last one (which was excluded)
        # at the beginning of the array so it gets shuffled next iteration
        if n_blks % 2 != 0:
            X = da.concatenate([X.blocks[-1], _X])
        else:
            X = _X
    return X


def check_blocks_match(arrs, axis=0):
    # check blocks consistent across inputs
    n_blks = arrs[axis].numblocks[0]
    assert all(a.numblocks[axis] == n_blks for a in arrs), (
        f"chunks should have the same size along axis {axis}!"
    )

    
def shuffle_blocks_together(arrs):
    """
    Shuffles the blocks of the input dask arrays by the same random permutation
    Used for synchronised shuffling operations on e.g. training inputs and labels.
    Blocks are assumed to be aligned along axis 0 only, as is typical for
    arrays containing training examples.
    Parameters
    ----------
    arrs: tuple of :obj:`dask.array.Array`
        A tuple of dask arrays with chunking along axis 0
    Returns
    -------
    tuple of :obj:`dask.array.Array`:
        The input dask arrays with their blocks shuffled by the same permutation
    """
    n_blks = arrs[0].numblocks[0]
    check_blocks_match(arrs)
    blk_inds = list(np.ndindex(n_blks))
    if len(blk_inds) == 1:
        return arrs
    log.debug("Shuffling blocks in dask arrays by same permutation...")
    rand_perm = np.random.permutation(len(blk_inds))
    blk_inds = [blk_inds[p] for p in rand_perm]
    for ix in range(len(arrs)):
        a = arrs[ix]
        arrs[ix] = da.concatenate(
            [a.blocks[blix] for blix in blk_inds], axis=0
        )
    return arrs


def fold_interleave_together(arrs, shuffle_blocks=True, repeats=4):
    """
    Performes synchronised pseudo-shuffling operation on the input dask arrays.
    Chunks are assumed to be along axis 0 only (the samples dimension). All
    chunks are assumed to be the same size!
    Parameters
    ----------
    arrs : list of :obj:`dask.array.Array`
        A list of dask arrays with at least two dimensions and multiple blocks along the
        first dimension (batches). Typically corresponding training inputs, labels etc.
    shuffle_blocks: bool
        Flag controlling whether block orders are additionally shuffled each
        iteration.
    repeats: int
        The number of times to perform the folding shuffle operation
    Returns
    -------
    tuple of :obj:`dask.array.Array`
        The pseudo-shuffled input arrays
    """
    # check blocks consistent across inputs
    n_blks = arrs[0].numblocks[0]
    check_blocks_match(arrs)
    if n_blks == 1:
        return arrs
    n_fold = n_blks // 2
    for ii in range(repeats):
        if shuffle_blocks:
            arrs = shuffle_blocks_together(arrs)
        for ix in range(len(arrs)):
            a = arrs[ix]
            ap1, ap2 = a.blocks[:n_fold], a.blocks[n_fold:2*n_fold]
            _a = stack_interleave_flatten([ap1, ap2])
            # if there's an odd number of blocks, put the last one (which was excluded)
            # at the beginning of the array so it gets shuffled next iteration
            if n_blks % 2 != 0:
                arrs[ix] = da.concatenate([a.blocks[-1], _a]).rechunk(a.chunks)
            else:
                arrs[ix] = _a.rechunk(a.chunks)
    return arrs


def get_full_blocks(X):
    """
    Returns the subset of the input dask array made of full blocks.
    Full blocks are assumed to be those of the max size present, and blocks
    are assumed to be aligned along axis 0.
    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0 only (typically the samples
        dimension)
    Returns
    -------
    :obj:`dask.array.Array`
        A subset of the input array where the blocks are all those with the
        max chunk size present in the original array.
    """
    # get block sizes
    size_chunks = X.chunks[0]
    full_blk_ixs = np.argwhere(np.array(size_chunks) == max(size_chunks))
    if len(full_blk_ixs) == X.numblocks[0]:
        return X
    return da.concatenate([X.blocks[ix] for ix in full_blk_ixs], axis=0)


def get_incomplete_blocks(X):
    """
    Returns the subset of the input dask array made of incomplete blocks.
    Incomplete blocks are assumed to be those smaller than the max size present,
    and blocks are assumed to be aligned along axis 0.
    Parameters
    ----------
    X: :obj:`dask.array.Array`
        A dask array with chunking along axis 0 only (typically the samples
        dimension)
    Returns
    -------
    :obj:`dask.array.Array`
        A subset of the input array where the blocks are all those with size
        smaller than the max chunk size present in the original array.
    """
    # get block sizes
    size_chunks = X.chunks[0]
    inc_blk_ixs = np.argwhere(np.array(size_chunks) != max(size_chunks))
    if len(inc_blk_ixs) == 1:
        return X.blocks[inc_blk_ixs[0]]
    elif len(inc_blk_ixs) > 1:
        return da.concatenate([X.blocks[ix] for ix in inc_blk_ixs], axis=0)
    
    
def combine_incomplete_blocks(X):
    """
    Merges incomplete blocks of a dask array into complete blocks
    Incomplete blocks are assumed to be those smaller than the max size present,
    and blocks are assumed to be divided only along axis 0 (e.g. training
    examples).
    New blocks formed by combining incomplete blocks are stuck on the end of the
    output array.
    Parameters
    ----------
    X : :obj:`dask.array.Array`
        A dask array with at least two dimensions and multiple blocks along the
        first dimension
    Returns
    -------
    :obj:`dask.array.Array`
        The input dask array, rechunked and reassembled with as many blocks with
        the max original size present as possible.
    """
    # get block sizes
    size_chunks = X.chunks[0]
    # identify incomplete blocks in X
    inc_blk_ixs = np.argwhere(np.array(size_chunks) != max(size_chunks))
    # if there are a few incomplete blocks, put em together and try to get full blocks
    if len(inc_blk_ixs) > 1:
        inc_blocks = da.concatenate([X.blocks[ix] for ix in inc_blk_ixs], axis=0)
        _chunk_cfg = [max(size_chunks)]
        # ensure chunks span full extent of all other dimensions than first
        _chunk_cfg.extend([-1 for _ in len(X.numblocks)-1])
        inc_blocks = inc_blocks.rechunk(_chunk_cfg)
        extra_full_blocks = [b for b in inc_blocks.blocks if b.chunksize[0] == max(size_chunks)]
    elif len(inc_blk_ixs) == 1:
        extra_full_blocks = []
        inc_blocks = X.blocks[inc_blk_ixs[0]]
    else:
        return X
    # identify full blocks in original X
    full_blk_ixs = np.argwhere(np.array(size_chunks) == max(size_chunks))
    # get the original full blocks
    full_blocks = da.concatenate([X.blocks[ix] for ix in full_blk_ixs], axis=0)
    # if we formed new full blocks by combining incomplete ones,  stick em together in a new arr
    if extra_full_blocks:
        extra_full_blocks = da.concatenate(extra_full_blocks)
        log.debug("extra full blocks formed by combining incomplete ones:", extra_full_blocks)
        full_blocks = da.concatenate([full_blocks, extra_full_blocks], axis=0)
        inc_blocks = da.concatenate([b for b in inc_blocks.blocks if b.chunksize[0] != max(size_chunks)])
    return da.concatenate([full_blocks, inc_blocks])


def synchronised_pseudoshuffle(arrs, shuffle_blocks=True, repeats=4):
    """
    Shuffles dask arrays so corresponding elements end up in the same place
    Designed to shuffle e.g. pairs of input/ground truth arrays to achieve a 
    degree of data dispersal without invoking a full random shuffle and 
    blowing up memory. Combines shuffling of blocks within the array with a repeated
    'fold interleave' operation. Accounts for complete and incomplete blocks.
    Parameters
    ----------
    arrs: tuple of :obj:`dask.array.Array`
        A dask array with chunks along axis = 0. Typically training examples.
    shuffle_blocks: bool, optional
        Flags whether to shuffle the blocks during the fold interleave operation
        (if not, will just do once at the end).
    repeats: int, optional
        Specifies how many times to perform the `fold_interleave_together`
        operation
        
    Returns
    -------
    tuple of :obj:`dask.array.Array`:
        Pseudo-shuffled versions of input arrays

    See Also
    --------
    `shuffle_blocks_together`
    `fold_interleave_together`
    """
    # merge any incomplete blocks in the array and stick em on the end
    arrs = [combine_incomplete_blocks(a) for a in arrs]
    # distinguish the complete and incomplete blocks
    arrs_inc = [get_incomplete_blocks(a) for a in arrs]
    # fold the part of the arrays with complete blocks on itself repeats times,
    # interleaving the now vertically-stacked elements and flattening into two new arrays
    # shuffle the blocks each repetition if flagged
    arrs_full = [get_full_blocks(a) for a in arrs]
    arrs_full_shf = fold_interleave_together(arrs_full,
                                             shuffle_blocks=shuffle_blocks,
                                             repeats=repeats)
    arrs_to_conc = [[afs] for afs in arrs_full_shf]
    for ix in range(len(arrs_to_conc)):
        if arrs_inc[ix] is not None:
            arrs_to_conc[ix].append(arrs_inc[ix])
    # tag on the incomplete blocks and shuffle them all one more time
    conc_arrs = [da.concatenate(a, axis=0) for a in arrs_to_conc]
    return shuffle_blocks_together(conc_arrs)


def chunk_generator(arrs, shuffle_blocks=False):
    """
    Generator function to yield corresponding blocks of a set of dask arrays
    Useful for machine learning model training data generators.
    
    Parameters
    ----------
    arrs : tuple of :obj:`dask.array.Array`
        An tuple of dask arrays chunked along axis 0
        
    Yields
    ------
    tuple of :obj:`np.ndarray` :
        A tuple of numpy arrays corresponding to the blocks of the inputs
    """
    check_blocks_match(arrs, axis=0)
    # get the index pairs along the target axis which make up each chunk
    blk_inds = list(np.ndindex(arrs[0].numblocks[0]))
    if shuffle_blocks:
        rand_perm = np.random.permutation(len(blk_inds))
        blk_inds = [blk_inds[p] for p in rand_perm]
    # we iterate over the block number and indices (shared for images/masks all darrs)
    for i, blk_ind in enumerate(blk_inds):
        yield tuple(a.blocks[blk_ind].compute() for a in arrs)