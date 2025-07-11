# tensorOps.py
# Misc tensor operations
import torch

# return MxM matrix, with all ones except the diagonal from T[0, 0] to T[M, M]
def diag_zeros(M):
    """
    Return MxM matrix of all ones except the diagonal from T[0, 0] to T[M, M]
    Args:
        M (int): The size of the matrix
    Returns:
        torch.Tensor: A MxM matrix of all ones except the diagonal from T[0, 0] to T[M, M]
    """
    diag_zeroes = torch.ones((M, M), dtype=torch.float32)       # Create all ones matrix
    diag_zeroes -= torch.eye(M)                                 # Remove ones in diagonal, connecting value in row to each column, exceps same index as itself
    return diag_zeroes


def undirected(T):
    """
    Returns the undirected matrix made by OR of both directions of a given matrix T
    Args:
        T (torch.Tensor): The input matrix
    Returns:
        torch.Tensor: The undirected matrix made by OR of both directions of T
    """
    return(torch.bitwise_or(T,  torch.transpose(T)))            # Or a matrix with its transpose, giving matrix that = 1  if [i]->[j] or [j]->[i]

def refine_mask(tensor, mask, index, value, in_place = False):
    """
    Returns a mask, that is the union of mask and the submask where tensor[mask, index] == value
    Args:
        tensor (torch.Tensor): The input tensor
        mask (torch.Tensor): The input mask
        index (int): The index of the value to check
        value (int): The value to check for
        in_place (bool): Whether to modify the input mask in place
    Returns:
        torch.Tensor: Mask(size of input mask) with union of input mask and submask where tensor[mask, index] == value
    """
    submask = (tensor[mask, index] == value)
    return sub_union(mask, submask, in_place)                   # return new mask

def sub_union(mask, submask, in_place = False):
    """
    Returns a mask, that is the union of the input mask and its submask
    Args:
        mask (torch.Tensor): The input mask
        submask (torch.Tensor): The submask
        in_place (bool): Whether to modify the mask in place
    Returns:
        torch.Tensor: Mask(size of input mask) with union of input mask and submask
    """
    if not in_place:
        new_mask = mask.clone()
    else:
        new_mask = mask
    new_mask[mask] &= submask
    return new_mask
