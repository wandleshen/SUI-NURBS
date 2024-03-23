import torch
import torch.nn.functional as F


def get_bounding_aabb(aabbs):
    """
    Compute the minimum and maximum points of the AABBs.

    Parameters:
    aabbs (torch.Tensor): A tensor of AABBs.

    Returns:
    tuple: A tuple containing the minimum and maximum points of the AABBs.
    """
    min_points = torch.min(torch.min(aabbs[..., 0, :], dim=-3)[0], dim=-2)[0]
    max_points = torch.max(torch.max(aabbs[..., 1, :], dim=-3)[0], dim=-2)[0]
    return min_points, max_points


def overlap_test(lhs, rhs):
    """
    Check overlaps for all combinations of AABBs.
    All three dimensions must overlap for the AABBs to overlap.

    Parameters:
    lhs (torch.Tensor): A tensor of AABBs.
    rhs (torch.Tensor): A tensor of AABBs.

    Returns:
    torch.Tensor: A tensor indicating which AABBs overlap.
    """
    min_lhs, max_lhs = get_bounding_aabb(lhs)
    min_rhs, max_rhs = get_bounding_aabb(rhs)

    # Check overlaps for all combinations of AABBs
    overlaps = (max_lhs[:, :, None, None] >= min_rhs[None, None]) & (
        min_lhs[:, :, None, None] <= max_rhs[None, None]
    )

    # All three dimensions must overlap for the AABBs to overlap
    overlaps = overlaps.all(dim=-1)

    return overlaps


def pad_and_split(tensor, a, b):
    """
    Pad the tensor and split it along the first and second axis.

    Parameters:
    tensor (torch.Tensor): The tensor to be padded and split.
    a (int): The number of parts to split the first axis into.
    b (int): The number of parts to split the second axis into.

    Returns:
    torch.Tensor: The padded and split tensor.
    """
    # Calculate padding sizes
    pad_size_a = ((a - tensor.shape[0] % a) % a).item()
    pad_size_b = ((b - tensor.shape[1] % b) % b).item()

    if tensor.shape[0] < a:
        a = tensor.shape[0]
        pad_size_a = 0
    if tensor.shape[1] < b:
        b = tensor.shape[1]
        pad_size_b = 0

    # Pad the tensor
    # Since `F.pad` doesn't support padding the first dimension w/ `replicate` mode
    tensor = tensor.permute(3, 2, 1, 0)
    tensor = F.pad(tensor, (0, int(pad_size_b), 0, int(pad_size_a)), mode="replicate")
    tensor = tensor.permute(3, 2, 1, 0)

    # Split the tensor along the first axis
    split1 = torch.split(tensor, tensor.shape[0] // a, dim=0)
    # Split each resulting tensor along the second axis
    split2 = [torch.split(t, t.shape[1] // b, dim=1) for t in split1]
    # Stack the resulting tensors along new axes
    result = torch.stack([torch.stack(t, dim=0) for t in split2], dim=0)

    return result, int(tensor.shape[0] // a), int(tensor.shape[1] // b)


def decompose_aabb(aabb1, aabb2, kl, orig_indices):
    """
    Decompose the AABBs and compute the overlaps.

    Parameters:
    aabb1 (torch.Tensor): The first tensor of AABBs.
    aabb2 (torch.Tensor): The second tensor of AABBs.
    kl (torch.Tensor): The list of parts to split the first and second axis into.
    orig_indices (torch.Tensor): The original indices of the AABBs.

    Returns:
    torch.Tensor: A tensor containing the indices of the overlapping AABBs.
    """
    pad1, i, j = pad_and_split(aabb1, kl[0], kl[1])  # [k1, l1, m/k1, n/l1, 2, 3]
    pad2, k, l = pad_and_split(aabb2, kl[2], kl[3])  # [k2, l2, m/k2, n/l2, 2, 3]

    overlaps = overlap_test(pad1, pad2)  # [k1, l1, k2, l2]
    length = torch.tensor([i, j, k, l], device=torch.device("cuda"))
    indices = torch.nonzero(overlaps) * length + orig_indices
    return torch.stack([indices, indices + length], dim=-1)


def region_extraction(aabb1, aabb2, d=4):
    """
    Extract regions from two surfaces.

    Parameters:
    aabb1 (torch.Tensor): The first tensor of AABBs.
    aabb2 (torch.Tensor): The second tensor of AABBs.
    d (int, optional): The maximum number of splits to perform for each iteration. Defaults to 4.

    Returns:
    tuple: A tuple containing two tensors. The first tensor contains the unique regions extracted from the first AABB.
           The second tensor contains the unique regions extracted from the second AABB.
    """
    mn = torch.ceil(
        torch.log2(
            torch.cat(
                [
                    torch.tensor(aabb1.shape[:2], device=torch.device("cuda")),
                    torch.tensor(aabb2.shape[:2], device=torch.device("cuda")),
                ]
            )
        )
    )
    kl = torch.where(mn % d != 0, mn % d, d)
    mn -= kl
    Rols = decompose_aabb(
        aabb1, aabb2, 2**kl, torch.tensor([0, 0, 0, 0], device=torch.device("cuda"))
    )
    while torch.sum(mn) > 0:
        kl = torch.clamp_max(mn, d)
        mn -= kl
        pieces = 2**kl
        loc_Rol = None
        for r in Rols:
            Rol = decompose_aabb(
                aabb1[r[0, 0].item() : r[0, 1].item(), r[1, 0].item() : r[1, 1].item()],
                aabb2[r[2, 0].item() : r[2, 1].item(), r[3, 0].item() : r[3, 1].item()],
                pieces,
                r[:, 0],
            )
            if loc_Rol is None:
                loc_Rol = Rol
            else:
                loc_Rol = torch.cat([loc_Rol, Rol], dim=0)
        Rols = loc_Rol
    return Rols[..., 0:2, 0].unique(dim=0), Rols[..., 2:4, 0].unique(dim=0)
