import torch


def gen_aabb(
    knotvector_u, knotvector_v, ctrlpts, m, n, p, q, delta=1.0, eps=0.01, scaler=100.0
):
    """
    Generate Axis-Aligned Bounding Boxes (AABBs) for a given NURBS surface

    Parameters:
    knotvector_u (torch.Tensor): Knot vector in the U direction
    knotvector_v (torch.Tensor): Knot vector in the V direction
    ctrlpts (torch.Tensor): Control points
    m, n (int): Number of AABBs in U or V direction
    p, q (int): Degree of the surface in U or V direction
    delta (float): Amount to expand the bounding box
    eps (float): Value for the edge of surface's bounding boxes expansion
    scaler (float): Scaler for the knot vectors

    Returns:
    aabb (torch.Tensor): Axis-Aligned Bounding Boxes
    u, v (torch,Tensor): Chosen intervals along the U or V direction
    """

    # Get all basic functions
    knotvector_u *= scaler
    knotvector_v *= scaler
    u, i = gen_intervals(knotvector_u, m, p, eps)
    Ni = basic_func(u, p, i, knotvector_u)  # [m, p+1, 3]
    v, j = gen_intervals(knotvector_v, n, q, eps)
    Nj = basic_func(v, q, j, knotvector_v)  # [n, q+1, 3]

    u[:, 1] *= delta
    v[:, 1] *= delta
    u = torch.stack([(u[:, 0] - u[:, 1]), (u[:, 0] + u[:, 1])], dim=1)
    v = torch.stack([(v[:, 0] - v[:, 1]), (v[:, 0] + v[:, 1])], dim=1)
    u = torch.clamp(u, knotvector_u[0], knotvector_u[-1])
    v = torch.clamp(v, knotvector_v[0], knotvector_v[-1])

    # Calculate `P({\hat u},{\hat v})`
    i_expand = i[:, None, :, None].expand(-1, n, -1, q + 1)  # [m, n, p+1, q+1]
    j_expand = j[None, :, None, :].expand(m, -1, p + 1, -1)  # [m, n, p+1, q+1]
    Pij = ctrlpts[i_expand, j_expand]  # [m, n, p+1, q+1, 4]
    Ni_expand = Ni[:, None, :, None, :].expand(
        -1, n, -1, q + 1, -1
    )  # [m, n, p+1, q+1, 3]
    Nj_expand = Nj[None, :, None, :, :].expand(
        m, -1, p + 1, -1, -1
    )  # [m, n, p+1, q+1, 3]
    NiNj = aa_times(Ni_expand, Nj_expand, True)  # [m, n, p+1, q+1, 3]

    # Calculate N_i({\hat u})N_j({\hat v})P_{i,j}
    NiNjP = torch.einsum("ijklm,ijkln->ijnm", NiNj, Pij)  # [m, n, 4, 3]
    NiNjP, weight = (
        NiNjP[..., 0:3, :],
        NiNjP[..., 3, :],
    )  # Discard NURBS weight -> [m, n, 3, 3], [m, n, 3]
    inverse_weight = torch.zeros_like(weight)
    inverse_weight[..., 0] = 1.0 / weight[..., 0]
    inverse_weight[..., 1] = -weight[..., 1] / (weight[..., 0] ** 2)
    inverse_weight[..., 2] = -weight[..., 2] / (weight[..., 0] ** 2)
    NiNjP = aa_times(
        NiNjP, inverse_weight[:, :, None, :].expand(-1, -1, 3, -1), True
    )  # [m, n, 3, 3]

    # Calculate AABB
    aabb = torch.zeros([m, n, 2, 3], device=torch.device("cuda"))
    aabb[..., 0, :] = NiNjP[..., 0] - torch.abs(NiNjP[..., 1:3]).sum(dim=-1)
    aabb[..., 1, :] = NiNjP[..., 0] + torch.abs(NiNjP[..., 1:3]).sum(dim=-1)
    return aabb, u, v


def gen_intervals(knotvector, n, p, eps=0.01):
    """
    Generate intervals for a given knot vector.

    Parameters:
    knotvector (torch.Tensor): The knot vector
    n (int): Number of intervals
    p (int): Degree of the B-spline basis function
    eps (float): Value for the edge of surface's bounding boxes expansion

    Returns:
    u (torch.Tensor): The affine arithmetics values for each interval
    i (torch.Tensor): The index of the left knot of each interval
    """
    if n < len(knotvector) - 2 * p:
        print("[ERROR] Too few sampled intervals for the given knotvector")
        raise Exception()
    # Generate intervals
    diff = knotvector[1:] - knotvector[:-1]
    # Start of nonzero intervals
    non_zeros = torch.nonzero(diff).flatten()
    i = non_zeros
    intervals = diff[non_zeros]
    # Split the intervals
    n_intervals = intervals.shape[0]
    if n > n_intervals:
        split = n - n_intervals
        div = split // n_intervals
        mod = split % n_intervals
        splits = torch.ones(n_intervals, dtype=torch.int32, device=torch.device("cuda"))
        splits += div
        _, indices = torch.topk(intervals, mod)
        splits[indices] += 1
        i = torch.repeat_interleave(i, splits)
        intervals = torch.repeat_interleave(intervals, splits)
        splits = torch.repeat_interleave(splits, splits)
        intervals /= splits
    # Generate the cumulative intervals
    cumsum = torch.zeros(
        intervals.shape[0] + 1, dtype=torch.float32, device=torch.device("cuda")
    )
    cumsum[1:] = torch.cumsum(intervals, dim=0)
    cumsum[0] -= eps
    cumsum[-1] += eps
    orig = torch.stack((cumsum[:-1], cumsum[1:]), dim=1)
    # Generate all `i`s and `u`s for N_{i,p}
    range_t = torch.arange(-p, 1, 1, device=torch.device("cuda"))
    i = i[:, None] + range_t[None, :]
    u = torch.zeros_like(orig, dtype=torch.float32, device=torch.device("cuda"))
    u[:, 0] = (orig[:, 1] + orig[:, 0]) / 2.0
    u[:, 1] = (orig[:, 1] - orig[:, 0]) / 2.0
    return u, i


def basic_func(u, p, i, knotvector):
    """
    Calculate the B-spline basis function N_{i,p}(u).

    Parameters:
    u (torch.Tensor): The affine arithmetics values
    p (int): The degree of the B-spline basis function
    i (torch.Tensor): The index of the left knot
    knotvector (torch.Tensor): The knot vector

    Returns:
    N (torch.Tensor): The B-spline basis function values
    """
    n = u.shape[0]
    m = i.shape[1]
    N = torch.zeros([n, m, 3], dtype=torch.float32, device=torch.device("cuda"))
    if p == 0:
        mask = (knotvector[i] <= u[:, None, 0]) & (u[:, None, 0] < knotvector[i + 1])
        N[..., 0] = torch.where(
            mask, torch.ones_like(N[..., 0]), torch.zeros_like(N[..., 0])
        )
        return N
    a = torch.zeros([n, m, 3], dtype=torch.float32, device=torch.device("cuda"))
    b = torch.zeros([n, m, 3], dtype=torch.float32, device=torch.device("cuda"))
    # Cal the first part of N_{i,p}
    divisor_ip = knotvector[i + p] - knotvector[i]
    mask = torch.abs(divisor_ip) > 1e-6
    a[..., 0] = torch.where(
        mask, (u[:, None, 0] - knotvector[i]), torch.zeros_like(a[..., 0])
    )
    a[..., 1] = torch.where(mask, u[:, None, 1], torch.zeros_like(a[..., 1]))
    N[mask] += (
        aa_times(a[mask], basic_func(u, p - 1, i, knotvector)[mask])
        / divisor_ip[mask][..., None]
    )
    # Cal the second part of N_{i,p}
    divisor_ip1i1 = knotvector[i + p + 1] - knotvector[i + 1]
    mask = torch.abs(divisor_ip1i1) > 1e-6
    b[..., 0] = torch.where(
        mask, (knotvector[i + p + 1] - u[:, None, 0]), torch.zeros_like(b[..., 0])
    )
    b[..., 1] = torch.where(mask, -u[:, None, 1], torch.zeros_like(b[..., 1]))
    N[mask] += (
        aa_times(b[mask], basic_func(u, p - 1, i + 1, knotvector)[mask])
        / divisor_ip1i1[mask][..., None]
    )

    return N


def aa_times(lhs, rhs, is3x3=False):
    """
    Perform affine arithmetic operations.

    Parameters:
    lhs (torch.Tensor): Left-hand side affine form
    rhs (torch.Tensor): Right-hand side affine form
    is3x3 (bool): Flag to indicate if the operation is for values w/ 2 noises

    Returns:
    res (torch.Tensor): The result of the affine arithmetic operation
    """
    res = torch.zeros_like(lhs, device=torch.device("cuda"))
    # abs_lhs = torch.abs(lhs)
    # abs_rhs = torch.abs(rhs)
    res[..., 0] = lhs[..., 0] * rhs[..., 0]
    res[..., 1] = lhs[..., 1] * rhs[..., 0] + lhs[..., 0] * rhs[..., 1]
    # res[..., 2] = (abs_lhs[..., 1] + abs_lhs[..., 2]) * (abs_rhs[..., 1] + abs_rhs[..., 2])
    if is3x3:
        res[..., 2] = lhs[..., 2] * rhs[..., 0] + lhs[..., 0] * rhs[..., 2]
    else:
        res[..., 2] = (lhs[..., 1] + lhs[..., 2]) * (rhs[..., 1] + rhs[..., 2])
    return res
