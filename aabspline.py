import torch

def gen_aabb(knotvector_u, knotvector_v, ctrlpts, m, n, p, q, delta=1., eps=.01):
    # Get all basic_funcs
    u, i = gen_intervals(knotvector_u, m, p, eps)
    Ni = basic_func(u, p, i, knotvector_u) # [m, p+1, 3]
    v, j = gen_intervals(knotvector_v, n, q, eps)
    Nj = basic_func(v, q, j, knotvector_v) # [n, q+1, 3]

    # Cal `P({\hat u},{\hat v})`
    # i: [m, p+1] j: [n, q+1]
    i_expand = i[:, None, :, None].expand(-1, n, -1, q+1) # [m, n, p+1, q+1]
    j_expand = j[None, :, None, :].expand(m, -1, p+1, -1) # [m, n, p+1, q+1]
    Pij = ctrlpts[i_expand, j_expand] # [m, n, p+1, q+1, 3]
    Ni_expand = Ni[:, None, :, None, :].expand(-1, n, -1, q+1, -1) # [m, n, p+1, q+1, 3]
    Nj_expand = Nj[None, :, None, :, :].expand(m, -1, p+1, -1, -1) # [m, n, p+1, q+1, 3]
    NiNj = aa_times(Ni_expand, Nj_expand) # [m, n, p+1, q+1, 3]

    # N_i({\hat u})N_j({\hat v})P_{i,j}
    NiNjP = torch.einsum('ijklm,ijkln->ijnm', NiNj, Pij) # [m, n, 3, 3]
    aabb = torch.zeros([m, n, 2, 3])
    aabb[..., 0, :] = NiNjP[..., 0] - torch.abs(NiNjP[..., 1:3]).sum(dim=-1) * delta
    aabb[..., 1, :] = NiNjP[..., 0] + torch.abs(NiNjP[..., 1:3]).sum(dim=-1) * delta
    return aabb.reshape(-1,2,3)

def gen_intervals(knotvector, n, p, eps=.01):
    if n < len(knotvector)-2*p:
        print('[ERROR] Too few sampled intervals for the given knotvector')
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
        splits = torch.ones(n_intervals, dtype=torch.int32).cuda()
        splits += div
        _, indices = torch.topk(intervals, mod)
        splits[indices] += 1
        i = torch.repeat_interleave(i, splits)
        intervals = torch.repeat_interleave(intervals, splits)
        splits = torch.repeat_interleave(splits, splits)
        intervals /= splits
    # Generate the cumulative intervals
    cumsum = torch.zeros(intervals.shape[0]+1, dtype=torch.float32).cuda()
    cumsum[1:] = torch.cumsum(intervals, dim=0)
    cumsum[0] -= eps
    cumsum[-1] += eps
    orig = torch.stack((cumsum[:-1], cumsum[1:]), dim=1)
    # Generate all `i`s and `u`s for N_{i,p}
    range_t = torch.arange(-p, 1, 1).cuda()
    i = i[:, None] + range_t[None, :]
    u = torch.zeros_like(orig, dtype=torch.float32).cuda()
    u[:, 0] = (orig[:, 1] + orig[:, 0]) / 2.
    u[:, 1] = (orig[:, 1] - orig[:, 0]) / 2.
    return u, i

def basic_func(u, p, i, knotvector):
    n = u.shape[0]
    m = i.shape[1]
    N = torch.zeros([n, m, 3], dtype=torch.float32).cuda()
    if p == 0:
        mask = (knotvector[i] <= u[:, None, 0]) & (u[:, None, 0] < knotvector[i+1])
        N[..., 0] = torch.where(mask, torch.ones_like(N[..., 0]), torch.zeros_like(N[..., 0]))
        return N
    a = torch.zeros([n, m, 3], dtype=torch.float32).cuda()
    b = torch.zeros([n, m, 3], dtype=torch.float32).cuda()
    # Cal the first part of N_{i,p}
    divisor_ip = knotvector[i+p] - knotvector[i]
    mask = torch.abs(divisor_ip) > 1e-6
    a[..., 0] = torch.where(mask, (u[:, None, 0] - knotvector[i]), torch.zeros_like(a[..., 0]))
    a[..., 1] = torch.where(mask, u[:, None, 1], torch.zeros_like(a[..., 1]))
    N[mask] += aa_times(a[mask], basic_func(u, p-1, i, knotvector)[mask]) / divisor_ip[mask][...,None]
    # Cal the second part of N_{i,p}
    divisor_ip1i1 = knotvector[i+p+1] - knotvector[i+1]
    mask = torch.abs(divisor_ip1i1) > 1e-6
    b[..., 0] = torch.where(mask, (knotvector[i+p+1] - u[:, None, 0]), torch.zeros_like(b[..., 0]))
    b[..., 1] = torch.where(mask, -u[:, None, 1], torch.zeros_like(b[..., 1]))
    N[mask] += aa_times(b[mask], basic_func(u, p-1, i+1, knotvector)[mask]) / divisor_ip1i1[mask][...,None]

    return N

def aa_times(lhs, rhs):
    res = torch.zeros_like(lhs).cuda()
    abs_lhs = torch.abs(lhs)
    abs_rhs = torch.abs(rhs)
    res[..., 0] = lhs[..., 0] * rhs[..., 0]
    res[..., 1] = lhs[..., 1] * rhs[..., 0] + lhs[..., 0] * rhs[..., 1]
    # res[..., 2] = (abs_lhs[..., 1] + abs_lhs[..., 2]) * (abs_rhs[..., 1] + abs_rhs[..., 2])
    res[..., 2] = (lhs[..., 1] + lhs[..., 2]) * (rhs[..., 1] + rhs[..., 2])
    return res