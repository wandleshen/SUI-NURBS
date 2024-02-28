import torch

# TODO: Get the min and max value in three axes
def gen_AABB(knotvector_u, knotvector_v, m, n, p):
    u, i = gen_intervals(knotvector_u, m, p)
    if u is None:
        return
    print(basic_func(u, p, i, knotvector_u))

def gen_intervals(knotvector, n, p):
    if n < len(knotvector)-2*p:
        print('[ERROR] Too few sampled intervals for the given knotvector')
        return
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
        splits = torch.zeros(n_intervals, dtype=torch.int32).cuda()
        splits[:mod] = div + 2
        splits[mod:] = div + 1
        i = torch.repeat_interleave(i, splits)
        intervals = torch.repeat_interleave(intervals, splits)
        splits = torch.repeat_interleave(splits, splits)
        intervals /= splits
    # Generate the cumulative intervals
    cumsum = torch.zeros(intervals.shape[0]+1, dtype=torch.float32).cuda()
    cumsum[1:] = torch.cumsum(intervals, dim=0)
    orig = torch.stack((cumsum[:-1], cumsum[1:]), dim=1)
    # Generate all `i`s and `u`s for N_{i,p}
    range_t = torch.arange(-p, 1, 1).cuda()
    i = i[:, None] + range_t[None, :]
    u = torch.zeros_like(orig, dtype=torch.float32).cuda()
    u[:, 0] = (orig[:, 1] + orig[:, 0]) / 2.
    u[:, 1] = (orig[:, 1] - orig[:, 0]) / 2.
    return u, i

def basic_func(u, p, i, knotvector):
    '''Function
    Generate the basic function N_{i,p} at u
    '''
    n = u.shape[0]
    m = i.shape[1]
    N = torch.zeros([n, m, 3], dtype=torch.float32).cuda()
    if p == 0:
        mask = (knotvector[i] <= u[:, None, 0]) & (u[:, None, 0] < knotvector[i+1])
        N[..., 0] = torch.where(mask, torch.ones_like(N[..., 0]), torch.zeros_like(N[..., 0]))
        return N
    a = torch.zeros([n, m, 3], dtype=torch.float32).cuda()
    b = torch.zeros([n, m, 3], dtype=torch.float32).cuda()

    divisor_ip = knotvector[i+p] - knotvector[i]
    mask = torch.abs(divisor_ip) > 1e-6
    a[..., 0] = torch.where(mask, (u[:, None, 0] - knotvector[i]), torch.zeros_like(a[..., 0]))
    a[..., 1] = torch.where(mask, u[:, None, 1], torch.zeros_like(a[..., 1]))
    N[mask] += AA_times(a[mask], basic_func(u, p-1, i, knotvector)[mask]) / divisor_ip[mask][...,None]

    divisor_ip1i1 = knotvector[i+p+1] - knotvector[i+1]
    mask = torch.abs(divisor_ip1i1) > 1e-6
    b[..., 0] = torch.where(mask, (knotvector[i+p+1] - u[:, None, 0]), torch.zeros_like(b[..., 0]))
    b[..., 1] = torch.where(mask, u[:, None, 1], torch.zeros_like(b[..., 1]))
    N[mask] += AA_times(b[mask], basic_func(u, p-1, i+1, knotvector)[mask]) / divisor_ip1i1[mask][...,None]

    return N

def AA_times(lhs, rhs):
    res = torch.zeros_like(lhs).cuda()
    abs_lhs = torch.abs(lhs)
    abs_rhs = torch.abs(rhs)
    res[..., 0] = lhs[..., 0] * rhs[..., 0]
    res[..., 1] = lhs[..., 1] * rhs[..., 0] + lhs[..., 0] * rhs[..., 1]
    res[..., 2] = (abs_lhs[..., 1] + abs_lhs[..., 2]) * (abs_rhs[..., 1] + abs_rhs[..., 2])
    return res