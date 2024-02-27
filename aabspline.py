import torch

def test(knotvector):
    print(basic_func(torch.tensor([[.5, .02, 0.],[.5, .02, 0.]]).cuda(), 3, torch.tensor([[8, 9, 10, 11],[8, 9, 10, 11]]).cuda(), knotvector))

# TODO: Get proper `i` for every point
#       Cal two direction's basic function at the same time
#       Get the min and max value in three axes
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

    divisor_ip1i1 = knotvector[i+p+1] - knotvector[i+1]
    mask = torch.abs(divisor_ip1i1) > 1e-6
    b[..., 0] = torch.where(mask, (knotvector[i+p+1] - u[:, None, 0]), torch.zeros_like(b[..., 0]))
    b[..., 1] = torch.where(mask, u[:, None, 1], torch.zeros_like(b[..., 1]))

    N += AA_times(a, basic_func(u, p-1, i, knotvector)) / divisor_ip[...,None]
    N += AA_times(b, basic_func(u, p-1, i+1, knotvector)) / divisor_ip1i1[...,None]

    return N

def AA_times(lhs, rhs):
    res = torch.zeros_like(lhs).cuda()
    abs_lhs = torch.abs(lhs)
    abs_rhs = torch.abs(rhs)
    res[..., 0] = lhs[..., 0] * rhs[..., 0]
    res[..., 1] = lhs[..., 1] * rhs[..., 0] + lhs[..., 0] * rhs[..., 1]
    res[..., 2] = (abs_lhs[..., 1] + abs_lhs[..., 2]) * (abs_rhs[..., 1] + abs_rhs[..., 2])
    return res