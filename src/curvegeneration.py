import torch


def cartesian_product_rowwise(tensor_a, tensor_b):
    n, _ = tensor_a.shape

    # 广播tensor_a和tensor_b以形成所有组合
    tensor_a_expanded = tensor_a.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
    tensor_b_expanded = tensor_b.unsqueeze(2).expand(-1, -1, 2).reshape(-1, 4)

    # 这样，我们得到每一对行的笛卡尔积
    cartesian_rowwise = torch.zeros([n, 4, 2]).cuda()
    for i in range(4):
        cartesian_rowwise[:, i] = torch.stack(
            [tensor_a_expanded[..., i], tensor_b_expanded[..., i]], dim=1
        )

    return cartesian_rowwise


def cal_min_aabbs(u, v, col, surf):
    pts = cartesian_product_rowwise(u[col[:, 0]], v[col[:, 1]])
    pts = (
        torch.tensor(surf.evaluate_list(pts.reshape(-1, 2).tolist()))
        .reshape(-1, 4, 3)
        .cuda()
    )
    min_vals, _ = torch.min(pts, dim=1, keepdim=True)
    max_vals, _ = torch.max(pts, dim=1, keepdim=True)

    return torch.cat([min_vals, max_vals], dim=1)


def strip_thinning(u1, v1, col1, surf1, u2, v2, col2, surf2):
    aabb1 = cal_min_aabbs(u1, v1, col1, surf1)  # [n, 2, 3]
    aabb2 = cal_min_aabbs(u2, v2, col2, surf2)  # [m, 2, 3]

    n = aabb1.shape[0]
    m = aabb2.shape[0]

    aabb1_expanded = aabb1.unsqueeze(1).expand(-1, m, -1, -1)
    aabb2_expanded = aabb2.unsqueeze(0).expand(n, -1, -1, -1)
    max_aabb1, min_aabb1 = aabb1_expanded[:, :, 1], aabb1_expanded[:, :, 0]
    max_aabb2, min_aabb2 = aabb2_expanded[:, :, 1], aabb2_expanded[:, :, 0]

    overlaps = (min_aabb1 <= max_aabb2) & (min_aabb2 <= max_aabb1)
    overlaps = overlaps.all(dim=-1)

    indices = torch.nonzero(overlaps)
    return aabb1[indices[:, 0]].cpu().numpy(), aabb2[indices[:, 1]].cpu().numpy()
