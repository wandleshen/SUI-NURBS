import torch
import math


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
    uv1 = torch.cat([u1[col1[:, 0]][:, None], v1[col1[:, 1]][:, None]], dim=1).permute(
        0, 2, 1
    )
    uv2 = torch.cat([u2[col2[:, 0]][:, None], v2[col2[:, 1]][:, None]], dim=1).permute(
        0, 2, 1
    )
    return uv1[indices[:, 0].unique()], uv2[indices[:, 1].unique()]


def gen_grids(aabb, surf):
    diff = aabb[:, 1] - aabb[:, 0]
    res = torch.tensor([torch.min(diff[:, 0]), torch.min(diff[:, 1])]).cuda()
    grid_min = torch.floor(aabb[:, 0] / res)
    grid_max = torch.ceil(aabb[:, 1] / res)

    grids = torch.stack([grid_min, grid_max], dim=1).int()
    graph = torch.zeros(
        [
            math.ceil(max(surf.knotvector_u) // res[0]),
            math.ceil(max(surf.knotvector_v) // res[1]),
        ],
        dtype=bool,
    ).cuda()
    for grid in grids:
        graph[grid[0, 0] : grid[1, 0] + 1, grid[0, 1] : grid[1, 1] + 1] = True
    return grids[0], graph, res


def is_has_feature(aabb, graph):
    area = graph[aabb[0, 0] : aabb[1, 0] + 1, aabb[0, 1] : aabb[1, 1] + 1]
    return torch.any(area)


def sequence_joining(uv, surf, width):
    cluster, graph, res = gen_grids(uv, surf)
    clusters = []
    expand = torch.tensor([True, True, True, True]).cuda()  # Up, Left, Right, Down
    neighbors = None
    offset = torch.tensor(
        [[[0, -1], [0, 0]], [[-1, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]]
    ).cuda()
    while torch.any(graph):
        while True:
            neighbors = torch.tensor(
                [
                    [
                        [cluster[0, 0], cluster[0, 1] - 1],
                        [cluster[1, 0], cluster[0, 1] - 1],
                    ],  # Up
                    [
                        [cluster[0, 0] - 1, cluster[0, 1]],
                        [cluster[0, 0] - 1, cluster[1, 1]],
                    ],  # Left
                    [
                        [cluster[1, 0] + 1, cluster[0, 1]],
                        [cluster[1, 0] + 1, cluster[1, 1]],
                    ],  # Right
                    [
                        [cluster[0, 0], cluster[1, 1] + 1],
                        [cluster[1, 0], cluster[1, 1] + 1],
                    ],  # Down
                ]
            ).cuda()
            for i in range(4):
                if expand[i]:
                    expand[i] = is_has_feature(neighbors[i], graph)
            if (~expand[0] & ~expand[2]) | (~expand[1] & ~expand[3]):
                break
            for i in range(4):
                if expand[i]:
                    cluster += offset[i]
        clusters.append(cluster)
        graph[cluster[0, 0] : cluster[1, 0] + 1, cluster[0, 1] : cluster[1, 1] + 1] = (
            False
        )
        for i in range(4):
            if expand[i]:
                expand = torch.ones_like(expand)
                expand[3 - i] = False
                minx, miny, maxx, maxy = (
                    neighbors[i, 0, 0],
                    neighbors[i, 0, 1],
                    neighbors[i, 1, 0],
                    neighbors[i, 1, 1],
                )
                area = graph[minx : maxx + 1, miny : maxy + 1]
                pos = torch.nonzero(area)
                if pos.shape[0] > 0:
                    cluster = torch.tensor(
                        [
                            [pos[0, 0] + minx, pos[0, 1] + miny],
                            [pos[0, 0] + minx, pos[0, 1] + miny],
                        ],
                        dtype=int,
                    ).cuda()
                    break
                else:
                    pos = torch.nonzero(graph)
                    cluster = torch.tensor(
                        [
                            [pos[0, 0] + minx, pos[0, 1] + miny],
                            [pos[0, 0] + minx, pos[1, 0] + miny],
                        ],
                        dtype=int,
                    ).cuda()
                    expand[3 - i] = True
                    break
    centroids = torch.zeros([len(clusters), 2]).cuda()
    for i, cluster in enumerate(clusters):
        centroids[i] = cluster.float().mean(dim=0) * res
    return centroids


def point_to_surface(evalpts, points):
    _, indices = torch.min(
        torch.norm(evalpts.unsqueeze(0) - points.unsqueeze(1), dim=2), dim=1
    )
    return evalpts[indices]


def accuracy_improvement(pts, surf1, surf2, max_iter=20, threshold=1e-3):
    mask = torch.ones(pts.shape[0], dtype=bool).cuda()
    evalpts1 = torch.tensor(surf1.evalpts).cuda()
    evalpts2 = torch.tensor(surf2.evalpts).cuda()
    while torch.any(mask) and max_iter > 0:
        d1 = point_to_surface(evalpts1, pts[mask])
        d2 = point_to_surface(evalpts2, pts[mask])
        d1p = pts[mask] - d1
        d2p = pts[mask] - d2
        mask2 = (torch.norm(d1p, dim=1) > threshold) & (
            torch.norm(d2p, dim=1) > threshold
        )
        d1p, d2p, d1, d2 = d1p[mask2], d2p[mask2], d1[mask2], d2[mask2]
        normals = torch.cross(d1p, d2p, dim=1)
        A = torch.stack([d1p, d2p, normals], dim=1)
        b = torch.stack(
            [
                torch.sum(d1p * d1, dim=1),
                torch.sum(d2p * d2, dim=1),
                torch.zeros_like(d1p[:, 0]),
            ],
            dim=1,
        )
        x = torch.linalg.solve(A, b)
        pts[mask][mask2] = x
        mask = (torch.norm(x - pts, dim=1)) > threshold
        max_iter -= 1
    return pts


def gen_curves(u1, v1, col1, surf1, u2, v2, col2, surf2):
    uv1, uv2 = strip_thinning(u1, v1, col1, surf1, u2, v2, col2, surf2)
    pts = sequence_joining(uv1, surf1, 0.0)
