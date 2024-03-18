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

    overlaps = (min_aabb1 <= max_aabb2) & (max_aabb1 >= min_aabb2)
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
        graph[grid[0, 0] : grid[1, 0], grid[0, 1] : grid[1, 1]] = True
    return graph, res


def has_feature(aabb, graph):
    # 判断 aabb 是否在 graph 的范围内
    if (
        aabb[0, 0] < 0
        or aabb[0, 1] < 0
        or aabb[1, 0] > graph.shape[0]
        or aabb[1, 1] > graph.shape[1]
    ):
        return False
    else:
        area = graph[aabb[0, 0] : aabb[1, 0], aabb[0, 1] : aabb[1, 1]]
        return torch.any(area)


def sequence_joining(uv, surf):
    graph, res = gen_grids(uv, surf)
    pos = torch.nonzero(graph)
    cluster = torch.tensor(
        [
            [pos[0, 0], pos[0, 1]],
            [pos[0, 0] + 1, pos[0, 1] + 1],
        ],
        dtype=int,
    ).cuda()
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
                        [cluster[1, 0], cluster[0, 1]],
                    ],  # Up
                    [
                        [cluster[0, 0] - 1, cluster[0, 1]],
                        [cluster[0, 0], cluster[1, 1]],
                    ],  # Left
                    [
                        [cluster[1, 0], cluster[0, 1]],
                        [cluster[1, 0] + 1, cluster[1, 1]],
                    ],  # Right
                    [
                        [cluster[0, 0], cluster[1, 1]],
                        [cluster[1, 0], cluster[1, 1] + 1],
                    ],  # Down
                ]
            ).cuda()
            for i in range(4):
                if expand[i]:
                    expand[i] = has_feature(neighbors[i], graph)
            if (~expand[0] & ~expand[3]) | (~expand[1] & ~expand[2]):
                break
            for i in range(4):
                if expand[i]:
                    cluster += offset[i]
        clusters.append(cluster)
        graph[cluster[0, 0] : cluster[1, 0], cluster[0, 1] : cluster[1, 1]] = False
        for i in range(4):
            if expand[i]:
                minx, miny, maxx, maxy = (
                    neighbors[i, 0, 0],
                    neighbors[i, 0, 1],
                    neighbors[i, 1, 0],
                    neighbors[i, 1, 1],
                )
                area = graph[minx:maxx, miny:maxy]
                pos = torch.nonzero(area)
                expand = torch.ones_like(expand)
                if pos.shape[0] > 0:
                    # Choose midpoint in pos instead of the first point
                    midpos = pos[pos.shape[0] // 2]
                    cluster = torch.tensor(
                        [
                            [midpos[0] + minx, midpos[1] + miny],
                            [midpos[0] + minx + 1, midpos[1] + miny + 1],
                        ],
                        dtype=int,
                    ).cuda()
                    expand[3 - i] = False
                else:
                    pos = torch.nonzero(graph)
                    cluster = torch.tensor(
                        [
                            [pos[0, 0] + minx, pos[0, 1] + miny],
                            [pos[0, 0] + minx + 1, pos[1, 0] + miny + 1],
                        ],
                        dtype=int,
                    ).cuda()
                break
    centroids = torch.zeros([len(clusters), 2, 2]).cuda()
    for i, cluster in enumerate(clusters):
        centroids[i] = cluster.float() * res
    return centroids, torch.mean(centroids, dim=1)


def point_to_surface(evalpts, points):
    _, indices = torch.min(
        torch.norm(evalpts.unsqueeze(0) - points.unsqueeze(1), dim=2), dim=1
    )
    return evalpts[indices]


def accuracy_improvement(pts, surf1, surf2, max_iter=40, threshold=1e-6):
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
        mask[mask.clone()][mask2] = (
            torch.norm(x - pts[mask][mask2], dim=1)
        ) > threshold
        max_iter -= 1
    return pts


def find_closest_points_and_midpoint(pts1, pts2):
    dists = torch.norm(
        pts1.unsqueeze(0) - pts2.unsqueeze(1), dim=2
    )  # [n, m] distance matrix
    indices = torch.argmin(dists, dim=1)
    closest_pts = pts2[indices]
    midpoints = (pts1 + closest_pts) / 2
    return midpoints


def gen_curves(u1, v1, col1, surf1, u2, v2, col2, surf2):
    uv1, uv2 = strip_thinning(u1, v1, col1, surf1, u2, v2, col2, surf2)
    aabb1, pts1 = sequence_joining(uv1, surf1)
    aabb2, pts2 = sequence_joining(uv2, surf2)
    pts3d1 = torch.tensor(surf1.evaluate_list(pts1.cpu().tolist())).cuda()
    pts3d2 = torch.tensor(surf2.evaluate_list(pts2.cpu().tolist())).cuda()
    midpoints = find_closest_points_and_midpoint(pts3d1, pts3d2)
    pts = accuracy_improvement(midpoints, surf1, surf2)

    # Gen imgui AABBs
    uv3d1 = torch.tensor(
        surf1.evaluate_list(uv1.reshape(-1, 2).cpu().tolist())
    ).reshape(-1, 2, 3)
    uv3d2 = torch.tensor(
        surf2.evaluate_list(uv2.reshape(-1, 2).cpu().tolist())
    ).reshape(-1, 2, 3)
    aabb3d1 = torch.tensor(
        surf1.evaluate_list(aabb1.reshape(-1, 2).cpu().tolist())
    ).reshape(-1, 2, 3)
    aabb3d2 = torch.tensor(
        surf2.evaluate_list(aabb2.reshape(-1, 2).cpu().tolist())
    ).reshape(-1, 2, 3)
    return uv3d1, uv3d2, aabb3d1, aabb3d2, pts
