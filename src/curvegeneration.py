import torch
import math


def cartesian_product_rowwise(tensor_a, tensor_b):
    """
    Compute the Cartesian product of two tensors row-wise.

    Parameters:
    tensor_a, tensor_b (torch.Tensor): Input tensors.

    Returns:
    cartesian_rowwise (torch.Tensor): The Cartesian product of the input tensors row-wise.
    """
    n, _ = tensor_a.shape

    # Broadcast tensor_a and tensor_b to form all combinations
    tensor_a_expanded = tensor_a.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
    tensor_b_expanded = tensor_b.unsqueeze(2).expand(-1, -1, 2).reshape(-1, 4)

    # This way, we get the Cartesian product of each pair of rows
    cartesian_rowwise = torch.zeros([n, 4, 2], device=torch.device("cuda"))
    for i in range(4):
        cartesian_rowwise[:, i] = torch.stack(
            [tensor_a_expanded[..., i], tensor_b_expanded[..., i]], dim=1
        )

    return cartesian_rowwise


def cal_min_aabbs(u, v, col, surf, scaler=100.0):
    """
    Calculate the minimum Axis-Aligned Bounding Boxes (AABBs) for a given NURBS surface.

    Parameters:
    u, v (torch.Tensor): Knot vectors in the U and V direction.
    col (torch.Tensor): Indices to extract.
    surf (object): A NURBS surface object.
    scaler (float): Scaler for the knot vectors. Default is 100.0.

    Returns:
    aabb (torch.Tensor): The minimum AABBs for the surface.
    """
    pts = cartesian_product_rowwise(u[col[:, 0]], v[col[:, 1]]) / scaler
    pts = torch.tensor(
        surf.evaluate_list(pts.reshape(-1, 2).tolist()), device=torch.device("cuda")
    ).reshape(-1, 4, 3)
    min_vals, _ = torch.min(pts, dim=1, keepdim=True)
    max_vals, _ = torch.max(pts, dim=1, keepdim=True)

    return torch.cat([min_vals, max_vals], dim=1)


def strip_thinning(u1, v1, col1, surf1, u2, v2, col2, surf2, scaler1, scaler2):
    """
    Perform strip thinning on two NURBS surfaces.

    Parameters:
    u1, v1, u2, v2 (torch.Tensor): Knot vectors in the U and V direction for the two surfaces.
    col1, col2 (torch.Tensor): Indices to extract for the two surfaces.
    surf1, surf2 (object): NURBS surface objects.
    scaler1, scaler2 (float): Scalers for the knot vectors.

    Returns:
    uv1, uv2 (torch.Tensor): The thinned AABBs in the U and V direction for the two surfaces.
    """
    aabb1 = cal_min_aabbs(u1, v1, col1, surf1, scaler1)  # [n, 2, 3]
    aabb2 = cal_min_aabbs(u2, v2, col2, surf2, scaler2)  # [m, 2, 3]

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


def gen_grids(aabb, surf, scaler=100.0):
    """
    Generate grids for a given NURBS surface.

    Parameters:
    aabb (torch.Tensor): A tensor of AABBs.
    surf (object): A NURBS surface object.
    scaler (float): Scaler for the knot vectors. Default is 100.0.

    Returns:
    graph, res (torch.Tensor): The generated grids and their resolution.
    """
    # Calculate the grid resolution
    diff = aabb[:, 1] - aabb[:, 0]
    res = torch.tensor(
        [torch.min(diff[:, 0]), torch.min(diff[:, 1])], device=torch.device("cuda")
    )

    # Calculate the grid bounds
    grid_min = torch.floor(aabb[:, 0] / res)
    grid_max = torch.ceil(aabb[:, 1] / res)

    # Generate the grids
    grids = torch.stack([grid_min, grid_max], dim=1).int()
    graph = torch.zeros(
        [
            math.ceil(max(surf.knotvector_u) // res[0] * scaler),
            math.ceil(max(surf.knotvector_v) // res[1] * scaler),
        ],
        dtype=torch.int8,
        device=torch.device("cuda"),
    )

    # Fill the grids with 1s
    for grid in grids:
        graph[grid[0, 0] : grid[1, 0], grid[0, 1] : grid[1, 1]] = 1
    return graph, res


def has_feature(aabb, graph):
    """
    Check if an AABB is within the range of the featured graph.

    Parameters:
    aabb (torch.Tensor): A tensor of AABBs.
    graph (torch.Tensor): A tensor representing the featured graph.

    Returns:
    bool: True if the AABB is within the range of the graph, False otherwise.
    """
    # Check if the AABB is within the range of the graph
    if (
        aabb[0, 0] < 0
        or aabb[0, 1] < 0
        or aabb[1, 0] > graph.shape[0]
        or aabb[1, 1] > graph.shape[1]
    ):
        return False
    else:
        # Check if the AABB overlaps with any feature in the graph
        area = graph[aabb[0, 0] : aabb[1, 0], aabb[0, 1] : aabb[1, 1]]
        return torch.any(area > 0)


def bound_reduce(aabb, graph, offset):
    """
    Reduce the bounds of an AABB that does not contain any features.

    Parameters:
    aabb (torch.Tensor): A tensor of AABBs.
    graph (torch.Tensor): A tensor representing a graph.
    offset (torch.Tensor): An offset tensor.

    Returns:
    aabb (torch.Tensor): The reduced AABB.
    """
    reduce = torch.ones(4, dtype=bool, device=torch.device("cuda"))
    while torch.any(reduce):
        sides = torch.tensor(
            [
                [
                    [aabb[0, 0], aabb[0, 1]],
                    [aabb[1, 0], aabb[0, 1] + 1],
                ],
                [
                    [aabb[0, 0], aabb[0, 1]],
                    [aabb[0, 0] + 1, aabb[1, 1]],
                ],
                [
                    [aabb[1, 0] - 1, aabb[0, 1]],
                    [aabb[1, 0], aabb[1, 1]],
                ],
                [
                    [aabb[0, 0], aabb[1, 1] - 1],
                    [aabb[1, 0], aabb[1, 1]],
                ],
            ],
            device=torch.device("cuda"),
        )
        for i in range(4):
            if reduce[i]:
                reduce[i] = ~has_feature(sides[i], graph)
            if reduce[i]:
                aabb -= offset[i]
    return aabb


def expand_cluster(cluster, graph, width, offset):
    """
    Expand a cluster of AABBs.

    Parameters:
    cluster (torch.Tensor): A tensor of AABBs.
    graph (torch.Tensor): A tensor representing a graph.
    width (float): The maximum width of the cluster.
    offset (torch.Tensor): An offset tensor.

    Returns:
    cluster, expand (torch.Tensor): The expanded cluster and a boolean tensor indicating which sides are able to expand.
    """
    expand = torch.ones(4, dtype=bool, device=torch.device("cuda"))
    flag = False
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
            ],
            device=torch.device("cuda"),
        )
        for i in range(4):
            if expand[i]:
                expand[i] = has_feature(neighbors[i], graph)
        if flag or ((~expand[0] & ~expand[3]) | (~expand[1] & ~expand[2])):
            break
        for i in range(4):
            if expand[i]:
                cluster += offset[i]
        lens = cluster[1] - cluster[0]
        if width > 0.0 and torch.any(lens > width):
            flag = True
    return cluster, expand


def gen_new_cluster(cluster, graph, offset):
    """
    Generate a new cluster of AABBs by expanding the current cluster towards the direction with the most features.

    Parameters:
    cluster (torch.Tensor): A tensor of AABBs representing the current cluster.
    graph (torch.Tensor): A tensor representing a graph.
    offset (torch.Tensor): An offset tensor.

    Returns:
    cluster (torch.Tensor): The new cluster of AABBs.
    """
    lens = cluster[1] - cluster[0]
    # Get optimal neighbor subgrid
    neighbor_subgrids = torch.tensor(
        [
            [
                [cluster[0, 0], cluster[0, 1] - lens[1]],
                [cluster[1, 0], cluster[0, 1]],
            ],
            [
                [cluster[0, 0] - lens[0], cluster[0, 1]],
                [cluster[0, 0], cluster[1, 1]],
            ],
            [
                [cluster[1, 0], cluster[0, 1]],
                [cluster[1, 0] + lens[0], cluster[1, 1]],
            ],
            [
                [cluster[0, 0], cluster[1, 1]],
                [cluster[1, 0], cluster[1, 1] + lens[1]],
            ],
        ],
        device=torch.device("cuda"),
    )
    neighbor_subgrids[..., 0] = neighbor_subgrids[..., 0].clamp(0, graph.shape[0])
    neighbor_subgrids[..., 1] = neighbor_subgrids[..., 1].clamp(0, graph.shape[1])
    feature_counts = torch.zeros(4, dtype=int, device=torch.device("cuda"))
    for i in range(4):
        area = graph[
            neighbor_subgrids[i, 0, 0] : neighbor_subgrids[i, 1, 0],
            neighbor_subgrids[i, 0, 1] : neighbor_subgrids[i, 1, 1],
        ]
        feature_counts[i] = torch.sum(area > 0)
    index = torch.argmax(feature_counts)
    target_neighbor = neighbor_subgrids[index]
    cluster = bound_reduce(target_neighbor, graph, offset)
    for i in range(4):
        if i != index:
            area = graph[
                neighbor_subgrids[i, 0, 0] : neighbor_subgrids[i, 1, 0],
                neighbor_subgrids[i, 0, 1] : neighbor_subgrids[i, 1, 1],
            ]
            area = 0
    new_lens = cluster[1] - cluster[0]
    half_lens = new_lens // 2
    quarter_lens = new_lens // 4
    if index == 1 or index == 2:
        cluster -= half_lens[0] * offset[index] + quarter_lens[1] * (
            offset[0] + offset[3]
        )
    else:
        cluster -= half_lens[1] * offset[index] + quarter_lens[0] * (
            offset[1] + offset[2]
        )
    return cluster


def adjust_first_two_clusters(clusters, graph, offset):
    """
    Adjust the first two clusters of AABBs since they are not constrained by the local width.

    Parameters:
    clusters (list): A list of tensors of AABBs representing the current clusters.
    graph (torch.Tensor): A tensor representing a graph.
    offset (torch.Tensor): An offset tensor.

    Returns:
    clusters (list): The list of adjusted clusters of AABBs.
    """
    if len(clusters) < 4:
        return clusters
    graph = torch.abs(graph)

    def process_cluster(clusters, index, graph, offset):
        width = torch.norm(
            clusters[index + 2].mean(dim=1, dtype=float)
            - clusters[index + 1].mean(dim=1, dtype=float)
        )
        clusters[index], _ = expand_cluster(new_cluster, graph, width, offset)
        graph[
            clusters[index][0, 0] : clusters[index][1, 0],
            clusters[index][0, 1] : clusters[index][1, 1],
        ] = 0
        return clusters, graph

    new_cluster = gen_new_cluster(clusters[2], graph, offset)
    clusters, graph = process_cluster(clusters, 1, graph, offset)
    new_cluster = gen_new_cluster(clusters[1], graph, offset)
    clusters, graph = process_cluster(clusters, 0, graph, offset)

    # Gen new clusters inversely
    if torch.any(graph > 0):
        cluster = gen_new_cluster(clusters[0], graph, offset)
        while True:
            width = torch.norm(
                clusters[0].mean(dim=1, dtype=float)
                - clusters[1].mean(dim=1, dtype=float)
            )
            cluster, expand = expand_cluster(cluster, graph, width, offset)
            clusters.insert(0, cluster)
            graph[cluster[0, 0] : cluster[1, 0], cluster[0, 1] : cluster[1, 1]] = 0
            if torch.any(expand):
                cluster = gen_new_cluster(cluster, graph, offset)
            else:
                break
    return clusters


def sequence_joining(uv, surf, scaler=100.0, threshold=5):
    """
    Join sequences of AABBs into clusters and generate their centroids and AABBs.

    Parameters:
    uv (torch.Tensor): The knot vectors in the U and V direction.
    surf (object): A NURBS surface object.
    scaler (float): Scaler for the knot vectors. Default is 100.0.
    threshold (int): The minimum number of AABBs for a cluster. Default is 5.

    Returns:
    all_centroids, all_means (list): The AABBs and centroids of the clusters.
    """
    graph, res = gen_grids(uv, surf, scaler)
    all_centroids = []
    all_means = []
    while torch.any(graph > 0):
        clusters = []
        pos = torch.nonzero(graph > 0)
        # Initial cluster
        cluster = torch.tensor(
            [
                [pos[0, 0], pos[0, 1]],
                [pos[0, 0] + 1, pos[0, 1] + 1],
            ],
            dtype=int,
            device=torch.device("cuda"),
        )
        offset = torch.tensor(
            [[[0, -1], [0, 0]], [[-1, 0], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]],
            device=torch.device("cuda"),
        )
        width = 5.0
        count = 0
        while True:
            if count == 2:
                width = torch.norm(
                    clusters[-1].mean(dim=1, dtype=float)
                    - clusters[-2].mean(dim=1, dtype=float)
                )
            cluster, expand = expand_cluster(cluster, graph, width, offset)
            clusters.append(cluster)
            if count < 2:
                count += 1
                graph[cluster[0, 0] : cluster[1, 0], cluster[0, 1] : cluster[1, 1]] = -1
            else:
                graph[cluster[0, 0] : cluster[1, 0], cluster[0, 1] : cluster[1, 1]] = 0
            if torch.any(expand):
                cluster = gen_new_cluster(cluster, graph, offset)
            else:
                break

        # clusters = adjust_first_two_clusters(clusters, graph, offset)
        if len(clusters) < threshold:
            continue

        centroids = torch.zeros([len(clusters) + 1, 2, 2], device=torch.device("cuda"))
        centroids[0] = (
            torch.tensor(
                [
                    [clusters[0][0, 0], clusters[0][0, 1]],
                    [clusters[0][0, 0], clusters[0][0, 1]],
                ],
                device=torch.device("cuda"),
            )
            * res
        )
        for i, cluster in enumerate(clusters):
            centroids[i + 1] = cluster.float() * res

        all_centroids.append(centroids / scaler)
        all_means.append(torch.mean(centroids, dim=1) / scaler)
    return all_centroids, all_means


def point_to_surface(evalpts, points):
    """
    Find the closest points on a surface to a set of points.

    Parameters:
    evalpts (torch.Tensor): The points on the surface.
    points (torch.Tensor): The set of points.

    Returns:
    torch.Tensor: The closest points on the surface.
    """
    _, indices = torch.min(
        torch.norm(evalpts.unsqueeze(0) - points.unsqueeze(1), dim=2), dim=1
    )
    return evalpts[indices]


def accuracy_improvement(pts, evalpts1, evalpts2, max_iter=20, threshold=1e-3):
    """
    Improve the accuracy of a set of points by iteratively moving them closer to two surfaces.

    Parameters:
    pts (torch.Tensor): The set of points.
    evalpts1, evalpts2 (torch.Tensor): The points on the two surfaces.
    max_iter (int): The maximum number of iterations. Default is 20.
    threshold (float): The distance threshold for convergence. Default is 1e-3.

    Returns:
    torch.Tensor: The set of points after accuracy improvement.
    """
    mask = torch.ones(pts.shape[0], dtype=bool, device=torch.device("cuda"))
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
        x = torch.linalg.lstsq(A, b).solution
        pts[mask][mask2] = x
        mask[mask.clone()][mask2] = (
            torch.norm(x - pts[mask][mask2], dim=1)
        ) > threshold
        max_iter -= 1
    return pts


def find_closest_points(pts1, pts2):
    """
    Find the closest points in pts2 for each point in pts1.

    Parameters:
    pts1, pts2 (torch.Tensor): The sets of points.

    Returns:
    torch.Tensor: The closest points in pts2 for each point in pts1.
    """
    dists = torch.norm(
        pts1.unsqueeze(1) - pts2.unsqueeze(0), dim=2
    )  # [n, m] distance matrix
    indices = torch.argmin(dists, dim=1)
    return pts2[indices]

def is_inside_polygon(polygon, points):
    # 获取每条边的起点和终点
    edge_start = polygon
    edge_end = polygon.roll(-1, dims=0)

    # 计算每个点和所有边的交点次数
    # 使用广播，点的y坐标与边的y坐标比较，获取每个点对所有边的位置情况
    below_start = (points[:, 1].unsqueeze(1) - edge_start[:, 1]) > 0
    below_end = (points[:, 1].unsqueeze(1) - edge_end[:, 1]) > 0
    segments_cross = below_start != below_end

    # 计算交叉点的x坐标并检查是否在点左侧
    edge_dx = edge_end[:, 0] - edge_start[:, 0]
    edge_dy = edge_end[:, 1] - edge_start[:, 1]
    dx = ((points[:, 1].unsqueeze(1) - edge_start[:, 1]) * edge_dx) / edge_dy + edge_start[:, 0]
    on_left = points[:, 0].unsqueeze(1) < dx

    # 交叉点数是否为奇数标记点是否在多边形内
    crossings = segments_cross & on_left
    inside = crossings.sum(dim=1) % 2 == 1
    return inside

def trim_surf(surf, uv, scaler):
    for trim in surf.trims:
        polygon = torch.tensor(trim.evalpts, dtype=float, device=torch.device("cuda")) * scaler
        index = ~(is_inside_polygon(polygon, uv[:, 0, :]) & is_inside_polygon(polygon, uv[:, 1, :]))
        uv = uv[index]
    return uv

def gen_curves(u1, v1, col1, surf1, u2, v2, col2, surf2, scaler1=100.0, scaler2=100.0):
    """
    Generate curves between two NURBS surfaces.

    Parameters:
    u1, v1, u2, v2 (torch.Tensor): Unstripped AABBs in the U and V direction for the two surfaces.
    col1, col2 (torch.Tensor): Indices to extract for the two surfaces.
    surf1, surf2 (object): NURBS surface objects.
    scaler1, scaler2 (float): Scalers for the knot vectors.

    Returns:
    uv3d1, uv3d2, aabb3d1, all_pts (torch.Tensor): The 3D AABBs for the two surfaces, the 3D clustered AABBs for the first surface, and the generated curves.
    """
    uv1, uv2 = strip_thinning(
        u1, v1, col1, surf1, u2, v2, col2, surf2, scaler1, scaler2
    )
    uv1 = trim_surf(surf1, uv1, scaler1)
    uv2 = trim_surf(surf2, uv2, scaler2)
    aabb1, pts1 = sequence_joining(uv1, surf1, scaler1)
    evalpts1 = torch.tensor(surf1.evalpts, device=torch.device("cuda"))
    evalpts2 = torch.tensor(surf2.evalpts, device=torch.device("cuda"))

    all_pts = []
    for i in range(len(aabb1)):
        pts3d1 = torch.tensor(
            surf1.evaluate_list(pts1[i].tolist()), device=torch.device("cuda")
        )

        closest_pts = find_closest_points(pts3d1, evalpts2)
        midpoints = (pts3d1 + closest_pts) / 2.0
        pts = accuracy_improvement(midpoints, evalpts1, evalpts2)
        all_pts.append(pts)

    # Gen imgui AABBs
    uv3d1 = torch.tensor(
        surf1.evaluate_list((uv1 / scaler1).reshape(-1, 2).tolist())
    ).reshape(-1, 2, 3)
    uv3d2 = torch.tensor(
        surf2.evaluate_list((uv2 / scaler2).reshape(-1, 2).tolist())
    ).reshape(-1, 2, 3)
    aabb3d1 = []
    for aabb in aabb1:
        aabb3d1.append(
            torch.tensor(surf1.evaluate_list(aabb.reshape(-1, 2).tolist())).reshape(
                -1, 2, 3
            )
        )
    return uv3d1, uv3d2, aabb3d1, all_pts, uv1 / scaler1, uv2 / scaler2
