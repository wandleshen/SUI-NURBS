# encoding: utf-8

import numpy as np
import torch
import cProfile
import argparse

from src.aabspline import gen_aabb
from src.overlaptest import region_extraction
from src.curvegeneration import (
    strip_thinning,
    sequence_joining,
    accuracy_improvement,
    find_closest_points,
)
from src import utils


def main(filename, M, N, P, Q, SCALER):
    ctrlpts4d = np.load(f"data/{filename}")
    # even_rows = np.arange(1, ctrlpts4d.shape[1], 2)
    # ctrlpts4d[:, even_rows, -1] = .9

    ctrlpts4d_rev = ctrlpts4d[..., [1, 0, 2, 3]]

    surf = utils.gen_surface(ctrlpts4d.tolist(), P, Q, 100)

    surf2 = utils.gen_surface(ctrlpts4d_rev.tolist(), P, Q, 100)

    knot_u1 = torch.tensor(surf.knotvector_u, device=torch.device("cuda"))
    knot_v1 = torch.tensor(surf.knotvector_v, device=torch.device("cuda"))
    ctrlpts1 = torch.tensor(surf.ctrlpts2d, device=torch.device("cuda"))
    knot_u2 = torch.tensor(surf2.knotvector_u, device=torch.device("cuda"))
    knot_v2 = torch.tensor(surf2.knotvector_v, device=torch.device("cuda"))
    ctrlpts2 = torch.tensor(surf2.ctrlpts2d, device=torch.device("cuda"))

    def gen_curves_perf(u1, v1, col1, surf1, u2, v2, col2, surf2, scaler=SCALER):
        uv1, _ = strip_thinning(u1, v1, col1, surf1, u2, v2, col2, surf2, scaler)
        _, pts1 = sequence_joining(uv1, surf1, scaler, threshold=1)
        pts3d1 = torch.tensor(
            surf1.evaluate_list(pts1[0].tolist()), device=torch.device("cuda")
        )
        evalpts1 = torch.tensor(surf1.evalpts, device=torch.device("cuda"))
        evalpts2 = torch.tensor(surf2.evalpts, device=torch.device("cuda"))

        closest_pts = find_closest_points(pts3d1, evalpts2)
        midpoints = (pts3d1 + closest_pts) / 2.0
        pts = accuracy_improvement(midpoints, evalpts1, evalpts2)

        return pts

    def warm_up():
        pts, u1, v1 = gen_aabb(
            knot_u1,
            knot_v1,
            ctrlpts1,
            M,
            N,
            P,
            Q,
            scaler=SCALER,
        )

        pts2, u2, v2 = gen_aabb(
            knot_u2,
            knot_v2,
            ctrlpts2,
            M,
            N,
            P,
            Q,
            scaler=SCALER,
        )

        col, col2 = region_extraction(pts, pts2)
        curve = gen_curves_perf(u1, v1, col, surf, u2, v2, col2, surf2, scaler=SCALER)
        return curve

    warm_up()
    pr = cProfile.Profile()
    pr.enable()

    pts, u1, v1 = gen_aabb(
        knot_u1,
        knot_v1,
        ctrlpts1,
        M,
        N,
        P,
        Q,
        scaler=SCALER,
    )

    pts2, u2, v2 = gen_aabb(
        knot_u2,
        knot_v2,
        ctrlpts2,
        M,
        N,
        P,
        Q,
        scaler=SCALER,
    )

    col, col2 = region_extraction(pts, pts2)
    curve = gen_curves_perf(u1, v1, col, surf, u2, v2, col2, surf2, scaler=SCALER)

    pr.disable()
    pr.print_stats(sort="cumtime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        required=False,
        default="plain.npy",
        help="The filename of the data",
    )
    parser.add_argument(
        "-m",
        "--u_intervals",
        dest="m",
        required=False,
        default=32,
        help="The number of intervals in the u direction",
    )
    parser.add_argument(
        "-n",
        "--v_intervals",
        dest="n",
        required=False,
        default=32,
        help="The number of intervals in the v direction",
    )
    parser.add_argument(
        "-p",
        "--u_degree",
        dest="p",
        required=False,
        default=3,
        help="The degree of the u direction",
    )
    parser.add_argument(
        "-q",
        "--v_degree",
        dest="q",
        required=False,
        default=3,
        help="The degree of the v direction",
    )
    parser.add_argument(
        "-s",
        "--scaler",
        dest="scaler",
        required=False,
        default=1.0,
        help="The scaler of the knotvectors",
    )
    args = parser.parse_args()
    main(args.filename, args.m, args.n, args.p, args.q, args.scaler)
