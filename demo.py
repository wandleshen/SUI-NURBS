# encoding: utf-8

import numpy as np
import torch
import argparse

from src.aabspline import gen_aabb
from src.overlaptest import region_extraction
from src.curvegeneration import gen_curves
from src import utils


def main(filename, M, N, P, Q, SCALER):
    ctrlpts4d = np.load(f"data/{filename}")
    # even_rows = np.arange(1, ctrlpts4d.shape[1], 2)
    # ctrlpts4d[:, even_rows, -1] = 0.9

    ctrlpts4d_rev = ctrlpts4d[..., [1, 0, 2, 3]]

    surf = utils.gen_surface(ctrlpts4d.tolist(), P, Q, 100)

    surf2 = utils.gen_surface(ctrlpts4d_rev.tolist(), P, Q, 100)

    pts, u1, v1 = gen_aabb(
        torch.tensor(surf.knotvector_u, device=torch.device("cuda")),
        torch.tensor(surf.knotvector_v, device=torch.device("cuda")),
        torch.tensor(surf.ctrlpts2d, device=torch.device("cuda")),
        M,
        N,
        P,
        Q,
        scaler=SCALER,
    )

    pts2, u2, v2 = gen_aabb(
        torch.tensor(surf2.knotvector_u, device=torch.device("cuda")),
        torch.tensor(surf2.knotvector_v, device=torch.device("cuda")),
        torch.tensor(surf2.ctrlpts2d, device=torch.device("cuda")),
        M,
        N,
        P,
        Q,
        scaler=SCALER,
    )

    col, col2 = region_extraction(pts, pts2)
    stripped, stripped2, cluster, curve = gen_curves(
        u1, v1, col, surf, u2, v2, col2, surf2, scaler=SCALER
    )
    extract, pts = utils.extract_aabb(pts, col)
    extract2, pts2 = utils.extract_aabb(pts2, col2)

    utils.render(
        surf,
        surf2,
        extract,
        extract2,
        stripped,
        stripped2,
        cluster,
        curve,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        required=True,
        help="The filename of the data",
    )
    parser.add_argument(
        "-m",
        "--u_intervals",
        dest="m",
        required=False,
        default=1024,
        help="The number of intervals in the u direction",
    )
    parser.add_argument(
        "-n",
        "--v_intervals",
        dest="n",
        required=False,
        default=1024,
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
        default=25.0,
        help="The scaler of the knotvectors",
    )
    args = parser.parse_args()
    main(args.filename, args.m, args.n, args.p, args.q, args.scaler)
