# encoding: utf-8

import numpy as np
import torch
import argparse

from src.aabspline import gen_aabb
from src.overlaptest import region_extraction
from src.curvegeneration import gen_curves
from src import utils


def main(filename0, filename1, m0, m1, n0, n1, p0, p1, q0, q1, scaler0, scaler1):
    ctrlpts4d = np.load(f"data/{filename0}")
    # even_rows = np.arange(1, ctrlpts4d.shape[1], 2)
    # ctrlpts4d[:, even_rows, -1] = 0.9

    ctrlpts4d_rev = np.load(f"data/{filename1}")

    RES = 50

    from geomdl import operations, BSpline, knotvector
    from geomdl.shapes import curve2d

    # Create trim curves
    curve = BSpline.Curve()
    curve.degree = 1
    curve.ctrlpts = [
        [1.1, 1.1],
        [0.24, 1.1],
        [0.24, 1.1],
        [0.24, 0.49],
        [0.24, 0.49],
        [1.1, 0.49],
        [1.1, 0.49],
        [1.1, 1.1],
    ]

    curve.knotvector = knotvector.generate(curve.degree, curve.ctrlpts_size)
    curve.delta = 0.001
    trim_curve = operations.translate(curve, (0.25, 0.25))

    circle = curve2d.full_circle(radius=0.15)
    operations.translate(circle, (0.5, 0.5), inplace=True)

    surf = utils.gen_surface(ctrlpts4d.tolist(), p0, q0, RES, [trim_curve])

    surf2 = utils.gen_surface(ctrlpts4d_rev.tolist(), p1, q1, RES)

    pts, u1, v1 = gen_aabb(
        torch.tensor(surf.knotvector_u, device=torch.device("cuda")),
        torch.tensor(surf.knotvector_v, device=torch.device("cuda")),
        torch.tensor(surf.ctrlpts2d, device=torch.device("cuda")),
        m0,
        n0,
        p0,
        q0,
        scaler=scaler0,
    )

    pts2, u2, v2 = gen_aabb(
        torch.tensor(surf2.knotvector_u, device=torch.device("cuda")),
        torch.tensor(surf2.knotvector_v, device=torch.device("cuda")),
        torch.tensor(surf2.ctrlpts2d, device=torch.device("cuda")),
        m1,
        n1,
        p1,
        q1,
        scaler=scaler1,
    )

    col, col2 = region_extraction(pts, pts2)
    stripped, stripped2, cluster, curve, uv1, uv2 = gen_curves(
        u1, v1, col, surf, u2, v2, col2, surf2, scaler0, scaler1
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
        uv1,
        uv2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    for i in range(2):
        parser.add_argument(
            f"-f{i}",
            f"--filename{i}",
            dest=f"filename{i}",
            required=True,
            help="The filename of the surface",
        )
        parser.add_argument(
            f"-m{i}",
            f"--u-intervals{i}",
            dest=f"m{i}",
            type=int,
            required=False,
            default=1024,
            help="Number of intervals in the u-direction",
        )
        parser.add_argument(
            f"-n{i}",
            f"--v-intervals{i}",
            dest=f"n{i}",
            type=int,
            required=False,
            default=1024,
            help="Number of intervals in the v-direction",
        )
        parser.add_argument(
            f"-p{i}",
            f"--u-degree{i}",
            dest=f"p{i}",
            type=int,
            required=False,
            default=3,
            help="Degree of the B-spline basis function in the u-direction",
        )
        parser.add_argument(
            f"-q{i}",
            f"--v-degree{i}",
            dest=f"q{i}",
            type=int,
            required=False,
            default=3,
            help="Degree of the B-spline basis function in the v-direction",
        )
        parser.add_argument(
            f"-s{i}",
            f"--scaler{i}",
            dest=f"scaler{i}",
            type=float,
            required=False,
            default=25.0,
            help="Scaler of the knotvectors",
        )
    args = parser.parse_args()
    main(
        args.filename0,
        args.filename1,
        args.m0,
        args.m1,
        args.n0,
        args.n1,
        args.p0,
        args.p1,
        args.q0,
        args.q1,
        args.scaler0,
        args.scaler1,
    )
