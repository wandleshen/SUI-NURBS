# encoding: utf-8

import numpy as np
import torch

from src.aabspline import gen_aabb
from src.overlaptest import region_extraction
from src.curvegeneration import (
    strip_thinning,
    sequence_joining,
    accuracy_improvement,
    find_closest_points,
)
from src import utils


# Control points
ctrlpts = np.array(
    [
        [
            [-7.61089252801589e-18, -0.2897777575459787, 0.24128292790000003],
            [-7.61089252801589e-18, -0.24607715997095897, 0.23866529070000003],
            [-7.61089252801589e-18, -0.20237656239593924, 0.2367717987],
            [-7.61089252801589e-18, -0.1586759648209195, 0.24234888940000002],
            [-7.61089252801589e-18, -0.1149753672458998, 0.2389167365],
            [-7.61089252801589e-18, -0.07127476967088009, 0.2431076015],
            [-7.61089252801589e-18, -0.027574172095860328, 0.24193291749999998],
            [-7.61089252801589e-18, 0.01612642547915938, 0.24012009532999998],
            [-7.61089252801589e-18, 0.059827023054179085, 0.228174578],
            [-7.61089252801589e-18, 0.10352762062919879, 0.229347744],
        ],
        [
            [0.024421897699675375, -0.2897777575459787, 0.2381480632],
            [0.024421897699675375, -0.24607715997095897, 0.23866529070000003],
            [0.024421897699675375, -0.20237656239593924, 0.2367717987],
            [0.024421897699675375, -0.1586759648209195, 0.24198836720000003],
            [0.024421897699675375, -0.1149753672458998, 0.23931915890000002],
            [0.024421897699675375, -0.07127476967088009, 0.23319710999999999],
            [0.024421897699675375, -0.027574172095860328, 0.232682633],
            [0.024421897699675375, 0.01612642547915938, 0.229347744],
            [0.024421897699675375, 0.059827023054179085, 0.222313173],
            [0.024421897699675375, 0.10352762062919879, 0.21708676899999998],
        ],
        [
            [0.048843795399350756, -0.2897777575459787, 0.2381480632],
            [0.048843795399350756, -0.24607715997095897, 0.23971904967000002],
            [0.048843795399350756, -0.20237656239593924, 0.23941031932],
            [0.048843795399350756, -0.1586759648209195, 0.23264622810000002],
            [0.048843795399350756, -0.1149753672458998, 0.23551773680000002],
            [0.048843795399350756, -0.07127476967088009, 0.225225547],
            [0.048843795399350756, -0.027574172095860328, 0.223425943],
            [0.048843795399350756, 0.01612642547915938, 0.21708676899999998],
            [0.048843795399350756, 0.059827023054179085, 0.213317081],
            [0.048843795399350756, 0.10352762062919879, 0.208976675],
        ],
        [
            [0.07326569309902613, -0.2897777575459787, 0.2403961699],
            [0.07326569309902613, -0.24607715997095897, 0.23167925960000002],
            [0.07326569309902613, -0.20237656239593924, 0.23291311110000001],
            [0.07326569309902613, -0.1586759648209195, 0.227828352],
            [0.07326569309902613, -0.1149753672458998, 0.222709868],
            [0.07326569309902613, -0.07127476967088009, 0.21650994499999998],
            [0.07326569309902613, -0.027574172095860328, 0.214997103],
            [0.07326569309902613, 0.01612642547915938, 0.208976675],
            [0.07326569309902613, 0.059827023054179085, 0.204813069],
            [0.07326569309902613, 0.10352762062919879, 0.20695907],
        ],
        [
            [0.09768759079870151, -0.2897777575459787, 0.229500361],
            [0.09768759079870151, -0.24607715997095897, 0.22854929100000002],
            [0.09768759079870151, -0.20237656239593924, 0.22066884700000003],
            [0.09768759079870151, -0.1586759648209195, 0.21862977000000003],
            [0.09768759079870151, -0.1149753672458998, 0.21582428500000003],
            [0.09768759079870151, -0.07127476967088009, 0.211354773],
            [0.09768759079870151, -0.027574172095860328, 0.208851874],
            [0.09768759079870151, 0.01612642547915938, 0.204235578],
            [0.09768759079870151, 0.059827023054179085, 0.20185667399999999],
            [0.09768759079870151, 0.10352762062919879, 0.202759823],
        ],
        [
            [0.1221094884983769, -0.2897777575459787, 0.22200751500000002],
            [0.1221094884983769, -0.24607715997095897, 0.21814036300000003],
            [0.1221094884983769, -0.20237656239593924, 0.21267818300000002],
            [0.1221094884983769, -0.1586759648209195, 0.21044493100000003],
            [0.1221094884983769, -0.1149753672458998, 0.207020784],
            [0.1221094884983769, -0.07127476967088009, 0.20556106999999998],
            [0.1221094884983769, -0.027574172095860328, 0.201062729],
            [0.1221094884983769, 0.01612642547915938, 0.197511831],
            [0.1221094884983769, 0.059827023054179085, 0.195832601],
            [0.1221094884983769, 0.10352762062919879, 0.193345452],
        ],
        [
            [0.14653138619805228, -0.2897777575459787, 0.211311675],
            [0.14653138619805228, -0.24607715997095897, 0.20956684300000003],
            [0.14653138619805228, -0.20237656239593924, 0.20713202800000002],
            [0.14653138619805228, -0.1586759648209195, 0.20604932100000004],
            [0.14653138619805228, -0.1149753672458998, 0.20194529900000002],
            [0.14653138619805228, -0.07127476967088009, 0.203969149],
            [0.14653138619805228, -0.027574172095860328, 0.199576122],
            [0.14653138619805228, 0.01612642547915938, 0.19421761999999998],
            [0.14653138619805228, 0.059827023054179085, 0.192907857],
            [0.14653138619805228, 0.10352762062919879, 0.194479408],
        ],
        [
            [0.17095328389772768, -0.2897777575459787, 0.207531335],
            [0.17095328389772768, -0.24607715997095897, 0.20450680100000002],
            [0.17095328389772768, -0.20237656239593924, 0.20307632700000003],
            [0.17095328389772768, -0.1586759648209195, 0.198040986],
            [0.17095328389772768, -0.1149753672458998, 0.19598473800000002],
            [0.17095328389772768, -0.07127476967088009, 0.194385906],
            [0.17095328389772768, -0.027574172095860328, 0.19596861799999998],
            [0.17095328389772768, 0.01612642547915938, 0.19314960199999998],
            [0.17095328389772768, 0.059827023054179085, 0.19418976599999999],
            [0.17095328389772768, 0.10352762062919879, 0.194676423],
        ],
        [
            [0.19537518159740305, -0.2897777575459787, 0.20314200400000001],
            [0.19537518159740305, -0.24607715997095897, 0.19578809100000003],
            [0.19537518159740305, -0.20237656239593924, 0.194028316],
            [0.19537518159740305, -0.1586759648209195, 0.19421588000000004],
            [0.19537518159740305, -0.1149753672458998, 0.19265553500000002],
            [0.19537518159740305, -0.07127476967088009, 0.19474430500000003],
            [0.19537518159740305, -0.027574172095860328, 0.191584667],
            [0.19537518159740305, 0.01612642547915938, 0.192335443],
            [0.19537518159740305, 0.059827023054179085, 0.190259548],
            [0.19537518159740305, 0.10352762062919879, 0.193471575],
        ],
        [
            [0.21979707929707842, -0.2897777575459787, 0.199092703],
            [0.21979707929707842, -0.24607715997095897, 0.19369303200000001],
            [0.21979707929707842, -0.20237656239593924, 0.19420942100000002],
            [0.21979707929707842, -0.1586759648209195, 0.19253541700000001],
            [0.21979707929707842, -0.1149753672458998, 0.19179110800000002],
            [0.21979707929707842, -0.07127476967088009, 0.19453510100000002],
            [0.21979707929707842, -0.027574172095860328, 0.189472547],
            [0.21979707929707842, 0.01612642547915938, 0.19585302999999998],
            [0.21979707929707842, 0.059827023054179085, 0.19295927699999998],
            [0.21979707929707842, 0.10352762062919879, 0.192552408],
        ],
        [
            [0.24421897699675382, -0.2897777575459787, 0.19353207700000002],
            [0.24421897699675382, -0.24607715997095897, 0.19170584300000001],
            [0.24421897699675382, -0.20237656239593924, 0.19206821300000002],
            [0.24421897699675382, -0.1586759648209195, 0.19422611000000004],
            [0.24421897699675382, -0.1149753672458998, 0.192600542],
            [0.24421897699675382, -0.07127476967088009, 0.19003549400000003],
            [0.24421897699675382, -0.027574172095860328, 0.194571098],
            [0.24421897699675382, 0.01612642547915938, 0.194581227],
            [0.24421897699675382, 0.059827023054179085, 0.194531569],
            [0.24421897699675382, 0.10352762062919879, 0.198418965],
        ],
        [
            [0.2686408746964292, -0.2897777575459787, 0.19071410400000002],
            [0.2686408746964292, -0.24607715997095897, 0.19266366000000001],
            [0.2686408746964292, -0.20237656239593924, 0.19471749100000002],
            [0.2686408746964292, -0.1586759648209195, 0.193096081],
            [0.2686408746964292, -0.1149753672458998, 0.19155679100000003],
            [0.2686408746964292, -0.07127476967088009, 0.19405540200000002],
            [0.2686408746964292, -0.027574172095860328, 0.193873768],
            [0.2686408746964292, 0.01612642547915938, 0.197695585],
            [0.2686408746964292, 0.059827023054179085, 0.200295174],
            [0.2686408746964292, 0.10352762062919879, 0.19943571999999998],
        ],
        [
            [0.29306277239610456, -0.2897777575459787, 0.19043683400000003],
            [0.29306277239610456, -0.24607715997095897, 0.192143632],
            [0.29306277239610456, -0.20237656239593924, 0.192500883],
            [0.29306277239610456, -0.1586759648209195, 0.19186572100000002],
            [0.29306277239610456, -0.1149753672458998, 0.197159292],
            [0.29306277239610456, -0.07127476967088009, 0.19304413100000004],
            [0.29306277239610456, -0.027574172095860328, 0.201695142],
            [0.29306277239610456, 0.01612642547915938, 0.20155146499999999],
            [0.29306277239610456, 0.059827023054179085, 0.200350636],
            [0.29306277239610456, 0.10352762062919879, 0.209189204],
        ],
        [
            [0.31748467009577996, -0.2897777575459787, 0.190993817],
            [0.31748467009577996, -0.24607715997095897, 0.19214587500000002],
            [0.31748467009577996, -0.20237656239593924, 0.197977854],
            [0.31748467009577996, -0.1586759648209195, 0.20169926600000002],
            [0.31748467009577996, -0.1149753672458998, 0.20004697300000002],
            [0.31748467009577996, -0.07127476967088009, 0.19732170000000002],
            [0.31748467009577996, -0.027574172095860328, 0.20536218],
            [0.31748467009577996, 0.01612642547915938, 0.20628407799999998],
            [0.31748467009577996, 0.059827023054179085, 0.20652380399999998],
            [0.31748467009577996, 0.10352762062919879, 0.213951269],
        ],
        [
            [0.34190656779545536, -0.2897777575459787, 0.20206964900000002],
            [0.34190656779545536, -0.24607715997095897, 0.19896244400000002],
            [0.34190656779545536, -0.20237656239593924, 0.19790584400000003],
            [0.34190656779545536, -0.1586759648209195, 0.20123195400000002],
            [0.34190656779545536, -0.1149753672458998, 0.20238759200000003],
            [0.34190656779545536, -0.07127476967088009, 0.20602075500000003],
            [0.34190656779545536, -0.027574172095860328, 0.20873628700000002],
            [0.34190656779545536, 0.01612642547915938, 0.212779859],
            [0.34190656779545536, 0.059827023054179085, 0.22190385099999999],
            [0.34190656779545536, 0.10352762062919879, 0.22182027599999998],
        ],
        [
            [0.3663284654951307, -0.2897777575459787, 0.20067726600000002],
            [0.3663284654951307, -0.24607715997095897, 0.20478332200000002],
            [0.3663284654951307, -0.20237656239593924, 0.19872739700000003],
            [0.3663284654951307, -0.1586759648209195, 0.20437576000000002],
            [0.3663284654951307, -0.1149753672458998, 0.21054040200000002],
            [0.3663284654951307, -0.07127476967088009, 0.213384498],
            [0.3663284654951307, -0.027574172095860328, 0.21837466000000003],
            [0.3663284654951307, 0.01612642547915938, 0.21788587799999998],
            [0.3663284654951307, 0.059827023054179085, 0.22929255499999998],
            [0.3663284654951307, 0.10352762062919879, 0.23501573669999998],
        ],
        [
            [0.3907503631948061, -0.2897777575459787, 0.20515258000000003],
            [0.3907503631948061, -0.24607715997095897, 0.203681199],
            [0.3907503631948061, -0.20237656239593924, 0.21017697200000002],
            [0.3907503631948061, -0.1586759648209195, 0.21418501800000003],
            [0.3907503631948061, -0.1149753672458998, 0.22149628200000002],
            [0.3907503631948061, -0.07127476967088009, 0.223567176],
            [0.3907503631948061, -0.027574172095860328, 0.22723658500000002],
            [0.3907503631948061, 0.01612642547915938, 0.2310638863],
            [0.3907503631948061, 0.059827023054179085, 0.2368903274],
            [0.3907503631948061, 0.10352762062919879, 0.23595058989999998],
        ],
        [
            [0.4151722608944815, -0.2897777575459787, 0.21017697200000002],
            [0.4151722608944815, -0.24607715997095897, 0.21199497400000003],
            [0.4151722608944815, -0.20237656239593924, 0.21927304900000003],
            [0.4151722608944815, -0.1586759648209195, 0.22315062200000002],
            [0.4151722608944815, -0.1149753672458998, 0.22857133500000001],
            [0.4151722608944815, -0.07127476967088009, 0.22965230400000003],
            [0.4151722608944815, -0.027574172095860328, 0.23310139800000002],
            [0.4151722608944815, 0.01612642547915938, 0.2377484811],
            [0.4151722608944815, 0.059827023054179085, 0.24010860769],
            [0.4151722608944815, 0.10352762062919879, 0.24123687459999998],
        ],
        [
            [0.43959415859415685, -0.2897777575459787, 0.21927304900000003],
            [0.43959415859415685, -0.24607715997095897, 0.22320697400000003],
            [0.43959415859415685, -0.20237656239593924, 0.228991882],
            [0.43959415859415685, -0.1586759648209195, 0.23194888030000002],
            [0.43959415859415685, -0.1149753672458998, 0.23462873860000003],
            [0.43959415859415685, -0.07127476967088009, 0.2368263197],
            [0.43959415859415685, -0.027574172095860328, 0.24436860180000003],
            [0.43959415859415685, 0.01612642547915938, 0.239770729],
            [0.43959415859415685, 0.059827023054179085, 0.23700326359999999],
            [0.43959415859415685, 0.10352762062919879, 0.24123687459999998],
        ],
        [
            [0.46401605629383225, -0.2897777575459787, 0.228991882],
            [0.46401605629383225, -0.24607715997095897, 0.23116480350000002],
            [0.46401605629383225, -0.20237656239593924, 0.24043857086],
            [0.46401605629383225, -0.1586759648209195, 0.23477968500000002],
            [0.46401605629383225, -0.1149753672458998, 0.24412456880000002],
            [0.46401605629383225, -0.07127476967088009, 0.24255738620000003],
            [0.46401605629383225, -0.027574172095860328, 0.24111023780000002],
            [0.46401605629383225, 0.01612642547915938, 0.239770729],
            [0.46401605629383225, 0.059827023054179085, 0.23700326359999999],
            [0.46401605629383225, 0.10352762062919879, 0.2416414023],
        ],
    ]
)

M = 32
N = 32

ctrlpts4d = np.concatenate(
    (ctrlpts, np.ones((ctrlpts.shape[0], ctrlpts.shape[1], 1), dtype=float)), axis=-1
)
# even_rows = np.arange(1, ctrlpts4d.shape[1], 2)
# ctrlpts4d[:, even_rows, -1] = .9

ctrlpts4d_rev = ctrlpts4d[..., [1, 0, 2, 3]]

surf = utils.gen_surface(ctrlpts4d.tolist(), 100)

surf2 = utils.gen_surface(ctrlpts4d_rev.tolist(), 100)

knot_u1 = torch.tensor(surf.knotvector_u, device=torch.device("cuda"))
knot_v1 = torch.tensor(surf.knotvector_v, device=torch.device("cuda"))
ctrlpts1 = torch.tensor(surf.ctrlpts2d, device=torch.device("cuda"))
knot_u2 = torch.tensor(surf2.knotvector_u, device=torch.device("cuda"))
knot_v2 = torch.tensor(surf2.knotvector_v, device=torch.device("cuda"))
ctrlpts2 = torch.tensor(surf2.ctrlpts2d, device=torch.device("cuda"))


def gen_curves_perf(u1, v1, col1, surf1, u2, v2, col2, surf2, scaler=1.0):
    uv1, uv2 = strip_thinning(u1, v1, col1, surf1, u2, v2, col2, surf2, scaler)
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


# Warm-up
pts, u1, v1 = gen_aabb(knot_u1, knot_v1, ctrlpts1, M, N, 3, 3, scaler=1.0)

pts2, u2, v2 = gen_aabb(knot_u2, knot_v2, ctrlpts2, M, N, 3, 3, scaler=1.0)

col, col2 = region_extraction(pts, pts2)
curve = gen_curves_perf(u1, v1, col, surf, u2, v2, col2, surf2)

# Warm-up End

import cProfile

pr = cProfile.Profile()
pr.enable()

pts, u1, v1 = gen_aabb(knot_u1, knot_v1, ctrlpts1, M, N, 3, 3, scaler=1.0)

pts2, u2, v2 = gen_aabb(knot_u2, knot_v2, ctrlpts2, M, N, 3, 3, scaler=1.0)

col, col2 = region_extraction(pts, pts2)
curve = gen_curves_perf(u1, v1, col, surf, u2, v2, col2, surf2)

pr.disable()
pr.print_stats(sort="cumtime")
