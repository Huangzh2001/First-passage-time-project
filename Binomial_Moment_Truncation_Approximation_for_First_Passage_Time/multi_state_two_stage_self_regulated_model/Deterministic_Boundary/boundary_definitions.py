"""批跑使用的首达判据 g。首达时间由 FirstEvent 根据 g 求解。"""

import numpy as np

from ssa import FirstEvent, affine_time_boundary, exponential_boundary


############################################
# 一、平滑指数边界（4种）
############################################

#   S(m,n) = 分子状态评分函数
#   B(t)   = floor + amplitude * exp(-t / decay)
#   g(t,m,n) = [S(m,n) - B(t)] / 4
# 首达条件：g(t,m,n) >= 0等价于：S(m,n) >= B(t)
# 物理含义：
#   系统的 mRNA 和蛋白质数量共同决定评分 S。
#   阈值 B(t) 随时间指数下降。
#   当评分首次达到当前阈值时，发生首达事件。


def smooth_presets():
    return {
        "product": (
            lambda m, n: (m + 1) * (n + 1),
            6., 5., 3.,
        ),
        "weighted": (
            lambda m, n: 1 + .8*m + 1.35*n + .30*m*n,
            5., 4., 3.5,
        ),
        "saturating": (
            lambda m, n: 1 + 1.25*m + .85*n + 2*m*n/(1+m+n),
            5., 4., 2.5,
        ),
        "curved": (
            lambda m, n: 1 + .9*m + .8*n + .18*(m+n)**2,
            5.5, 4.5, 4.,
        ),
    }


SMOOTH_BOUNDARIES = smooth_presets()
def smooth_boundary_pair(name):
    return exponential_boundary(*SMOOTH_BOUNDARIES[name])

############################################
# 二、固定计数条件（2种）
############################################

def fixed_count_boundary(name):
    if name not in ("either_count", "joint_or_high"):
        raise ValueError(f"Unknown fixed-count boundary: {name}")

    def g(t, m, n):
        if name == "either_count":
            return (np.maximum(2*m, 1.5*n) - 6) / 4

        score = np.minimum(2*m, 1.5*n) - 3
        high = (13*m + 8*n - 80) / 10
        return np.maximum(score, high) / 4

    return FirstEvent(g)


############################################
# 三、随时间持续放宽（2种）
############################################

def relaxing_boundary(name):
    if name not in ("delayed_ramp", "step_release"):
        raise ValueError(f"Unknown relaxing boundary: {name}")

    def g(t, m, n):
        score = 11*m + 9*n

        if name == "delayed_ramp":
            opening = 1.5 + (80-score)/20
            return np.clip(
                (t-opening)/2,
                (score-80)/40,
                (score-40)/40,
            )

        return (score - np.where(t < 2.5, 80, 48)) / 40

    return FirstEvent(g)


############################################
# 四、临时放宽（2种）
############################################

def pulse_boundary(name):
    if name not in ("guarded_pulse", "guarded_double_pulse"):
        raise ValueError(f"Unknown pulse boundary: {name}")

    centers = (3.,) if name == "guarded_pulse" else (2., 5.)

    def g(t, m, n):
        score = 9*m + 11*n
        value = (score-80)/40
        half_width = 1.5*(score-50.5)/29.5

        for center in centers:
            start = center-half_width
            end = center+half_width
            value = np.maximum(
                value,
                (29.5/60)*np.minimum(t-start, end-t),
            )

        return value

    return FirstEvent(g)


############################################
# 五、计数范围条件（2种）
############################################

def count_band_boundary(name):
    if name not in ("protein_band", "balance_band"):
        raise ValueError(f"Unknown count-band boundary: {name}")

    def g(t, m, n):
        high = (13*m + 8*n - 80) / 10

        if name == "protein_band":
            score = np.minimum.reduce((m-1, n-2, 4-n))
        else:
            score = np.minimum(m+n-4, 1-np.abs(m-n))

        return np.maximum(score, high) / 4

    return FirstEvent(g)


# 保留 run_models.py 现有的调用方式。
NEW_BOUNDARIES = (
    "either_count", "joint_or_high",
    "delayed_ramp", "step_release",
    "guarded_pulse", "guarded_double_pulse",
    "protein_band", "balance_band",
)


def comparison_boundary(name):
    if name in ("either_count", "joint_or_high"):
        return fixed_count_boundary(name)
    if name in ("delayed_ramp", "step_release"):
        return relaxing_boundary(name)
    if name in ("guarded_pulse", "guarded_double_pulse"):
        return pulse_boundary(name)
    if name in ("protein_band", "balance_band"):
        return count_band_boundary(name)

    raise ValueError(f"Unknown comparison boundary: {name}")


############################################
# 六、多条件路线随时间变化（6种）
############################################

COUPLED_BOUNDARIES = (
    "shifting_ratio", "widening_ratio",
    "merging_routes", "swapping_routes",
    "coupled_downcross", "moving_downcross",
)


def coupled_boundary(name):
    def routes(m, n):
        total = m+n
        high = ((0.9*m+1.1*n-9.5, 0.),)

        if name == "shifting_ratio":
            residual = m-(2/3)*total
            drift = total/24
            return (
                (
                    (total-5, 0.),
                    (1.1-residual, -drift),
                    (1.1+residual, drift),
                ),
                high,
            )

        if name == "widening_ratio":
            residual = m-0.6*total
            width = 0.60+0.18*(total-5)
            return (
                (
                    (total-5, 0.),
                    (width-residual, 0.10),
                    (width+residual, 0.10),
                ),
            )

        if name == "merging_routes":
            left = ((m-4, 0.), (2-n, 0.))
            right = ((n-5, 0.), (2-m, 0.))
            bridge = (
                (m-2, 0.),
                (n-2, 0.),
                (total-10.5, 0.75),
            )
            return left, right, bridge, high

        if name == "swapping_routes":
            left = ((m-3, -0.375), (3-n, 0.))
            right = ((n-6, 0.375), (3-m, 0.))
            return left, right

        if name == "coupled_downcross":
            ceiling = 4.5-3*m/(m+4)
            return (
                (
                    (m-2, 0.),
                    (ceiling-n, 0.22),
                ),
            )

        if name == "moving_downcross":
            offset = 0.8*(m-2)
            return (
                (
                    (m-2, 0.),
                    (4.5-n-offset, 0.30),
                    (4.5-n+offset, -0.30),
                ),
                high,
            )

        raise ValueError(f"Unknown coupled boundary: {name}")

    return affine_time_boundary(routes)


############################################
# 七、窗口边界（6种）效果不好
############################################

# WINDOW_BOUNDARIES = (
#     "ratio_window", "tilted_ellipse", "parabolic_double",
#     "periodic_windows", "moving_ring", "opposing_bands",
# )


# def window_boundary(family, variant):
#     if not 0 <= family < len(WINDOW_BOUNDARIES):
#         raise ValueError("Unknown window family")

#     def intervals(m, n):
#         m = np.asarray(m, dtype=float)
#         n = np.asarray(n, dtype=float)
#         shift = .12*variant
#         active = m*n >= 3.5+variant

#         if family == 0:
#             center = 8*(n+1)/(m+n+2)+shift
#             bands = [(center-.55, center+.55)]

#         elif family in (1, 4):
#             if family == 1:
#                 x, y = (
#                     (m-.20*n-2.4)/1.9,
#                     (n+.20*m-2.0)/2.8,
#                 )
#                 vx, vy = .35/1.9, 1.05/2.8
#             else:
#                 x, y = (
#                     (m-.12*n-3.2)/2.6,
#                     (n+.12*m-4)/3.5,
#                 )
#                 vx, vy = .35/2.6, .9/3.5

#             speed = vx*vx+vy*vy
#             center = (x*vx+y*vy)/speed+shift
#             distance = (x*vy-y*vx)**2/speed
#             outer = np.sqrt(np.maximum(1-distance, 0)/speed)
#             active &= distance < 1
#             bands = [(center-outer, center+outer)]

#             if family == 4:
#                 inner = np.sqrt(
#                     np.maximum(.58**2-distance, 0)/speed
#                 )
#                 bands = [
#                     (center-outer, center-inner),
#                     (center+inner, center+outer),
#                 ]

#         elif family == 2:
#             center = 1+.11*(m-.5*n)**2+.17*n+shift
#             bands = [
#                 (center-.35, center+.35),
#                 (center+1.3, center+1.9),
#             ]

#         elif family == 3:
#             phase = .20*m+.13*n+shift
#             period = 1.8+.12*variant
#             phase = np.mod(phase, period)
#             width = .22+.04*variant
#             bands = [
#                 (
#                     phase+k*period-width,
#                     phase+k*period+width,
#                 )
#                 for k in range(-1, 7)
#             ]

#         else:
#             a = .45*m+.19*n+shift
#             b = 7-.22*m-.31*n-shift
#             active = (m+n >= 5.5+variant) & (m >= 1)
#             bands = [
#                 (a-.38, a+.38),
#                 (b-.27, b+.27),
#             ]

#         return [
#             (
#                 np.where(active, lo, np.inf),
#                 np.where(active, hi, -np.inf),
#             )
#             for lo, hi in bands
#         ]

#     def g(t, m, n):
#         scores = [
#             np.minimum(t-lo, hi-t)
#             for lo, hi in intervals(m, n)
#         ]
#         value = np.maximum.reduce(scores)
#         return np.where(np.isfinite(value), value, -1.)

#     return FirstEvent(g)