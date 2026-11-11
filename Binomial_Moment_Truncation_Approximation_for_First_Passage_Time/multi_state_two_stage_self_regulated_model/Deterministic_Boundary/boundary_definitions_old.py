"""批跑使用的边界函数与对应的解析首达区间。"""

import numpy as np

import ssa
from ssa import FirstEvent, affine_time_boundary, exponential_boundary


def interval_first_hit(windows):
    return ssa.interval_first_hit(windows)


NEW_BOUNDARIES = (
    "either_count", "joint_or_high", "delayed_ramp", "step_release",
    "guarded_pulse", "guarded_double_pulse", "protein_band", "balance_band",
)
COUPLED_BOUNDARIES = (
    "shifting_ratio", "widening_ratio", "merging_routes", "swapping_routes",
    "coupled_downcross", "moving_downcross",
)


WINDOW_BOUNDARIES = (
    "ratio_window", "tilted_ellipse", "parabolic_double",
    "periodic_windows", "moving_ring", "opposing_bands",
)


def window_boundary(family, variant):
    if not 0 <= family < len(WINDOW_BOUNDARIES):
        raise ValueError("Unknown window family")

    def windows(m,n):
        m=np.asarray(m,dtype=float); n=np.asarray(n,dtype=float)
        shift=.12*variant
        active=m*n >= 3.5+variant
        if family==0: # 联合比例控制的时间窗
            center=8*(n+1)/(m+n+2)+shift
            bands=[(center-.55,center+.55)]
        elif family in (1, 4):  # 移动椭圆和环带共用同一套交点计算。
            if family == 1:
                x, y = (m-.20*n-2.4)/1.9, (n+.20*m-2.0)/2.8
                vx, vy = .35/1.9, 1.05/2.8
            else:
                x, y = (m-.12*n-3.2)/2.6, (n+.12*m-4)/3.5
                vx, vy = .35/2.6, .9/3.5
            speed = vx*vx+vy*vy
            center = (x*vx+y*vy)/speed+shift
            distance = (x*vy-y*vx)**2/speed
            outer = np.sqrt(np.maximum(1-distance,0)/speed)
            active &= distance < 1
            bands = [(center-outer,center+outer)]
            if family == 4:
                inner = np.sqrt(np.maximum(.58**2-distance,0)/speed)
                bands = [(center-outer,center-inner),(center+inner,center+outer)]
        elif family==2: # 抛物线脊线上的双窗口
            center=1+.11*(m-.5*n)**2+.17*n+shift
            bands=[(center-.35,center+.35),(center+1.3,center+1.9)]
        elif family==3: # 周期性联合反馈窗口
            phase=.20*m+.13*n+shift
            period=1.8+.12*variant
            phase=np.mod(phase,period)
            width=.22+.04*variant
            bands=[(phase+k*period-width,phase+k*period+width)
                   for k in range(-1,7)]
        else: # 双向扫过的联合计数条带
            a=.45*m+.19*n+shift
            b=7-.22*m-.31*n-shift
            active=(m+n>=5.5+variant) & (m>=1)
            bands=[(a-.38,a+.38),(b-.27,b+.27)]
        return [(np.where(active,lo,np.inf),np.where(active,hi,-np.inf))
                for lo,hi in bands]

    return FirstEvent.from_windows(windows)


def smooth_presets():
    return {
        "product": (lambda m, n: (m + 1) * (n + 1), 6., 5., 3.),
        "weighted": (lambda m, n: 1 + .8*m + 1.35*n + .30*m*n, 5., 4., 3.5),
        "saturating": (lambda m, n: 1 + 1.25*m + .85*n + 2*m*n/(1+m+n), 5., 4., 2.5),
        "curved": (lambda m, n: 1 + .9*m + .8*n + .18*(m+n)**2, 5.5, 4.5, 4.),
    }


SMOOTH_BOUNDARIES = smooth_presets()


def smooth_boundary_pair(name):
    return exponential_boundary(*SMOOTH_BOUNDARIES[name])


def comparison_boundary(name):
    if name in ("either_count", "joint_or_high", "protein_band", "balance_band"):
        def g(t, m, n):
            if name == "either_count":
                return (np.maximum(2*m, 1.5*n) - 6) / 4
            high = (13*m + 8*n - 80) / 10
            if name == "joint_or_high":
                score = np.minimum(2*m, 1.5*n) - 3
            elif name == "protein_band":
                score = np.minimum.reduce((m-1, n-2, 4-n))
            else:
                score = np.minimum(m+n-4, 1-np.abs(m-n))
            return np.maximum(score, high) / 4
        def windows(m, n):
            hit = g(0., m, n) >= 0
            return [(np.where(hit, 0., np.inf), np.where(hit, np.inf, -np.inf))]
    elif name in ("delayed_ramp", "step_release"):
        ramp = name == "delayed_ramp"

        def ramp_opening(score):
            return 1.5 + (80-score)/20
        def g(t, m, n):
            score = 11*m + 9*n
            if ramp:
                return np.clip((t-ramp_opening(score))/2, (score-80)/40, (score-40)/40)
            return (score - np.where(t < 2.5, 80, 48)) / 40
        def windows(m, n):
            score = 11*m + 9*n
            eligible = score >= (40 if ramp else 48)
            opening = ramp_opening(score) if ramp else 2.5
            start = np.where(score >= 80, 0., opening)
            return [(np.where(eligible, start, np.inf), np.inf)]
    elif name in ("guarded_pulse", "guarded_double_pulse"):
        centers = (3.,) if name == "guarded_pulse" else (2., 5.)

        def pulse(score, center):
            half_width = 1.5*(score - 50.5)/29.5
            return center-half_width, center+half_width
        def g(t, m, n):
            score = 9*m + 11*n
            value = (score-80)/40
            for center in centers:
                start, end = pulse(score, center)
                value = np.maximum(value, (29.5/60)*np.minimum(t-start, end-t))
            return value
        def windows(m, n):
            score = 9*m + 11*n
            bands = [(np.where(score >= 80, 0., np.inf), np.inf)]
            temporary = (score > 50.5) & (score < 80)
            for center in centers:
                lo, hi = pulse(score, center)
                bands.append((np.where(temporary, lo, np.inf), np.where(temporary, hi, -np.inf)))
            return bands
    else:
        raise ValueError(f"Unknown comparison boundary: {name}")

    return FirstEvent(g, interval_first_hit(windows))


def coupled_boundary(name):
    def routes(m, n):
        total = m+n
        high = ((0.9*m+1.1*n-9.5, 0.),)
        if name == "shifting_ratio":
            residual = m-(2/3)*total
            drift = total/24
            return (((total-5, 0.), (1.1-residual, -drift), (1.1+residual, drift)), high)
        if name == "widening_ratio":
            residual = m-0.6*total
            width = 0.60+0.18*(total-5)
            return (((total-5, 0.), (width-residual, 0.10), (width+residual, 0.10)),)
        if name == "merging_routes":
            left = ((m-4, 0.), (2-n, 0.))
            right = ((n-5, 0.), (2-m, 0.))
            bridge = ((m-2, 0.), (n-2, 0.), (total-10.5, 0.75))
            return left, right, bridge, high
        if name == "swapping_routes":
            left = ((m-3, -0.375), (3-n, 0.))
            right = ((n-6, 0.375), (3-m, 0.))
            return left, right
        if name == "coupled_downcross":
            ceiling = 4.5-3*m/(m+4)
            return (((m-2, 0.), (ceiling-n, 0.22)),)
        if name == "moving_downcross":
            offset = 0.8*(m-2)
            return (((m-2, 0.), (4.5-n-offset, 0.30), (4.5-n+offset, -0.30)), high)
        raise ValueError(f"Unknown coupled boundary: {name}")

    return affine_time_boundary(routes)
