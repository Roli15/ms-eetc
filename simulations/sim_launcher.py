import numpy as np
from matplotlib import pyplot as plt

from mseetc.efficiency import totalLossesFunction
from mseetc.etcs import EtcsBrakingCurveCalculator, BrakingTarget, trimCurveToMaxVelocity
from mseetc.ocp import casadiSolver


def get_power_loss_function(train, mode="perfect",* ,auxiliaries: float = 27_000, eta_gear: float = 0.96):

    if mode == "perfect":

        return lambda f, v: 0

    elif mode == "static":

        return lambda f, v: (f>0)*f*v*(1-train.etaTraction)/train.etaTraction - (f<0)*f*v*(1-train.etaRgBrake)

    elif mode == "dynamic":

        return totalLossesFunction(train, auxiliaries=auxiliaries, etaGear=eta_gear)

    else:

        raise ValueError("mode must be one of: 'perfect', 'static', 'dynamic'")


def getTrackVelocityAtPositions(speedLimitPositions, speedLimits, positions):
    """
    Return stepwise track speed limit at given positions.
    Assumes speedLimitPositions are sorted increasingly.
    """

    indices = np.searchsorted(speedLimitPositions, positions, side="right") - 1
    indices = np.clip(indices, 0, len(speedLimits) - 1)

    return speedLimits[indices]


def getBrakingTargetsFromSpeedLimits(track):

    speedLimitPositions = track.speedLimits.index.to_numpy(dtype=float)
    speedLimits = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)

    # Add final stop target at end of track
    speedLimitPositions = np.append(speedLimitPositions, track.length)
    speedLimits = np.append(speedLimits, 0.0)

    v_max = max(speedLimits)

    targets = []

    for idx in range(1, len(speedLimitPositions)):

        previousVelocity = speedLimits[idx - 1]
        targetVelocity = speedLimits[idx]

        # Only speed decreases require a braking curve
        if targetVelocity < previousVelocity:

            target = BrakingTarget(
                position=speedLimitPositions[idx],
                overlap=100,
                permittedVelocity=v_max,
                targetVelocity=targetVelocity,
            )

            targets.append(target)

    return targets, speedLimitPositions, speedLimits


def getEtcsSpeedLimits2(track, trainBrakingData, positionStep=10.0):

    calculator = EtcsBrakingCurveCalculator(trainBrakingData, track)

    targets, speedLimitPositions, speedLimits = getBrakingTargetsFromSpeedLimits(track)

    # Compute P curves for all speed decreases
    pCurves = []

    for target in targets:

        curveSet = calculator.computeTarget(target)
        curveSet["P"].loc[target.position, "Velocity [m/s]"] = target.targetVelocity
        pCurves.append(curveSet["P"])

    # Build common position grid
    positions = np.arange(0.0, track.length + positionStep, positionStep)

    # Start with the ordinary track speed limit
    etcsVelocities = getTrackVelocityAtPositions(speedLimitPositions, speedLimits, positions)

    # Apply every ETCS P curve as an additional restriction
    for pCurve in pCurves:

        curvePositions = pCurve.index.to_numpy(dtype=float)
        curveVelocities = pCurve["Velocity [m/s]"].to_numpy(dtype=float)

        mask = ((curvePositions.min() <= positions) & (positions <= curvePositions.max()))

        interpolatedCurveVelocities = np.interp(positions[mask], curvePositions, curveVelocities)

        etcsVelocities[mask] = np.minimum(etcsVelocities[mask], interpolatedCurveVelocities)

    return positions, etcsVelocities



def getEtcsSpeedLimits(track, trainBrakingData):

    etcsLimitsPositions = []
    etcsLimitsVelocities = []

    speedLimitPositions = track.speedLimits.index.to_numpy(dtype=float)
    speedLimits = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)
    speedLimitPositions = np.append(speedLimitPositions, track.length)
    speedLimits = np.append(speedLimits, 0)

    v_max = max(speedLimits)

    calculator = EtcsBrakingCurveCalculator(trainBrakingData, track, distancePre=5000, distancePost=1000)

    rev_idx = len(speedLimitPositions) - 1

    while rev_idx > 0:

        if speedLimits[rev_idx-1] < speedLimits[rev_idx]:

            # speed limit increase -> no breaking curve needed
            if etcsLimitsPositions[0] - 20 < speedLimitPositions[rev_idx] < etcsLimitsPositions[0] + 20:

                etcsLimitsPositions = np.concatenate([np.array([speedLimitPositions[rev_idx]]), etcsLimitsPositions])
                etcsLimitsVelocities = np.concatenate([np.array([speedLimits[rev_idx-1]]), etcsLimitsVelocities])

            else:

                etcsLimitsPositions = np.concatenate([np.array([speedLimitPositions[rev_idx], speedLimitPositions[rev_idx]+1]), etcsLimitsPositions])
                etcsLimitsVelocities = np.concatenate([np.array([speedLimits[rev_idx-1], speedLimits[rev_idx]]), etcsLimitsVelocities])

            rev_idx = rev_idx - 1
            continue

        # compute braking curve
        target = BrakingTarget(
            position=speedLimitPositions[rev_idx],
            overlap=100,
            permittedVelocity=v_max,
            targetVelocity=speedLimits[rev_idx]
        )

        curve_set = calculator.computeTarget(target)
        curve = curve_set["P"]

        curvePositions = curve.index.to_numpy(dtype=float)
        curveVelocities = curve["Velocity [m/s]"].to_numpy(dtype=float)

        # find intersection with track speed limit profile
        found = False

        while not found:

            sectionStart = speedLimitPositions[rev_idx-1]

            # case 1: braking curve intersects speed limit in current section
            positionAtVLimit = np.interp(speedLimits[rev_idx-1], curveVelocities[::-1], curvePositions[::-1])

            if sectionStart < positionAtVLimit:

                curve = trimCurveToMaxVelocity(curve, speedLimits[rev_idx-1])

                curvePositions = curve.index.to_numpy(dtype=float)
                curveVelocities = curve["Velocity [m/s]"].to_numpy(dtype=float)

                etcsLimitsPositions = np.concatenate([np.array([curvePositions[0]-1]), curvePositions, etcsLimitsPositions])
                etcsLimitsVelocities = np.concatenate([np.array([speedLimits[rev_idx-1]]), curveVelocities, etcsLimitsVelocities])

                rev_idx = rev_idx - 1
                break

            # case 2: speed limit increases at sectionStart and braking curve intersects speed increase
            if speedLimits[rev_idx-2] < speedLimits[rev_idx-1]:

                # speed limit increases at sectionStart
                vAtSectionStart = np.interp(speedLimitPositions[rev_idx-1], curvePositions, curveVelocities)

                if speedLimits[rev_idx-2] < vAtSectionStart < speedLimits[rev_idx-1]:

                    # braking curve intersects speed increase
                    curve = trimCurveToMaxVelocity(curve, vAtSectionStart)

                    curvePositions = curve.index.to_numpy(dtype=float)
                    curveVelocities = curve["Velocity [m/s]"].to_numpy(dtype=float)

                    etcsLimitsPositions = np.concatenate([curvePositions, etcsLimitsPositions])
                    etcsLimitsVelocities = np.concatenate([curveVelocities, etcsLimitsVelocities])

                    rev_idx = rev_idx - 1
                    break

            rev_idx = rev_idx - 1

    etcsLimitsPositions = np.concatenate([np.array([0]), etcsLimitsPositions])
    etcsLimitsVelocities = np.concatenate([np.array([speedLimits[0]]), etcsLimitsVelocities])
    return etcsLimitsPositions, etcsLimitsVelocities


if __name__ == '__main__':

    from mseetc.train import Train
    from mseetc.track import Track

    # Timetable
    startPosition = 0       # [m]
    endPosition = 10000     # [m]
    duration = (endPosition-startPosition) / (90/3.6)       # [s]

    train = Train(config={'id':'CH_Stadler_FLIRT_TPF'}, pathJSON='../trains')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    track = Track(config={'id':'00_var_speed_limit_quick_change'}, pathJSON='../tracks')
    # track = Track(config={'id':'CH_StGallen_Wil'}, pathJSON='../tracks')
    track.updateTrainLengthDependentValues(train)
    track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

    trainBrakingData = {
        "A_brake_emergency [m/s^2]": {
            "velocity [m/s]": [0, 20, 40, 60],
            "value [m/s^2]": [-0.9, -0.85, -0.8, -0.75],
        },
        "A_brake_service [m/s^2]": {
            "velocity [m/s]": [0, 20, 40, 60],
            "value [m/s^2]": [-0.5, -0.45, -0.4, -0.35],
        },
        "K_dry_rst [-]": 0.8,
        "M_NVAVADH [-]": 0,
        "K_wet_rst [-]": 0.9,
        "T_traction [s]": 1,
        "T_be [s]": 4,
        "Kt_int [-]": 1.15,
        "v_uncertainty [%]": 2.98,
        "T_bs [s]": 3,
        "T_bs1 [s]": 3,
        "T_bs2 [s]": 3,
    }

    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    # solver = casadiSolver(train, track, opts)
    #
    # df, stats = solver.solve(duration)
    #
    # # print some info
    # if df is not None:
    #
    #     print("")
    #     print("Objective value = {:.2f} {}".format(stats['Cost'], 'kWh' if solver.opts.energyOptimal else 's'))
    #     print("")
    #     print("Maximum acceleration: {:5.2f}, with bound {}".format(df.max()['Acceleration [m/s^2]'], train.accMax if train.accMax is not None else 'None'))
    #     print("Maximum deceleration: {:5.2f}, with bound {}".format(df.min()['Acceleration [m/s^2]'], train.accMin if train.accMin is not None else 'None'))
    #
    # else:
    #
    #     print("Solver failed!")


    ### Plot Trajectory

    fig, ax = plt.subplots(figsize=(16, 8))

    x = track.speedLimits.index.to_numpy(dtype=float)
    v = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)
    x_plot = np.append(x, track.length)
    v_plot = np.append(v, v[-1])

    etcsLimitsPositions, etcsLimitsVelocities = getEtcsSpeedLimits2(track, trainBrakingData)

    ax.step(x_plot/1000, v_plot*3.6, where="post", label="Track Speed Limit")
    ax.plot(etcsLimitsPositions/1000, etcsLimitsVelocities*3.6, label="ETCS Speed Limit")
    # ax.plot(df["Position [m]"] / 1000, df["Velocity [m/s]"] * 3.6, label="non-adjusted speed profile")
    ax.set_title("Speed Profile Comparison")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    # ax.set_xlim(0, df["Position [m]"].max() / 1000)
    ax.figure.tight_layout()

    plt.show()