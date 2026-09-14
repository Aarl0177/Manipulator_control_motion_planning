"""
UR5e pick-and-place in MuJoCo with feedback-linearizing joint control.

Control law (per joint):
    v   = qdd_ref - Kd*(qd - qd_ref) - Kp*(q - q_ref)
    tau = M(q) @ v + bias(q, qd) - passive(q, qd)

Six torque motors drive the arm. IK finds joint angles for each Cartesian
waypoint; quintic interpolation gives smooth q_ref/qd_ref/qdd_ref between
them. The gripper is an idealized suction cup (an on/off weld constraint),
not a friction-based grasp.
"""

import argparse
import time
from contextlib import ExitStack
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import imageio.v2 as imageio
import scipy.optimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import make_interp_spline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
DT = 0.005
KP = 100 * np.eye(6)
KD = 20 * np.eye(6)
TORQUE_LIMIT = np.array([150., 150., 150., 28., 28., 28.])

CENTER = np.array([0.55, 0.0, 0.30])  # metres
RADIUS = 0.1                       # metres

HOLD_START = 0
CIRCLE_TIME = 12.0
HOLD_END = 0

TOTAL_TIME = HOLD_START + CIRCLE_TIME + HOLD_END

def build_scene():
    tree = ET.parse(ROOT / "model" / "ur5e_original.xml")
    root = tree.getroot()
    root.find("option").set("timestep", str(DT))
    root.find("option").set("integrator", "implicitfast")
    root.remove(root.find("actuator"))
    root.remove(root.find("keyframe"))

    actuator = ET.SubElement(root, "actuator")
    for name, limit in zip(JOINTS, TORQUE_LIMIT):
        ET.SubElement(actuator, "motor", name=name + "_motor", joint=name,
                      ctrllimited="true", ctrlrange=f"{-limit} {limit}")

    world = root.find("worldbody")
    ET.SubElement(world, "geom", type="plane", size="2 2 0.1")
    ET.SubElement(world, "light", pos="1 1 3")

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")

    # Static visual of the desired circular path (no collision, no dynamics).
    n_markers = 60
    for i in range(n_markers):
        theta = 2 * np.pi * i / n_markers
        pos = CENTER + np.array([RADIUS*np.cos(theta), RADIUS*np.sin(theta), 0.0])
        ET.SubElement(world, "site", name=f"path_marker_{i}", type="sphere",
                      pos=" ".join(map(str, pos)), size="0.003",
                      rgba="0.2 0.85 0.45 0.6")

    wrist = root.find(".//body[@name='wrist_3_link']")
    cup = ET.SubElement(wrist, "body", name="suction_cup",
                         pos="0 0.1 0", quat="-1 1 0 0")
    ET.SubElement(cup, "geom", type="cylinder", pos="0 0 0.025",
                  size="0.022 0.025", mass="0.05", contype="0", conaffinity="0")
    ET.SubElement(cup, "site", name="tcp", pos="0 0 0.05", size="0.004")

    path = ROOT / "model" / "trajectory_scene.xml"
    tree.write(path, encoding="unicode")
    return path


def indices(model):
    """Map joint names to their qpos/qvel array indices."""
    ids = [model.joint(name).id for name in JOINTS]
    return model.jnt_qposadr[ids], model.jnt_dofadr[ids]


def full_mass(model, data, M):
    # MuJoCo changed this Python signature after the tested 3.3.7 release.
    try:
        mujoco.mj_fullM(model, M, data.qM)
    except TypeError:
        mujoco.mj_fullM(model, data, M)


def inverse_kinematics(model, data, q_index, target_pos, seed):
    site = model.site("tcp").id
    down = np.diag([1., -1., -1.])

    def residual(q):
        data.qpos[q_index] = q
        mujoco.mj_forward(model, data)
        pos_err = data.site_xpos[site] - target_pos
        rot_err = Rotation.from_matrix(
            down @ data.site_xmat[site].reshape(3, 3).T).as_rotvec()
        return np.r_[pos_err, 0.3 * rot_err]

    bounds = np.array([model.jnt_range[model.joint(n).id] for n in JOINTS])
    sol = scipy.optimize.least_squares(residual, seed,
                                        bounds=(bounds[:, 0], bounds[:, 1]))
    return sol.x

def circle_position(t):
    s = np.clip(t / CIRCLE_TIME, 0.0, 1.0)
    h = 10*s**3 - 15*s**4 + 6*s**5
    theta = 2*np.pi*h
    return CENTER + np.array([RADIUS*np.cos(theta), RADIUS*np.sin(theta), 0.0])

def desired_position(t):
    if t < HOLD_START:
        return circle_position(0.0)
    elif t < HOLD_START + CIRCLE_TIME:
        return circle_position(t - HOLD_START)
    else:
        return circle_position(CIRCLE_TIME)


def build_joint_reference(model):
    ik_data = mujoco.MjData(model)
    qi, _ = indices(model)
    number_of_steps = round(CIRCLE_TIME / DT)
    times = np.linspace(0.0, CIRCLE_TIME, number_of_steps + 1)
    joint_samples = np.zeros((len(times), 6))
    seed = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
    for i, t in enumerate(times):
        p_desired = circle_position(t)
        q_solution = inverse_kinematics(model, ik_data, qi, p_desired, seed)
        if i > 0:
            change = np.max(np.abs(q_solution - joint_samples[i - 1]))
            if change > 0.15:
                raise RuntimeError(f"Large IK jump near t={t:.3f} s.")
        joint_samples[i] = q_solution
        seed = q_solution.copy()
        if i % 500 == 0:
            print(f"Preparing IK: {i}/{number_of_steps}")
    zero = np.zeros(6)
    spline = make_interp_spline(times, joint_samples, k=5,
        bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
    return spline


def joint_reference(t, spline):
    zero = np.zeros(6)
    if t < HOLD_START:
        return spline(0.0), zero.copy(), zero.copy()
    elif t < HOLD_START + CIRCLE_TIME:
        local_time = t - HOLD_START
        return spline(local_time), spline(local_time, 1), spline(local_time, 2)
    else:
        return spline(CIRCLE_TIME), zero.copy(), zero.copy()

def quintic(q0, q1, t, duration):
    """Smooth interpolation with zero velocity/acceleration at both ends."""
    s = np.clip(t / duration, 0., 1.)
    h = 10*s**3 - 15*s**4 + 6*s**5
    hd = (30*s**2 - 60*s**3 + 30*s**4) / duration
    hdd = (60*s - 180*s**2 + 120*s**3) / duration**2
    delta = q1 - q0
    return q0 + h*delta, hd*delta, hdd*delta


def torque_command(model, data, q_index, v_index, q_ref, qd_ref, qdd_ref):
    q, qd = data.qpos[q_index], data.qvel[v_index]
    v = qdd_ref - KD @ (qd - qd_ref) - KP @ (q - q_ref)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    M_arm = M[np.ix_(v_index, v_index)]
    tau = M_arm @ v + data.qfrc_bias[v_index] - data.qfrc_passive[v_index]
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


def run_phase(model, data, q_index, v_index, motor_ids, q_start, q_end,
              duration, viewer=None):
    for step in range(round(duration / DT)):
        q_ref, qd_ref, qdd_ref = quintic(q_start, q_end, step * DT, duration)
        mujoco.mj_forward(model, data)
        tau = torque_command(model, data, q_index, v_index, q_ref, qd_ref, qdd_ref)
        data.ctrl[motor_ids] = tau
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def add_trace_marker(scene, pos, rgba=(1.0, 0.25, 0.1, 1.0), size=0.004):
    """Append one small sphere to an MjvScene. Works for both viewer.user_scn
    (persists across frames - add once per new point) and renderer.scene
    (rebuilt every update_scene() call - re-add the whole trace each frame)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                         size=np.array([size, 0, 0]), pos=np.asarray(pos),
                         mat=np.eye(3).flatten(),
                         rgba=np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    out = ROOT / "trajectory_outputs"
    out.mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    qi, vi = indices(model)
    motor_ids = [model.actuator(name + "_motor").id for name in JOINTS]
    tcp = model.site("tcp").id
    M = np.zeros((model.nv, model.nv))

    spline = build_joint_reference(model)
    q_initial, _, _ = joint_reference(0.0, spline)
    data.qpos[qi] = q_initial
    data.qvel[vi] = 0.0
    mujoco.mj_forward(model, data)

    reference_data = mujoco.MjData(model)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.25, 0.0, 0.25]
    camera.distance = 1.65
    camera.azimuth = 135
    camera.elevation = -25

    rows = []
    trace_points = []
    trace_every = round(0.1 / DT)  # one marker every 0.1 s of sim time
    next_frame = 0.0

    with ExitStack() as stack:
        viewer = None
        if not args.headless:
            from mujoco import viewer as mjviewer
            viewer = stack.enter_context(mjviewer.launch_passive(model, data))
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation

        renderer = None
        writer = None
        if not args.no_video:
            renderer = mujoco.Renderer(model, height=720, width=960)
            stack.callback(renderer.close)
            writer = imageio.get_writer(out / "trajectory.mp4", fps=30,
                                        codec="libx264", quality=8)
            stack.callback(writer.close)

        wall_start = time.perf_counter()
        number_of_steps = round(TOTAL_TIME / DT)

        for step in range(number_of_steps):
            if viewer is not None and not viewer.is_running():
                raise RuntimeError("Viewer closed before completion.")
            t = data.time
            p_desired = desired_position(t)
            q_ref, qd_ref, qdd_ref = joint_reference(t, spline)
            mujoco.mj_forward(model, data)
            q = data.qpos[qi].copy()
            qd = data.qvel[vi].copy()
            full_mass(model, data, M)
            M_arm = M[np.ix_(vi, vi)]
            bias = data.qfrc_bias[vi].copy()
            passive = data.qfrc_passive[vi].copy()
            e = q - q_ref
            ed = qd - qd_ref
            v = qdd_ref - KD @ ed - KP @ e
            tau = M_arm @ v + bias - passive
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
            data.ctrl[motor_ids] = tau
            p_actual = data.site_xpos[tcp].copy()
            if step % trace_every == 0:
                trace_points.append(p_actual.copy())
                if viewer is not None:
                    add_trace_marker(viewer.user_scn, p_actual)
            reference_data.qpos[qi] = q_ref
            mujoco.mj_forward(model, reference_data)
            p_from_joint_reference = reference_data.site_xpos[tcp].copy()
            if step % 5 == 0:
                rows.append(np.r_[t, q, q_ref, tau, p_actual, p_desired, p_from_joint_reference])
            if renderer is not None and t + 1e-9 >= next_frame:
                renderer.update_scene(data, camera=camera)
                for p in trace_points:
                    add_trace_marker(renderer.scene, p)
                writer.append_data(renderer.render())
                next_frame += 1.0 / 30.0
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Simulation became non-finite.")
            if viewer is not None:
                if step % 10 == 0:
                    viewer.sync()
                remaining = wall_start + data.time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    save_results(out, rows)
    print(f"Saved results in {out}")


def save_results(out, rows):
    a = np.asarray(rows)
    t = a[:, 0]
    q = a[:, 1:7]
    qr = a[:, 7:13]
    tau = a[:, 13:19]
    p_actual = a[:, 19:22]
    p_desired = a[:, 22:25]
    p_reference = a[:, 25:28]
    names = (["time"] + [f"q{i+1}" for i in range(6)] + [f"qref{i+1}" for i in range(6)]
             + [f"tau{i+1}" for i in range(6)] + ["actual_x", "actual_y", "actual_z"]
             + ["desired_x", "desired_y", "desired_z"] + ["reference_x", "reference_y", "reference_z"])
    np.savetxt(out / "results.csv", a, delimiter=",", header=",".join(names), comments="")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, p_actual[:, j], label="Actual")
        ax.plot(t, p_desired[:, j], "--", label="Desired")
        ax.set_ylabel(f"{'xyz'[j]} (m)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out / "cartesian_tracking.png", dpi=160)
    plt.close(fig)

    error = p_actual - p_desired
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, label in enumerate("xyz"):
        ax.plot(t, 1000*error[:, j], label=f"{label} error")
    ax.set(xlabel="Time (s)", ylabel="Position error (mm)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cartesian_error.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for j in range(6):
        axes[0].plot(t, np.rad2deg(q[:, j] - qr[:, j]), label=f"J{j+1}")
        axes[1].plot(t, tau[:, j], label=f"J{j+1}")
    axes[0].set_ylabel("Joint error (deg)")
    axes[1].set_ylabel("Torque (N m)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(ncol=6)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "joint_errors_torques.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*p_actual.T, label="Actual tool path")
    ax.plot(*p_desired.T, "--", label="Requested circle")
    ax.plot(*p_reference.T, ":", label="FK of joint reference")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tool_path.png", dpi=160)
    plt.close(fig)

    tracking_error = np.linalg.norm(error, axis=1)
    reference_error = np.linalg.norm(p_reference - p_desired, axis=1)
    print("RMS tool position error:", 1000*np.sqrt(np.mean(tracking_error**2)), "mm")
    print("Maximum tool position error:", 1000*np.max(tracking_error), "mm")
    print("Maximum sampled reference-generation error:", 1000*np.max(reference_error), "mm")


if __name__ == "__main__":
    main()
