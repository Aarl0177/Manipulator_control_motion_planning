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

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import scipy.optimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
DT = 0.005
KP = 100 * np.eye(6)
KD = 20 * np.eye(6)
TORQUE_LIMIT = np.array([150., 150., 150., 28., 28., 28.])
PICK = np.array([0.45, -0.20, 0.025])   # cube center, metres
PLACE = np.array([0.45, 0.20, 0.025])
CUBE_HALF = 0.025


def build_scene():
    """Load the stock UR5e, swap its position servos for torque motors, and
    add a suction cup, a cube to pick up, and the weld used to "grasp" it."""
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

    wrist = root.find(".//body[@name='wrist_3_link']")
    cup = ET.SubElement(wrist, "body", name="suction_cup",
                         pos="0 0.1 0", quat="-1 1 0 0")
    ET.SubElement(cup, "geom", type="cylinder", pos="0 0 0.025",
                  size="0.022 0.025", mass="0.05", contype="0", conaffinity="0")
    ET.SubElement(cup, "site", name="tcp", pos="0 0 0.05", size="0.004")

    # Flat visual target; it does not collide with the cube.
    ET.SubElement(world, "site", name="place_marker", type="box",
                    pos="0.45 0.20 0.001", size="0.06 0.06 0.001",
                    rgba="0.2 0.85 0.45 0.5")
    cube = ET.SubElement(world, "body", name="cube",
                          pos=" ".join(map(str, PICK)))
    ET.SubElement(cube, "freejoint")
    ET.SubElement(cube, "geom", name="cube_geom", type="box",
                  size="0.025 0.025 0.025", mass="0.04",
                  rgba="1 0.55 0.12 1", friction="1 0.01 0.001")

    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="suction", body1="suction_cup",
                  body2="cube", active="false", solref="0.01 1")

    path = ROOT / "model" / "pick_place_scene.xml"
    tree.write(path, encoding="unicode")
    return path


def inverse_kinematics(model, data, q_index, target_pos, seed):
    """Solve for joint angles that place the TCP at target_pos, pointing down."""
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


def quintic(q0, q1, t, duration):
    """Smooth interpolation with zero velocity/acceleration at both ends."""
    s = np.clip(t / duration, 0., 1.)
    h = 10*s**3 - 15*s**4 + 6*s**5
    hd = (30*s**2 - 60*s**3 + 30*s**4) / duration
    hdd = (60*s - 180*s**2 + 120*s**3) / duration**2
    delta = q1 - q0
    return q0 + h*delta, hd*delta, hdd*delta


def torque_command(model, data, q_index, v_index, q_ref, qd_ref, qdd_ref):
    """Feedback-linearizing torque: tau = M @ v + bias - passive."""
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


def attach_cube(model, data):
    """Weld the cube to the suction cup at its current relative pose."""
    cup, cube = model.body("suction_cup").id, model.body("cube").id
    cup_R = data.xmat[cup].reshape(3, 3)
    cube_R = data.xmat[cube].reshape(3, 3)
    rel_pos = cup_R.T @ (data.xpos[cube] - data.xpos[cup])
    rel_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rel_quat, (cup_R.T @ cube_R).ravel())
    eq = model.equality("suction").id
    model.eq_data[eq, 3:6] = rel_pos
    model.eq_data[eq, 6:10] = rel_quat
    data.eq_active[eq] = 1


def main():
    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)  # scratch data so IK doesn't disturb the sim

    q_index = model.jnt_qposadr[[model.joint(n).id for n in JOINTS]]
    v_index = model.jnt_dofadr[[model.joint(n).id for n in JOINTS]]
    motor_ids = [model.actuator(n + "_motor").id for n in JOINTS]
    suction = model.equality("suction").id

    pick_top = PICK + [0, 0, CUBE_HALF]
    place_top = PLACE + [0, 0, CUBE_HALF + 0.003]
    pick_above = pick_top + [0, 0, 0.22]
    place_above = place_top + [0, 0, 0.22]

    seed = np.array([-1.57, -1.57, 1.57, np.pi/4, -1.57, 0.])
    q_home = inverse_kinematics(model, ik_data, q_index, pick_above, seed)
    q_pick = inverse_kinematics(model, ik_data, q_index, pick_top, q_home)
    q_place_above = inverse_kinematics(model, ik_data, q_index, place_above, q_home)
    q_place = inverse_kinematics(model, ik_data, q_index, place_top, q_place_above)

    data.qpos[q_index] = q_home
    mujoco.mj_forward(model, data)

    # (target joint angles, duration in seconds, action to take on arrival)
    phases = [
        (q_home, 1.0, None),
        (q_pick, 3.0, None),
        (q_pick, 0.6, "attach"),
        (q_home, 3.0, None),
        (q_place_above, 4.0, None),
        (q_place, 3.0, None),
        (q_place, 0.6, "release"),
        (q_place_above, 3.0, None),
    ]

    from mujoco import viewer as mjviewer
    with mjviewer.launch_passive(model, data) as viewer:
        q_start = q_home.copy()
        for q_end, duration, action in phases:
            run_phase(model, data, q_index, v_index, motor_ids,
                      q_start, q_end, duration, viewer)
            if action == "attach":
                attach_cube(model, data)
            elif action == "release":
                data.eq_active[suction] = 0
            q_start = q_end.copy()

    print("Final cube position:", data.xpos[model.body("cube").id])


if __name__ == "__main__":
    main()
