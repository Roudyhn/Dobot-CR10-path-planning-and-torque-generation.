import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
from cr10_kinematics import CR10Kinematics


def damped_pinv(J, damping=0.1):
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(J.shape[0]))


def time_scaling(t, T):
    r = t / T
    s = 10*r**3 - 15*r**4 + 6*r**5
    sdot = (30*r**2 - 60*r**3 + 30*r**4) / T
    sddot = (60*r - 180*r**2 + 120*r**3) / T**2
    return s, sdot, sddot


def circle_trajectory(t, T, center, R):
    s, sdot, sddot = time_scaling(t, T)

    theta = 2*np.pi*s
    theta_dot = 2*np.pi*sdot
    theta_ddot = 2*np.pi*sddot

    x = np.array([
        center[0] + R*np.cos(theta),
        center[1] + R*np.sin(theta),
        center[2]
    ])

    xdot = np.array([
        -R*np.sin(theta)*theta_dot,
         R*np.cos(theta)*theta_dot,
         0.0
    ])

    xddot = np.array([
        -R*np.cos(theta)*theta_dot**2 - R*np.sin(theta)*theta_ddot,
        -R*np.sin(theta)*theta_dot**2 + R*np.cos(theta)*theta_ddot,
         0.0
    ])

    return x, xdot, xddot


def plot_joint_data(time, data, ylabel, title):
    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(time, data[:, i], label=f"Joint {i+1}")

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)


ik = CR10Kinematics()

model = pin.buildModelFromUrdf("urdf/project.urdf")
data = model.createData()
frame_id = model.getFrameId("Link6")

T = 6.0
dt = 0.01
time = np.arange(0, T + dt, dt)

center = np.array([0.45, 0.00, 0.40])
R = 0.08

qs = []
qdots = []
qddots = []
taus = []

previous_q = None

for t in time:
    x, xdot, xddot = circle_trajectory(t, T, center, R)

    q = np.array(
        ik.inverse_kinematics(
            x[0], x[1], x[2],
            tool_pitch=np.radians(90),
            reference_joints=previous_q
        )
    )

    J = pin.computeFrameJacobian(
        model, data, q, frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    Jp = J[:3, :]

    qdot = damped_pinv(Jp) @ xdot
    qdot = np.clip(qdot, -2.0, 2.0)

    pin.computeJointJacobiansTimeVariation(model, data, q, qdot)

    Jdot = pin.getFrameJacobianTimeVariation(
        model, data, frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    Jpdot = Jdot[:3, :]

    qddot = damped_pinv(Jp) @ (xddot - Jpdot @ qdot)
    qddot = np.clip(qddot, -5.0, 5.0)

    tau = pin.rnea(model, data, q, qdot, qddot)

    qs.append(q)
    qdots.append(qdot)
    qddots.append(qddot)
    taus.append(tau)

    previous_q = tuple(q)

qs = np.array(qs)
qdots = np.array(qdots)
qddots = np.array(qddots)
taus = np.array(taus)

print("Max torque =", np.max(np.abs(taus), axis=0))

plot_joint_data(time, qs, "Angular Position (rad)", "Joint Angular Positions - Circular")
plot_joint_data(time, qdots, "Angular Velocity (rad/s)", "Joint Angular Velocities - Circular")
plot_joint_data(time, qddots, "Angular Acceleration (rad/s²)", "Joint Angular Accelerations - Circular")
plot_joint_data(time, taus, "Torque (Nm)", "Joint Torques - Circular")

plt.show()