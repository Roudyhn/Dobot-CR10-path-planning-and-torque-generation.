import pinocchio as pin
import numpy as np

model = pin.buildModelFromUrdf("urdf/project.urdf")
data = model.createData()

q = np.array([0.3, 0.4, -0.5, 0.2, 0.1, 0.0])

frame_id = model.getFrameId("Link6")

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

J = pin.computeFrameJacobian(
    model,
    data,
    q,
    frame_id,
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)

print("Jacobian shape =", J.shape)
print(J)
Jp=J[:3,:]

xdot = np.array([0.05, 0.0, 0.0])  # desired EE velocity in x

qdot = np.linalg.pinv(Jp) @ xdot

print("qdot =", qdot)

cond = np.linalg.cond(Jp)
print("condition number =", cond)

if cond > 100:
    print("condition number is too large")
else:
    print("cond is good please proceed")
pin.computeJointJacobiansTimeVariation(model, data, q, qdot)

Jdot = pin.getFrameJacobianTimeVariation(
    model,
    data,
    frame_id,
    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)

Jpdot = Jdot[:3, :]

xddot = np.array([0.0, 0.0, 0.0])

qddot = np.linalg.pinv(Jp) @ (xddot - Jpdot @ qdot)

print("qddot =", qddot)

tau = pin.rnea(model, data, q, qdot, qddot)

print("tau =", tau)
print(model.inertias)