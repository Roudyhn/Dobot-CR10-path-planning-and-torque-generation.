#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple


Point3 = Tuple[float, float, float]
JointVector = Tuple[float, float, float, float, float, float]
Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to the [-pi, pi] interval."""

    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class CR10Geometry:
    """Simplified CR10-like geometry used for RViz kinematics demos."""

    base_height: float = 0.42
    shoulder_offset: float = 0.12
    upper_arm_length: float = 0.42
    forearm_length: float = 0.34
    wrist_roll_offset: float = 0.08
    tcp_offset: float = 0.10
    tool_length: float = 0.18
    default_tool_pitch: float = math.pi / 2.0
    ik_tolerance: float = 1e-6


class CR10Kinematics:
    """Forward and inverse kinematics for a 6-DOF arm with a fixed tool pitch."""

    _JOINT_LIMITS = (
        (-math.pi, math.pi),
        (math.radians(-170.0), math.radians(170.0)),
        (math.radians(-170.0), math.radians(170.0)),
        (math.radians(-190.0), math.radians(190.0)),
        (math.radians(-120.0), math.radians(120.0)),
        (-math.pi, math.pi),
    )

    def __init__(self, geometry: CR10Geometry | None = None) -> None:
        self.geometry = geometry or CR10Geometry()

    def forward_transform(self, joints: Iterable[float]) -> Matrix4:
        """Return the TCP transform that matches the current URDF chain.

        The wrist/tool chain in ``urdf/cr10_project.urdf`` places the TCP
        ``0.10 m`` after ``joint6`` and ``0.08 m`` after ``joint5``. That means
        joint6 changes TCP position, while joint5 is the pure axial rotation for
        top-down grasps when the tool pitch is fixed to point downward.
        """

        q1, q2, q3, q4, q5, q6 = joints

        transform = _translation(0.0, 0.0, 0.0)
        chain = (
            _rotation_z(q1),
            _translation(0.0, 0.0, self.geometry.base_height),
            _translation(self.geometry.shoulder_offset, 0.0, 0.0),
            _rotation_y_joint(q2),
            _translation(self.geometry.upper_arm_length, 0.0, 0.0),
            _rotation_y_joint(q3),
            _translation(self.geometry.forearm_length, 0.0, 0.0),
            _rotation_y_joint(q4),
            _translation(self.geometry.wrist_roll_offset, 0.0, 0.0),
            _rotation_x(q5),
            _rotation_z(q6),
            _translation(self.geometry.tcp_offset, 0.0, 0.0),
        )
        for step in chain:
            transform = _matmul(transform, step)
        return transform

    def forward_kinematics(self, joints: Iterable[float]) -> Point3:
        """Return the TCP position (x, y, z) in the base frame."""

        transform = self.forward_transform(joints)
        return (transform[0][3], transform[1][3], transform[2][3])

    def inverse_kinematics(
        self,
        x: float,
        y: float,
        z: float,
        *,
        tool_pitch: float | None = None,
        wrist_yaw: float = 0.0,
        prefer_elbow_up: bool = False,
        reference_joints: Sequence[float] | None = None,
    ) -> JointVector:
        """Solve for joint angles that place the TCP at the desired position.

        The arm is solved as a planar 2-link chain after the base yaw rotation.
        Joint 4 is chosen so the tool keeps a constant pitch. The current URDF
        places the TCP offset after joint6, so the yaw used for top-down grasping
        is assigned to joint5 instead. This keeps the TCP stationary while still
        rotating the gripper around its vertical tool axis.
        """

        pitch = self.geometry.default_tool_pitch if tool_pitch is None else tool_pitch
        radial = math.hypot(x, y)
        wrist_radius = radial - self.geometry.tool_length * math.cos(pitch)
        wrist_z = z + self.geometry.tool_length * math.sin(pitch)

        shoulder_radius = wrist_radius - self.geometry.shoulder_offset
        shoulder_z = wrist_z - self.geometry.base_height

        q1 = normalize_angle(math.atan2(y, x))
        q5 = normalize_angle(wrist_yaw)
        q6 = 0.0

        cos_q3 = (
            shoulder_radius**2
            + shoulder_z**2
            - self.geometry.upper_arm_length**2
            - self.geometry.forearm_length**2
        ) / (2.0 * self.geometry.upper_arm_length * self.geometry.forearm_length)

        tolerance = self.geometry.ik_tolerance
        if cos_q3 < -1.0 - tolerance or cos_q3 > 1.0 + tolerance:
            raise ValueError(
                "Target is outside the reachable workspace for the simplified CR10 model."
            )

        cos_q3 = max(-1.0, min(1.0, cos_q3))
        elbow_magnitude = math.acos(cos_q3)
        elbow_candidates = (
            (-elbow_magnitude, elbow_magnitude)
            if prefer_elbow_up
            else (elbow_magnitude, -elbow_magnitude)
        )

        valid_candidates: list[tuple[JointVector, float, float]] = []

        for q3 in elbow_candidates:
            k1 = self.geometry.upper_arm_length + self.geometry.forearm_length * math.cos(q3)
            k2 = self.geometry.forearm_length * math.sin(q3)
            q2 = math.atan2(-shoulder_z, shoulder_radius) - math.atan2(k2, k1)
            q4 = pitch - q2 - q3

            candidate = (
                normalize_angle(q1),
                normalize_angle(q2),
                normalize_angle(q3),
                normalize_angle(q4),
                q5,
                normalize_angle(q6),
            )

            if not self._within_joint_limits(candidate):
                continue

            error = self.position_error(candidate, (x, y, z))
            cost = sum(abs(value) for value in candidate[1:4])
            valid_candidates.append((candidate, error, cost))

        if not valid_candidates:
            raise ValueError(
                "No valid IK solution satisfied the joint limits for the requested target."
            )

        best_error = min(error for _, error, _ in valid_candidates)
        if best_error > 1e-5:
            raise ValueError(
                "No valid IK solution satisfied the joint limits for the requested target."
            )

        accurate_candidates = [
            (candidate, cost)
            for candidate, error, cost in valid_candidates
            if math.isclose(error, best_error, abs_tol=tolerance)
        ]

        if reference_joints is not None:
            return min(
                accurate_candidates,
                key=lambda item: (
                    self.configuration_distance(item[0], reference_joints),
                    item[1],
                ),
            )[0]

        return min(accurate_candidates, key=lambda item: item[1])[0]

    def position_error(self, joints: Sequence[float], target: Point3) -> float:
        achieved = self.forward_kinematics(joints)
        return math.dist(achieved, target)

    def tool_pitch(self, joints: Sequence[float]) -> float:
        _, q2, q3, q4, _, _ = joints
        return normalize_angle(q2 + q3 + q4)

    def configuration_distance(
        self,
        joints: Sequence[float],
        reference_joints: Sequence[float],
    ) -> float:
        return sum(
            normalize_angle(current - reference) ** 2
            for current, reference in zip(joints, reference_joints)
        )

    def validate_joint_solution(
        self,
        joints: Sequence[float],
        target: Point3,
        *,
        tolerance: float = 1e-5,
    ) -> bool:
        return self.position_error(joints, target) <= tolerance

    def _within_joint_limits(self, joints: Sequence[float]) -> bool:
        for index, value in enumerate(joints):
            lower, upper = self._JOINT_LIMITS[index]
            if value < lower - self.geometry.ik_tolerance:
                return False
            if value > upper + self.geometry.ik_tolerance:
                return False
        return True


def _matmul(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _translation(x: float, y: float, z: float) -> Matrix4:
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_x(angle: float) -> Matrix4:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, cosine, -sine, 0.0),
        (0.0, sine, cosine, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_y_joint(angle: float) -> Matrix4:
    """Match the positive rotation direction used by the URDF chain."""

    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        (cosine, 0.0, sine, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-sine, 0.0, cosine, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_z(angle: float) -> Matrix4:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        (cosine, -sine, 0.0, 0.0),
        (sine, cosine, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )