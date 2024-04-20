import cvxpy as cp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer

def solve_minimum_wrench(tip_poses, contact_normals, mu, min_force=5.0):
    """
    tip_poses: (N, 3)
    contact_normals: (N, 3)
    """
    center = tip_poses.mean(dim=0)
    r = tip_poses - center
    Rs = []
    ns = []
    Rp = []
    F = cp.Variable((r.shape[0],3))
    for i in range(r.shape[0]):
        Rs.append(vector_to_skew_symmetric_matrix(r[i]))
        ns.append(cp.Parameter((1,3)))
        Rp.append(cp.Parameter((3,3)))
    R = torch.stack(Rs)

    constraints = [ns[i] @ F[i] <= -np.sqrt(1/(1+mu**2)) * cp.pnorm(F[i]) for i in range(r.shape[0])]
    # minimum normal direction force.
    constraints = [ns[i] @ F[i] <= -min_force for i in range(r.shape[0])] + constraints

    torque_var = Rp[0]@F[0]
    for i in range(1,r.shape[0]):
        torque_var += Rp[i]@F[i]

    objective = cp.Minimize(0.5 * cp.pnorm(cp.sum(F,axis=0), p=2)+0.5 * cp.pnorm(torque_var, p=2))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    layer = CvxpyLayer(problem, parameters=ns+Rp, variables=[F])
    contact_normals = contact_normals.view(r.shape[0],1,3)
    solution = layer(*contact_normals, *R, solver_args={"eps": 1e-8, 
                                                        "max_iters": 10000, 
                                                        "acceleration_lookback": 0})[0]
    total_force = solution.sum(dim=0)
    total_torque = torch.cross(r, solution, dim=1).sum(dim=0)
    return solution, total_force.norm()+total_torque.norm()

def minimum_wrench_reward(tip_pose, contact_normals, mu, min_force):
    """
    tip_pose: (N, 3)
    contact_normals: (N, 3)
    mu: scalar
    """
    forces, total_wrench = solve_minimum_wrench(tip_pose, contact_normals, mu, min_force)
    margin = compute_margin(forces, contact_normals, mu)
    return total_wrench, margin.unsqueeze(0), forces

def dummy_minimum_wrench_reward(tip_pose, contact_normals, mu, min_force):
    forces = contact_normals * tip_pose.norm(dim=1, keepdim=True)
    total_wrench = forces.sum(dim=0).norm()
    margin = compute_margin(forces, contact_normals, mu)
    return total_wrench, margin.unsqueeze(0), forces

def compute_margin(forces, normals, mu):
    """
    forces: (N, 3)
    normals: (N, 3)
    mu: scalar
    compliance: scalar
    """
    forces_dir = -forces / torch.norm(forces, dim=1, keepdim=True)
    ang_diff = torch.sum(forces_dir * normals, dim=1)
    margin = ang_diff - np.sqrt(1/(1+mu**2))
    return margin


# TODO: Should be differentiable
def vector_to_skew_symmetric_matrix(v):
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], 
                         requires_grad=v.requires_grad,
                         device=v.device)

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

if __name__ == "__main__":
    tip_pose = torch.tensor([[0.0,0.0,0.0],
                        [1.0,0.0,0.0],
                        [0.0,1.0,0.0],
                        [0.0,0.0,1.0]]).double()
    center = tip_pose.mean(dim=0)
    n1 = center -torch.tensor([[0.0,0.0,0.0]])
    n2 = center - torch.tensor([[1.0,0.0,0.0]])
    n3 = center - torch.tensor([[0.0,1.0,0.0]])
    n4 = center - torch.tensor([[0.0,0.0,1.0]])
    n1 = (n1 / torch.norm(n1)).double().requires_grad_(True)
    n2 = (n2 / torch.norm(n2)).double().requires_grad_(True)
    n3 = (n3 / torch.norm(n3)).double().requires_grad_(True)
    n4 = (n4 / torch.norm(n4)).double().requires_grad_(True)
    n = torch.stack([n1,n2,n3,n4])

    tip_pose = tip_pose.requires_grad_(True)


    f = solve_minimum_wrench(tip_pose, n)

    f.sum().backward()
    print(f.shape)

    print(f.sum(dim=0), n1.grad)

