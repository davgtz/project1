import einops
from robot.urdf_mass_matrix_kuka import compute_mass_matrix_urdf
import time
import torch


def dh_to_transformation(a, alpha, d, theta, conv):
    """Convert DH parameters to a homogenous transformation matrix

    Args:
        a (float): link length
        alpha (float): link twist
        d (float): link offset
        theta (float): joint angle

    Returns:
        T (4x4 torch tensor): homogenous transformation matrix

    """
    horizon = theta.shape[0]
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)

    T = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0).expand(horizon, -1, -1).clone()
    if conv == "SDH":
        T[:, 0, 0] = cos_theta
        T[:, 0, 1] = -sin_theta * cos_alpha
        T[:, 0, 2] = sin_theta * sin_alpha
        T[:, 0, 3] = a * cos_theta
        T[:, 1, 0] = sin_theta
        T[:, 1, 1] = cos_theta * cos_alpha
        T[:, 1, 2] = -cos_theta * sin_alpha
        T[:, 1, 3] = a * sin_theta
        T[:, 2, 1] = sin_alpha
        T[:, 2, 2] = cos_alpha
        T[:, 2, 3] = d

    elif conv == "MDH":
        # MDH
        T[:, 0, 0] = cos_theta
        T[:, 0, 1] = -sin_theta
        T[:, 0, 2] = 0.0
        T[:, 0, 3] = a
        T[:, 1, 0] = sin_theta * cos_alpha
        T[:, 1, 1] = cos_theta * cos_alpha
        T[:, 1, 2] = -sin_alpha
        T[:, 1, 3] = -d * sin_alpha
        T[:, 2, 0] = sin_theta * sin_alpha
        T[:, 2, 1] = cos_theta * sin_alpha
        T[:, 2, 2] = cos_alpha
        T[:, 2, 3] = d * cos_alpha
    else:
        raise ValueError("Convention not specified")

    return T


def compute_fk(q, conv):
    """Computes the forward kinematics of the UR5 manipulator

    Args:
        q (6x1 torch tensor): joint angles

    Returns:
        T (4x4 torch tensor): homogenous transformation matrix

    """
    DH_params = torch.tensor(
        [
            [0.0, torch.pi / 2, 0.3105],  # link 1 parameters 0.3105
            [0.0, -torch.pi / 2, 0.0],  # link 2 parameters
            [0.0, -torch.pi / 2, 0.4],  # link 3 parameters
            [0.0, torch.pi / 2, 0.0],  # link 4 parameters
            [0.0, torch.pi / 2, 0.39],  # link 5 parameters
            [0.0, -torch.pi / 2, 0.00],  # link 6 parameters
            [0.0, 0.0, 0.0933],
        ]
    ).to("cuda")

    # KUKA Modified DH parameters (a_n_1, alpha_n_1, d_n, theta_n)
    MDH_params = torch.tensor(
        [
            [0.0, 0.0, 0.3105],  # 0.3105
            [0.0, torch.pi / 2, 0.0],  # link 1 parameters
            [0.0, -torch.pi / 2, 0.4],  # link 2 parameters
            [0.0, -torch.pi / 2, 0.0],  # link 3 parameters
            [0.0, torch.pi / 2, 0.39],  # link 4 parameters
            [0.0, torch.pi / 2, 0.0],  # link 5 parameters
            [0.0, -torch.pi / 2, 0.0933],  # link 6 parameters
        ]
    ).to("cuda")
    horizon = q.shape[1]
    assert len(q) == 7, "There should be 7 joint angles"
    T = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0).expand(horizon, -1, -1)
    if conv == "SDH":
        for i, dh in enumerate(DH_params):
            a = dh[0]
            alpha = dh[1]
            d = dh[2]
            T = torch.matmul(T, dh_to_transformation(a, alpha, d, q[i, :], conv))
        return T
    elif conv == "MDH":
        for i, mdh in enumerate(MDH_params):
            a_n_1 = mdh[0]
            alpha_n_1 = mdh[1]
            d = mdh[2]
            T = torch.matmul(T, dh_to_transformation(a_n_1, alpha_n_1, d, q[i, :], conv))
        return T
    else:
        raise ValueError("NO convention specified, select SDH or MDH")


def compute_rpy_xyz(H):
    """Compute the roll, pitch, yaw angles of a x-y-z rotation sequence from an
    homogenous transformation matrix. The code was extracted and adapted to pytorch
    from Peter Corke's robotics toolbox for Python.
    Link: https://petercorke.github.io/robotics-toolbox-python

    Args:
        H: 4x4 torch tensor

    Returns:
        rpy: 3x1 torch tensor

    """
    horizon = H.shape[0]
    rpy = torch.empty((horizon, 3), dtype=torch.float32).to("cuda")
    tol = 20
    eps = 1e-6

    horizon = H.shape[0]
    rpy = torch.empty((horizon, 3), dtype=torch.float32).to("cuda")
    tol = 20
    eps = 1e-6

    # Check for singularity
    abs_r13_minus_1 = torch.abs(torch.abs(H[:, 0, 2]) - 1)
    singularity_mask = abs_r13_minus_1 < tol * eps

    # Compute roll (rpy[:, 0])
    rpy[:, 0] = torch.where(
        singularity_mask,
        torch.tensor(0, dtype=H.dtype, device=H.device),
        -torch.atan2(H[:, 0, 1], H[:, 0, 0]),
    )
    # Compute yaw (rpy[:, 2])
    mask2 = H[:, 0, 2] > 0
    rpy[:, 2] = torch.where(
        singularity_mask,
        torch.where(mask2, torch.atan2(H[:, 2, 1], H[:, 1, 1]), -torch.atan2(H[:, 1, 0], H[:, 2, 0])),
        -torch.atan2(H[:, 1, 2], H[:, 2, 2]),
    )

    # Compute pitch (rpy[:, 1])
    rpy[:, 1] = torch.where(singularity_mask, torch.asin(torch.clip(H[:, 0, 2], -1.0, 1.0)), -9999999)

    # Handling cases where k is not 0 or 1 (k == 2 or k == 3)
    k = torch.argmax(torch.abs(torch.stack((H[:, 0, 0], H[:, 0, 1], H[:, 1, 2], H[:, 2, 2]), dim=1)), dim=1)
    cr = torch.cos(-torch.atan2(H[:, 0, 1], H[:, 0, 0]))
    sr = torch.sin(-torch.atan2(H[:, 0, 1], H[:, 0, 0]))
    sr2 = torch.sin(-torch.atan2(H[:, 1, 2], H[:, 2, 2]))
    cr2 = torch.cos(-torch.atan2(H[:, 1, 2], H[:, 2, 2]))

    rpy[:, 1] = torch.where(k == 0, torch.atan(H[:, 0, 2] * cr / H[:, 0, 0]), rpy[:, 1])
    rpy[:, 1] = torch.where(k == 1, -torch.atan(H[:, 0, 2] * sr / H[:, 0, 1]), rpy[:, 1])
    rpy[:, 1] = torch.where(k == 2, -torch.atan(H[:, 0, 2] * sr2 / H[:, 1, 2]), rpy[:, 1])
    rpy[:, 1] = torch.where(k == 3, torch.atan(H[:, 0, 2] * cr2 / H[:, 2, 2]), rpy[:, 1])

    return rpy[:, 0], rpy[:, 1], rpy[:, 2]


def hom_to_pose(H):
    """Converts a 4x4 homogenous transformation matrix to a 6x1 compact pose

    Args:
        H (4x4 torch tensor): homogenous transformation matrix

    Returns:
        pose (6x1 torch tensor): [x, y, z, roll, pitch, yaw]

    """
    horizon = H.shape[0]
    pose = torch.empty((horizon, 6), dtype=torch.float32).to("cuda")
    pose[:, 0] = H[:, 0, 3]
    pose[:, 1] = H[:, 1, 3]
    pose[:, 2] = H[:, 2, 3]
    pose[:, 3], pose[:, 4], pose[:, 5] = compute_rpy_xyz(H)

    return pose


def fkine(q, conv="SDH"):
    """Computes the forward kinematics of the UR5 manipulator
    in compact representation [x, y, z, roll, pitch, yaw]

    Args:
        q (6x(B H) torch tensor): joint angles

    Returns:
        compact (6x(B H) torch tensor): [x, y, z, roll, pitch, yaw]

    """
    H = compute_fk(q, conv)
    # print(H)
    compact = hom_to_pose(H)
    return compact


def compute_analytical_jacobian(q):
    horizon = q.shape[1]
    timer = False

    def __fkine__(inputs):
        return fkine(inputs).sum(axis=0)

    if timer:
        start_time = time.time()
        outputs = fkine(q)
        end_time = time.time()
        print("Forward kinematics time: ", end_time - start_time)
        start_time = time.time()
        jac = torch.autograd.functional.jacobian(
            __fkine__, q, retain_graph=True, create_graph=False, vectorize=True
        ).to("cuda")
        end_time = time.time()
        print("Jacobian computation time: ", end_time - start_time)
        start_time = time.time()
        J = einops.rearrange(jac, "m n k -> k m n")
        end_time = time.time()
        print("Jacobian rearrange time: ", end_time - start_time)
    else:
        jac = torch.autograd.functional.jacobian(__fkine__, q, create_graph=False, vectorize=True).to("cuda")
        J = einops.rearrange(jac, "m n k -> k m n").to("cuda")

    return J


def compute_inertia_matrix(q):
    """Computes the inertia matrix of the UR5 manipulator based on
    https://github.com/kkufieta/ur5_modeling_force_estimate and
    adapted to pytorch.

    Matrix structure

    Mq = [m11 m12 m13 m14 m15 m16;
         m21 m22 m23 m24 m25 m26;
         m31 m32 m33 m34 m35 m36;
         m41 m42 m43 m44 m45 m46;
         m51 m52 m53 m54 m55 m56;
         m61 m62 m63 m64 m65 m66];

    Args:
        q (6xBatch torch tensor): joint angles

    Returns:
        Mq (Batchx6x6 torch tensor): inertia matrix
    """

    ## Returns the dynamics of the Kuka manipulator
    return compute_mass_matrix_urdf(q)


def compute_reflected_mass(q, u):
    """Computes the reflected mass of the UR5 manipulator
    along some direction u

    Args:
        q (Bx6xH) torch tensor): joint angles
        u (6x(B H) torch tensor): direction vector

    Returns:
        mu (float): reflected mass along u
    """
    b, t, h = q.shape
    q = einops.rearrange(q, "b t h -> t (b h)").to("cuda")
    J = compute_analytical_jacobian(q)
    Mq = compute_inertia_matrix(q)
    J_T = J.permute(0, 2, 1)
    M_x_inv = (J @ torch.linalg.solve(Mq, J_T))[:, :3, :3]
    u_T = u.permute(0, 2, 1)
    mu = 1 / (torch.matmul(u_T, torch.matmul(M_x_inv, u)).squeeze())
    mu = einops.rearrange(mu, "(b h) -> b h", b=b, h=h)

    return mu


def compute_kinetic_energy_matrix(q):
    """Computes the reflected mass of the UR5 manipulator
    along some direction u

    Args:
        q (Bx6xH) torch tensor): joint angles
        u (6x(B H) torch tensor): direction vector

    Returns:
        mu (float): reflected mass along u
    """
    b, t, h = q.shape
    q = einops.rearrange(q, "b t h -> t (b h)").to("cuda")
    J = compute_analytical_jacobian(q)
    Mq = compute_inertia_matrix(q)
    J_T = J.permute(0, 2, 1)

    M_x_inv = (J @ torch.linalg.solve(Mq, J_T))[:, :3, :3]
    return M_x_inv
