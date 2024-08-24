import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio.casadi as cpin
import PyKDL
from robotRenderer import RobotAnimator
from utils import create_trajectory
from scipy.spatial import cKDTree
import time

class Node:
    def __init__(
        self, position, orientation, joint_config, cost, acc_dis_cost=None, acc_rm_cost=None, rm_cost=None, parent=None
    ):
        self.position = position
        self.orientation = orientation
        self.joint_config = joint_config
        self.acc_dis_cost = acc_dis_cost
        self.acc_rm_cost = acc_rm_cost
        self.cost = cost
        self.rm_cost = rm_cost
        self.parent = parent

    def __repr__(self):
        return f"Node({self.joint_config})"


class ReflectedMass:
    def __init__(self, robot, end_effect_frame_id):

        self.robot = robot
        self._cmodel = cpin.Model(self.robot.model)
        self._cdata = self._cmodel.createData()
        self.end_effector_frame_id = end_effect_frame_id
        self.n_joints = self.robot.model.nq

        self.q_casadi = ca.SX.sym("q", self.n_joints)
        self.u_casadi = ca.SX.sym("u", 3)
        J_casadi, M_casadi = self._create_casadi_functions()

        J_trans = J_casadi(self.q_casadi)
        M_trans = M_casadi(self.q_casadi)
        K_trans = J_trans @ ca.inv(M_trans) @ J_trans.T
        K_trans = K_trans[:3, :3]
        self._reflected_mass = ca.inv(self.u_casadi.T @ K_trans @ self.u_casadi)
        self._rm_function = ca.Function("compute_rm", [self.q_casadi, self.u_casadi], [self._reflected_mass])
        self._gradient_reflected_mass = ca.gradient(self._reflected_mass, self.q_casadi)

        self._gradient_function = ca.Function(
            "compute_reflected_mass_gradient", [self.q_casadi, self.u_casadi], [self._gradient_reflected_mass]
        )

    def _create_casadi_functions(self):
        def compute_jacobian_and_mass_matrix(q):

            q_np = q
            # q_np = np.array(ca.DM(q))
            cpin.forwardKinematics(self._cmodel, self._cdata, q_np)
            cpin.computeJointJacobians(self._cmodel, self._cdata, q_np)
            J = cpin.getFrameJacobian(
                self._cmodel, self._cdata, self.end_effector_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            J_casadi = J
            M = cpin.crba(self._cmodel, self._cdata, q_np)

            M_casadi = M

            return J_casadi, M_casadi

        J_casadi_sym, M_casadi_sym = compute_jacobian_and_mass_matrix(self.q_casadi)

        J_casadi = ca.Function("J_casadi", [self.q_casadi], [J_casadi_sym])
        M_casadi = ca.Function("M_casadi", [self.q_casadi], [M_casadi_sym])

        return J_casadi, M_casadi

    def compute_gradient(self, q, u):

        return np.array(self._gradient_function(q, u)).reshape(self.n_joints)

    def compute_rm(self, q, u):

        return np.array(self._rm_function(q, u)).reshape(1)

def random_quaternion():
    """Random Quaternion according to Shoemake, K.. Uniform Random Rotations. Graphic Gems III

    Returns:
        quaternion: np.array() [x,y,z,w]
    """
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    w = np.random.uniform(0, 1)
    theta_1 = 2 * np.pi * v
    theta_2 = 2 * np.pi * w
    s1 = np.sin(theta_1)
    c1 = np.cos(theta_1)
    s2 = np.sin(theta_2)
    c2 = np.sin(theta_2)
    r1 = np.sqrt(1 - u)
    r2 = np.sqrt(u)

    q_x = s1 * r1
    q_y = c1 * r1
    q_z = s2 * r2
    q_w = c2 * r2

    return np.array([q_x, q_y, q_z, q_w])


class RRTStar:
    def __init__(
        self,
        start,
        goal,
        bounds,
        obstacles,
        robot_type,
        gain,
        verbose,
        max_iter=10000,
        step_size=0.1,
        step_size_ori=0.5,
        search_radius=1.0,
    ):
        self.start = Node(
            position=start[0],
            orientation=start[1],
            joint_config=start[2],
            cost=start[3],
            acc_dis_cost=0.0,
            acc_rm_cost=0.0,
            rm_cost=0.0,
        )
        self.goal = Node(
            position=goal[0],
            orientation=goal[1],
            joint_config=goal[2],
            cost=goal[3],
            acc_dis_cost=0.0,
            acc_rm_cost=0.0,
            rm_cost=0.0,
        )
        self.bounds = bounds
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.step_size_ori = step_size_ori
        self.search_radius = search_radius
        self.tree = [self.start]
        self._beta = 0.5
        self.kd_tree = cKDTree([self.start.position])  # K-Dimensional Tree to search for nearest neighboors
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.lines = []
        self.robot_type = robot_type
        if robot_type == "kuka":
            urdf_path = "robot/urdfs/kuka_lwr.urdf"
            frame_name = "link_7"  # from self.robot.model.frames[-1].name  # name: link_7
        elif robot_type == "ur5":
            urdf_path = "robot/urdfs/ur5.urdf"

        else:
            raise ValueError("robot type not defined")
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=["/scripts"])
        frame_name = self.robot.model.frames[-1].name  # name: link_7
        self.data = self.robot.model.createData()
        self.tcp_frame_id = self.robot.model.getFrameId(frame_name)  # name: link7
        self.J_damp = 1e-12
        self.dt = 1e-3
        self.K = 1e-1  # KUKA 1e-1
        self.rm_K = gain  # KUKA 6e-1
        self.rm_solver = ReflectedMass(self.robot, self.tcp_frame_id)
        self.verbose = verbose
        if verbose:
            print("Robot model loaded")
            print(self.robot.model)
        self.stop = False
        np.random.seed()

    def _add_to_goal_node(self, new_node):
        self.goal.parent = new_node
        self.goal.joint_config, rm_cost = self.compute_new_config(new_node, self.goal.position)
        if rm_cost is None:
            return None
        acc_rm_cost = self.compute_traj_rm_cost(new_node, rm_cost)
        new_dis_cost = new_node.acc_dis_cost + self.distance(new_node.position, self.goal.position)
        new_cost = self._beta * (new_dis_cost) + (1 - self._beta) * acc_rm_cost
        self.goal.cost = new_cost
        self.goal.acc_dis_cost = new_dis_cost
        self.goal.acc_rm_cost = acc_rm_cost
        self.goal.rm_cost = rm_cost
        return self.goal

    def check_if_descendant(self, node1, node2):
        node = node1
        if self.verbose:
            print("Checking descendant...")
        while node is not None:

            if str(node) == str(node2):
                if self.verbose:
                    print("Found node")
                return True
            node = node.parent
        if self.verbose:
            print("nothing found")
        return False

    def compute_direction(self, node, target_pose):
        dir_pos = target_pose[:3] - node.position
        # relative quaternion
        q_0 = PyKDL.Rotation.Quaternion(
            x=node.orientation[0], y=node.orientation[1], z=node.orientation[2], w=node.orientation[3]
        )  # Accordint to PyKDL (x,y,z,w)
        q_t = PyKDL.Rotation.Quaternion(x=target_pose[3], y=target_pose[4], z=target_pose[5], w=target_pose[6])
        q_rel = q_0 * q_t.Inverse()
        quat_rel = q_rel.GetQuaternion()
        u = np.array(quat_rel[:2])

        theta = np.array([2 * np.arccos(np.abs(quat_rel[-1]))])
        return np.concatenate([dir_pos, u, theta])

    def compute_distance_error(self, node, target_pose):
        """Function that computes distance metric. There are many flavors for this since SE(3) has no Riemannian metric
        other than the combination of two metrics for translation and rotation, see [1]. This means there must be a choice
        of distance in orientation and also a weighting for the importance of the orientation part.

        [1] Zefran, M. Choice of Riemannian Metrics for Rigid Body Kinematics
        [2] Alvarez-Tunon, O. et al. (2023), Euclidean and Riemannian Metrics inLearning-based Visual Odometry
        [3] Huynh, D. Q. (2009) Metrics for 3D Rotations: Comparison and Analysis

        In [1] and [2], they combine the metrics by using weights on the rotational distance, e.g., norm(d_x - w_k*d_quat)

        Args:
            node (_type_): _description_
            target_pose (_type_): _description_
        """
        e_pos = target_pose[:3] - node.position
        # Geodesic distance
        q_0 = node.orientation
        q_1 = target_pose[3:]
        dot_q0q1 = np.dot(q_0, q_1)
        d_orien = 2 * np.arccos(np.abs(dot_q0q1))
        w_0 = 2
        distance = np.norm(e_pos) + w_0 * d_orien
        return distance

    def compute_jacobian(self, q):
        self.robot.computeJointJacobians(q)
        frame_name = "link_7"
        frame_id = self.robot.model.getFrameId(frame_name)
        self.robot.forwardKinematics(q)
        pin.updateFramePlacements(self.robot.model, self.data)
        J_world = pin.computeFrameJacobian(self.robot.model, self.data, q, frame_id)
        return J_world

    def compute_new_config(self, prev_node, target_position) -> tuple:

        q = prev_node.joint_config
        iterations = 0
        while True:

            pin.forwardKinematics(self.robot.model, self.data, q)
            pin.updateFramePlacements(self.robot.model, self.data)
            pin.computeJointJacobians(self.robot.model, self.data, q)
            J = pin.getFrameJacobian(
                self.robot.model, self.data, self.tcp_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )  
            J = J[:3, :]  # Translational part

            n_dof = len(q)
            lam = 5e-5
            Id = np.eye(n_dof)
            J_star = np.linalg.inv(J.T @ J + lam * Id) @ J.T
            # J_star = np.linalg.pinv(J)  # Pseudoinverse only of the translational part
            H_0T = self.data.oMi[-1]

            curr_tcp_position = H_0T.translation
            err = target_position - curr_tcp_position
            k_err = self.K * err

            v = (curr_tcp_position - self.goal.position) / np.linalg.norm(curr_tcp_position - self.goal.position)
            rm_grad = self.rm_K * self.rm_solver.compute_gradient(q, v)
            q_vel = J_star @ k_err + (np.eye(n_dof) - J_star @ J) @ rm_grad
            q_new = q + self.dt * q_vel
            if np.linalg.norm(err) > 0.05:

                q = q_new
                iterations += 1
                if iterations > 10000:
                    if self.verbose:
                        print("IK solver did not find a solution. \n Sampling again...")
                    return None, None
            else:
                np.set_printoptions(suppress=True)
                rm_cost = self.rm_solver.compute_rm(q, v)
                # print("Final JAcob", J)
                break
        return q_new, rm_cost

    def compute_orientation(self, q):
        pin.forwardKinematics(self.robot.model, self.data, q)
        pin.updateFramePlacements(self.robot.model, self.data)
        H = self.data.oMi[-1]
        tcp_orientation = pin.Quaternion(H.rotation)
        return tcp_orientation

    def compute_traj_rm_cost(self, current_node, new_rm_cost):
        # count nodes I + 1 term
        I_p_1 = 0
        node = current_node
        while node:
            node = node.parent
            if node:
                I_p_1 += 1

            if I_p_1 > 100:
                import ipdb

                ipdb.set_trace()
        # number of nodes visited
        if current_node.parent:
            final_cost = 1 / (I_p_1 + 1) * ((I_p_1 * current_node.acc_rm_cost) + new_rm_cost)
        else:
            final_cost = 1 / (I_p_1 + 1) * (new_rm_cost)
        return final_cost

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def sample_free(self):
        while True:
            quat = random_quaternion()
            sample = (
                np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
                np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
                np.random.uniform(self.bounds[2][0], self.bounds[2][1]),
                quat[0],  # x
                quat[1],  # y
                quat[2],  # z
                quat[3],  # w
            )
            if not self.in_obstacle(sample[:3]):  # Check only position
                return sample

    def in_obstacle(self, point):
        for ox, oy, oz, radius in self.obstacles:
            if self.distance(point, (ox, oy, oz)) <= radius:
                return True
        return False

    def steer(self, from_node, to_pose):
        d = self.compute_direction(from_node, to_pose)  # d is (e_pos, u_x, u_y, u_z, theta)
        length = np.linalg.norm(d[:3])
        d_pos = d[:3] / length
        new_position = np.array(from_node.position) + d_pos * min(self.step_size, length)

        if not self.in_obstacle(new_position):
            new_joint_config, rm_cost = self.compute_new_config(from_node, new_position)
            if new_joint_config is None:

                return None
            new_orientation = self.compute_orientation(new_joint_config)
            acc_rm_cost = self.compute_traj_rm_cost(from_node, rm_cost)
            new_dis_cost = from_node.acc_dis_cost + self.distance(from_node.position, new_position)
            new_cost = self._beta * (new_dis_cost) + (1 - self._beta) * acc_rm_cost

            return Node(
                position=new_position,
                orientation=new_orientation,
                joint_config=new_joint_config,
                cost=new_cost,
                acc_dis_cost=new_dis_cost,
                acc_rm_cost=acc_rm_cost,
                rm_cost=rm_cost,
                parent=from_node,
            )
        return None

    def get_nearest(self, position):
        distances, indices = self.kd_tree.query(position)
        return self.tree[indices]

    def get_near(self, position):

        indices = self.kd_tree.query_ball_point(position, self.search_radius)
        return [self.tree[i] for i in indices]

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.stop:
                ipdb.set_trace()
            acc_rm_cost = self.compute_traj_rm_cost(new_node, near_node.rm_cost)
            new_dis_cost = near_node.acc_dis_cost + self.distance(new_node.position, near_node.position)
            new_cost = self._beta * (new_dis_cost) + (1 - self._beta) * acc_rm_cost

            if new_cost < near_node.cost:

                if self.check_if_descendant(new_node, near_node):
                    break

                near_node.parent = new_node
                near_node.cost = new_cost
                near_node.acc_rm_cost = acc_rm_cost
                near_node.acc_dis_cost = new_dis_cost

    def plan(self, visualize=True):
        for _ in range(self.max_iter):
            cnt = 0
            while True:
                random_pose = self.sample_free()
                random_position = random_pose[:3]
                nearest_node = self.get_nearest(random_position)
                if cnt % 100 == 100:
                    if self.verbose:
                        print("stuck in plan() -> while loop")
                    import ipdb

                    ipdb.set_trace()
                new_node = self.steer(nearest_node, random_pose)
                cnt += 1
                if new_node is not None:
                    break

            if new_node:
                near_nodes = self.get_near(new_node.position)
                self.rewire(new_node, near_nodes)
                self.tree.append(new_node)
                self.kd_tree = cKDTree([node.position for node in self.tree])

                # Update the plot
                if visualize:
                    self.update_plot(new_node)

                if self.distance(new_node.position, self.goal.position) < 0.1:
                    if self.verbose:
                        print("Goal reached!")
                    new_node = self._add_to_goal_node(new_node)
                    if new_node is None:  # Goal not reachable
                        return None, None, None, None

                    if visualize:
                        self.plot_path(new_node)

                    return self.get_path(new_node)

        print("Goal not reached within max iterations.")
        return None

    def update_plot(self, new_node):
        if new_node.parent:
            (line,) = self.ax.plot(
                [new_node.position[0], new_node.parent.position[0]],
                [new_node.position[1], new_node.parent.position[1]],
                [new_node.position[2], new_node.parent.position[2]],
                "b-",
            )
            self.lines.append(line)
            plt.pause(0.01)

    def get_path(self, node):
        path = []
        config = []
        rm_masses = []
        acc_masses = []
        while node:
            path.append(node.position)
            if self.robot_type == "ur5":
                config.append(node.joint_config - np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # UR5 possibly necessary
            elif self.robot_type == "kuka":
                config.append(
                    node.joint_config - np.array([-2.0943951023931953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                )  # Necesary on Kuka
            else:
                raise ValueError("Invalid value on robot_type")
            rm_masses.append(node.rm_cost)
            acc_masses.append(node.acc_rm_cost)
            node = node.parent

        return path[::-1], config[::-1], rm_masses[::-1], acc_masses[::-1]

    def plot_path(self, node):
        path, _, _, _ = self.get_path(node)
        x_coords = [pos[0] for pos in path]
        y_coords = [pos[1] for pos in path]
        z_coords = [pos[2] for pos in path]
        self.ax.plot(x_coords, y_coords, z_coords, "g-", linewidth=2)

    def setup_plot(self):
        self.ax.set_xlim(self.bounds[0])
        self.ax.set_ylim(self.bounds[1])
        self.ax.set_zlim(self.bounds[2])
        for ox, oy, oz, radius in self.obstacles:
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = ox + radius * np.cos(u) * np.sin(v)
            y = oy + radius * np.sin(u) * np.sin(v)
            z = oz + radius * np.cos(v)
            self.ax.plot_wireframe(x, y, z, color="orange")
        self.ax.plot(self.start.position[0], self.start.position[1], self.start.position[2], "ro")
        self.ax.plot(self.goal.position[0], self.goal.position[1], self.goal.position[2], "go")


def run_experiment(exp_id: str, num_trials: float, robot_type: str, gain: float, verbose=True):
    ROBOT_TYPE = robot_type

    # Node structure following:
    # T, Pardi et al., Planning Maximum-Manipulability Cutting Paths.
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978478
    # node = (position, orientation, joint configuration, cost)

    # Cartesian initial position
    if ROBOT_TYPE == "kuka":
        p_0 = np.array([0.15, 0.40, 0.20])  # kuka environment
        # Cartesian initial orientation
        o_0 = np.array([0.7071067849526632, 0.7071067774204319, 0.0, 0.0])  # (x, y, z, w)
        # Joint initial configuration
        q_0 = np.array([1.31003418, -0.63709131, -0.11878658, 1.99867398, 0.14396832, -0.51364376, 2.65958823])
        # Joing initial configuration lwr_description frames
        q_0 = q_0 * np.array([1, 1, 1, 1, 1, 1, 1]) + np.array(
            [-2.0943951023931953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # KUKA robot

    elif ROBOT_TYPE == "ur5":
        p_0 = np.array([0.15, 0.30, 0.30])  # UR5 environment
        q_0 = np.array([1.44163478, -1.17557097, -2.14798643, -1.38881448, 1.57077003, -0.12910096])  # UR5
        q_0 = q_0 * np.array([1, 1, 1, 1, 1, 1]) + np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [-3 / 2 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]
        )  # UR5 adjustment to DH table of manufacturer
    else:
        raise ValueError("ROBOT_TYPE not defined")

    # Cartesian initial orientation
    o_0 = np.array([0.7071067849526632, 0.7071067774204319, 0.0, 0.0])  # (x, y, z, w)
    # Cartesian goal position
    p_goal = np.array([0.458, 0.0, 0.171])
    # Cartesian goal orientation
    o_goal = np.array([-0.7071067811864628, -0.7071067811866323, 0.0, 0.0])  # (x,y,z,w)

    start = (p_0, o_0, q_0, 0)
    goal = (p_goal, o_goal, None, None)
    hand = np.array([0.5, 0.1, 0.084])

    bounds = ((-0.1, 0.5), (-0.1, 0.5), (0.0, 0.5), ())  # ((-0.1, 0.5), (-0.1, 0.5), (0.0, 0.5), ())
    obstacles = [(0.0, 0.0, 0.0, 0.02)]

    ## Experiment Code
    N_EXPERIMENTS = num_trials
    VISUALIZE = verbose
    exp_rm_masses = []
    exp_traj_config = []
    exp_traj_points = []
    computation_time_l = []
    print(f"Starting Experiment {exp_id}...")
    for i in range(N_EXPERIMENTS):
        start_time = time.time()
        if i > 0 and i % 10 == 0:
            print(
                f"***********************************Running experiment number {i} from worker {exp_id}**********************************************************"
            )
        rrt_star = RRTStar(start, goal, bounds, obstacles, ROBOT_TYPE, gain, verbose)
        if VISUALIZE:
            rrt_star.setup_plot()

        path, config, rm_mass, acc_rm_mass = rrt_star.plan(visualize=VISUALIZE)
        if path is None:
            continue
        if VISUALIZE:
            plt.show()
        if VISUALIZE:
            animator = RobotAnimator(robot_type=robot_type)

        trajectory_points, trajectory_configs = create_trajectory(np.array(path), np.array(config))
        rm_masses = []
        end_time = time.time()
        computation_time = (end_time - start_time) / 60  # in minutes
        print("computation_time: ", computation_time)
        for config in trajectory_configs:
            if ROBOT_TYPE == "kuka":
                config = config + np.array([-2.0943951023931953, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            pin.forwardKinematics(rrt_star.robot.model, rrt_star.data, config)
            pin.updateFramePlacements(rrt_star.robot.model, rrt_star.data)
            H_0T = rrt_star.data.oMi[-1]
            curr_tcp_position = H_0T.translation
            v = (curr_tcp_position - p_goal) / np.linalg.norm(curr_tcp_position - p_goal)
            rm_masses.append(rrt_star.rm_solver.compute_rm(config, v))

        exp_rm_masses.append(rm_masses)
        exp_traj_config.append(trajectory_configs)
        exp_traj_points.append(trajectory_points)
        computation_time_l.append(computation_time)

        if VISUALIZE:
            animator.load_trajectory(trajectory_configs)

            animator.render_robot_animation(save=True)
    print(f"Worker {exp_id} DONE!!...")
    return {
        "name": exp_id,
        "joint_position": np.array(exp_traj_config),
        "traj_points": np.array(exp_traj_points),
        "rm_masses": np.array(exp_rm_masses),
        "time": np.array(computation_time_l),
    }


if __name__ == "__main__":
    someDictionary = run_experiment(1, 1, "ur5", 1e-1, verbose=True)
    import ipdb

    ipdb.set_trace()
