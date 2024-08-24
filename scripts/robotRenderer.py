import numpy as np

import einops
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch


class AnimatorBase:
    def __init__(self):
        self._x = None
        pass

    def get_x_traj(self):
        return self._x


class RobotAnimator(AnimatorBase):
    def __init__(self, sample_directory=None, robot_type="kuka"):
        super().__init__()
        if robot_type == "kuka":
            from rm_analysisKUKA import (
                extract_trajectory,
                forward_kinematics,
                compute_trajectory_in_cartesian,
            )
            from robot.KUKALWR4KinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix

            self.compute_kinetic_energy_matrix = compute_kinetic_energy_matrix
            self.forward_kinematics = forward_kinematics
        elif robot_type == "ur5":
            from rm_analysisUR5 import (
                extract_trajectory,
                forward_kinematics,
                compute_trajectory_in_cartesian,
            )
            from robot.UR5kinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix

            self.compute_kinetic_energy_matrix = compute_kinetic_energy_matrix
            self.forward_kinematics = forward_kinematics
        else:
            from rm_analysisKUKA import (
                extract_trajectory,
                forward_kinematics,
                compute_trajectory_in_cartesian,
            )
            from robot.KUKALWR4KinematicsAndDynamics_vectorized import compute_kinetic_energy_matrix

            self.compute_kinetic_energy_matrix = compute_kinetic_energy_matrix
            self.forward_kinematics = forward_kinematics
        self.compute_trajectory_in_cartesian = compute_trajectory_in_cartesian
        print(f"importing {robot_type} rm_analysis")
        self._directory = sample_directory
        experiment_file_number = 0
        self._ellipsoid = self._L = self._V = None
        if sample_directory:

            q, goal, hand = extract_trajectory(directory=self._directory, exp_number=experiment_file_number)
            self._q = q
            self._goal = goal
            self._hand = hand
        else:
            self._q = None
            self._goal = None
            self._hand = None

    def load_trajectory(self, theta=None):
        """Function to load the trajectory to the class

        Args:
            theta (numpy.array): Trajectory in C-Space. Shape: (horizon, n_dof)
                                 typically (48, 7)

        Raises:
            ValueError: Returns an error if no trajectory was given and there is
                        no sample directory to use.
        """
        if theta is None:
            if self._directory is not None:
                traj_number = 1
                batch, sth, horizon, n_dof = np.shape(self._q)
                self._q = theta = np.reshape(self._q[traj_number], (48, n_dof))
            else:
                raise ValueError("Neither sample directory was given nor theta trajectory!")

        self._x = self.compute_trajectory_in_cartesian(theta)
        self._q = theta
        self._goal = np.expand_dims(self._x[-1], axis=0)
        self._hand = np.expand_dims(self._x[-1], axis=0)

    def _compute_traj_eigendecomposition(self, theta):
        """Computes the eigendecomposition of the Kinetic energy matrix
        at each vector in the trajectory

        Args:
            theta (BxHxT): Trajectory array of a batch of trajectories

        Returns:
            L, V: Eigenvectors and Eigenvalues in matrix form.
        """

        q_ = torch.tensor(theta).unsqueeze(dim=0).to(torch.float32)
        q_ = einops.rearrange(q_, "B H T -> B T H")

        M_x_inv = self.compute_kinetic_energy_matrix(q_)
        L, V_ = torch.linalg.eig(M_x_inv)
        L = 1 / L.real.cpu()
        L = torch.sqrt(L) * 0.1
        V = V_.real.cpu().numpy()
        return L, V

    def _ellipsoid_surface_points(self, center, radii, rotation, num_points=20):
        # Generate spherical coordinates
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, np.pi, num_points)

        # Convert spherical coordinates to Cartesian coordinates
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Apply rotation matrix
        points = np.stack((x, y, z), axis=-1)
        rotated_points = np.dot(points, rotation) + center
        return rotated_points

    def _plot_ellipsoid(self, x_init, Lambda, Q):

        # Define ellipsoid parameters
        center = (x_init[0], x_init[1], x_init[2])  # Center of the ellipsoid
        radii = (Lambda[0], Lambda[1], Lambda[2])  # Length of the semi-axes along each axis

        rotation = Q.T  # Rotation matrix (identity matrix for no rotation)

        # Generate surface points of the ellipsoid
        surface_points = self._ellipsoid_surface_points(center, radii, rotation)

        # Plot the ellipsoid mesh
        ellipsoid = self._ax.plot_wireframe(
            surface_points[:, :, 0],
            surface_points[:, :, 1],
            surface_points[:, :, 2],
            color="b",
            alpha=0.3,
        )
        return ellipsoid

    # Plotting function
    def _update_plot(self, frame, line, line2, scatter, quiver):

        # Update joint angles
        new_positions, T = self.forward_kinematics(self._q[frame])
        # Update plot
        line.set_data_3d(
            [p[0] for p in new_positions[:-1]],
            [p[1] for p in new_positions[:-1]],
            [p[2] for p in new_positions[:-1]],
        )
        line2.set_data_3d(
            [p[0] for p in new_positions[-2:]],
            [p[1] for p in new_positions[-2:]],
            [p[2] for p in new_positions[-2:]],
        )
        scatter._offsets3d = (
            np.array([p[0] for p in new_positions[:-2]]),
            np.array([p[1] for p in new_positions[:-2]]),
            np.array([p[2] for p in new_positions[:-2]]),
        )

        # Update frame at the tip of the robot
        tip_position = new_positions[-1]  # Get the position of the end effector (tip)
        frame_axes = T[:3, :3]  # Extract rotation matrix from the transformation matrix
        # Define scaling factor
        scale_factor = 0.05  # Adjust this value to shorten or lengthen the axes
        # Plot ellipsoid
        # Remove wireframe plot

        if self._ellipsoid:
            self._ellipsoid.remove()

        self._ellipsoid = self._plot_ellipsoid(tip_position, self._L[frame], self._V[frame])
        # Scale the frame axes
        frame_axes = frame_axes * scale_factor
        quiver.set_segments([np.array([tip_position, tip_position + frame_axes[:, i]]) for i in range(3)])
        quiver._offsets3d = (tip_position[0], tip_position[1], tip_position[2])
        quiver.set_color(["r", "g", "b"])
        return line, line2, scatter, quiver

    def _initialize_plot(self):

        # initial TCP position
        x_init = self._x[0]

        self._fig = plt.figure(facecolor="none", figsize=(14,10))
        self._ax = self._fig.add_subplot(111, projection="3d")

        # Plot entire trajectory points
        for pos in self._x:
            self._ax.scatter(pos[0], pos[1], pos[2], color="r", s=5)

        goal = self._goal[0, :3]

        self._ax.scatter(goal[0], goal[1], goal[2], marker="^", s=50, color="green")
        hand = self._hand[0].squeeze()
        self._ax.scatter(hand[0], hand[1], hand[2], marker="*", s=50, color="red")
        # Define the table top dimensions
        table_width = 1.0
        table_length = 1.5
        table_height = 0.0  # Height above the origin

        # Plot table top as a rectangle
        table_top = self._ax.plot(
            [-table_width / 2, table_width / 2],
            [-table_length / 2, -table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )
        self._ax.plot(
            [table_width / 2, table_width / 2],
            [-table_length / 2, table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )
        self._ax.plot(
            [table_width / 2, -table_width / 2],
            [table_length / 2, table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )
        self._ax.plot(
            [-table_width / 2, -table_width / 2],
            [table_length / 2, -table_length / 2],
            [table_height, table_height],
            color="goldenrod",
            linewidth=2,
        )

        self._L, self._V = self._compute_traj_eigendecomposition(self._q)
        # Plot Ellipsoid

        self._ellipsoid = self._plot_ellipsoid(x_init, self._L[0], self._V[0])

        # Initialize plot
        positions, T = self.forward_kinematics(self._q[0])
        (line,) = self._ax.plot(
            [p[0] for p in positions[:-1]],
            [p[1] for p in positions[:-1]],
            [p[2] for p in positions[:-1]],
            color="lightblue",
            linewidth=22,
            alpha=1,
        )
        (line2,) = self._ax.plot(
            [p[0] for p in positions[-2:]],
            [p[1] for p in positions[-2:]],
            [p[2] for p in positions[-2:]],
            color="dimgray",
            linewidth=12,
            alpha=1,
        )
        scatter = self._ax.scatter(
            [p[0] for p in positions[:-1]],
            [p[1] for p in positions[:-1]],
            [p[2] for p in positions[:-1]],
            color="lightgray",
            s=500,
            alpha=1,
        )

        # Draw initial frame at the tip of the robot
        tip_position = positions[-1]  # Get the position of the end effector (tip)
        frame_axes = T[:3, :3]  # Extract rotation matrix from the transformation matrix
        quiver = self._ax.quiver(
            tip_position[0],
            tip_position[1],
            tip_position[2],
            [frame_axes[0, 0], frame_axes[1, 0], frame_axes[2, 0]],
            [frame_axes[0, 1], frame_axes[1, 1], frame_axes[2, 1]],
            [frame_axes[0, 2], frame_axes[1, 2], frame_axes[2, 2]],
            color=["r", "g", "b"],
            length=0.1,
        )

        # Setting labels and aspect ratio
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        self._ax.set_box_aspect([1, 1, 1])
        ticks = np.linspace(-0.5, 0.5, num=5)
        ticksz = np.linspace(-0.5, 1.0, num=7)
        self._ax.set_xticks(ticks)
        self._ax.set_yticks(ticks)
        self._ax.set_zticks(ticksz)
        self._ax.set_xlim(-0.5, 0.5)
        self._ax.set_ylim(-0.5, 0.5)
        self._ax.set_zlim(-0.5, 1.0)

        # Set camera perspective
        self._ax.view_init(elev=10, azim=-25)  # Adjust elevation and azimuth as desired
        # ax.dist = 5
        # ax.grid(False)
        return line, line2, scatter, quiver

    def render_robot_animation(self, save: bool = False, name: str = None):

        line, line2, scatter, quiver = self._initialize_plot()
        # Create animation
        ani = FuncAnimation(
            self._fig, self._update_plot, frames=48, fargs=(line, line2, scatter, quiver), interval=48, blit=False
        )

        if save:
            if name:
                ani.save(name + ".gif", writer="pillow", fps=30)
            else:
                ani.save("robot_animation1.gif", writer="pillow", fps=15)

        # Show plot
        plt.axis("off")
        plt.show()


if __name__ == "__main__":

    "****** Example usage **********"
    # Directory containing .npz files
    directory = (
        "logs/kuka_coppeliasim_full_path/plans/release_H48_T20_LimitsNormalizer_b64_condFalse/0/"  # Current directory
    )

    animator = RobotAnimator(sample_directory=directory)
    animator.load_trajectory(theta=None)
    animator.render_robot_animation(save=False)
