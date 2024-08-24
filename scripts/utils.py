import numpy as np
from scipy.interpolate import interp1d


def trapezoidal_velocity_profile(num_steps, max_velocity, acceleration_time):
    """Generates a trapezoidal velocity profile."""
    total_time = num_steps
    deceleration_time = acceleration_time

    # Time steps for the acceleration, constant velocity, and deceleration phases
    t_accel = int(acceleration_time)
    t_decel = int(deceleration_time)
    t_const = total_time - t_accel - t_decel

    velocity_profile = np.zeros(num_steps)

    # Acceleration phase
    for i in range(t_accel):
        velocity_profile[i] = max_velocity * (i / t_accel)

    # Constant velocity phase
    for i in range(t_accel, t_accel + t_const):
        velocity_profile[i] = max_velocity

    # Deceleration phase
    for i in range(t_accel + t_const, total_time):
        velocity_profile[i] = max_velocity * ((total_time - i) / t_decel)

    return velocity_profile


def create_trajectory(support_points, robot_configs, num_steps=48, max_velocity=1.0, acceleration_time=12):
    """Creates a trajectory with a trapezoidal velocity profile."""
    # Number of support points
    num_support_points = support_points.shape[0]

    # Calculate cumulative distances between support points
    distances = np.zeros(num_support_points)
    for i in range(1, num_support_points):
        distances[i] = distances[i - 1] + np.linalg.norm(support_points[i] - support_points[i - 1])

    # Normalize distances to [0, 1]
    distances /= distances[-1]

    # Create an interpolation function for support points and robot configurations
    point_interp_func = interp1d(distances, support_points, axis=0, kind="linear")
    config_interp_func = interp1d(distances, robot_configs, axis=0, kind="linear")

    # Trapezoidal velocity profile
    velocity_profile = trapezoidal_velocity_profile(num_steps, max_velocity, acceleration_time)

    # Calculate cumulative distance covered at each time step
    delta_distance = np.cumsum(velocity_profile)
    delta_distance /= delta_distance[-1]  # Normalize to 1

    # Interpolated positions and configurations along the path based on cumulative distance
    interpolated_points = point_interp_func(delta_distance)
    interpolated_configs = config_interp_func(delta_distance)

    return interpolated_points, interpolated_configs
