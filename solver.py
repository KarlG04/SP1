import numpy as np
import time
from scipy.spatial import KDTree

def initialize_fluid(fluid_length, fluid_height, spacing):
    inlet_points_x, inlet_points_y = [], []

    inlet_points_x = np.linspace(spacing, fluid_length, int(fluid_length / spacing))
    inlet_points_y = np.linspace(spacing, fluid_height, int(fluid_height / spacing))

    inlet_points_x, inlet_points_y = np.meshgrid(inlet_points_x, inlet_points_y)
    inlet_points = np.vstack([inlet_points_x.ravel(), inlet_points_y.ravel()]).T

    # Add random Offset
    random_shift = np.random.uniform(0, 0.1, inlet_points.shape)
    inlet_points += random_shift
    
    # Konvert inlet_points in a list of tupel
    fluid_positions = [tuple(point) for point in inlet_points]
    fluid_velocities = [(0.0, 0.0) for _ in fluid_positions]  # Initialize velocities to (0, 0) for each point

    return fluid_positions, fluid_velocities

def find_neighbors(fluid_positions, smoothing_length):
    num_particles = len(fluid_positions)
    neighbors = [[] for _ in range(num_particles)]
    distances = [[] for _ in range(num_particles)]
    tree = KDTree(fluid_positions)
    
    for i, position in enumerate(fluid_positions):
        neighbor_indices = tree.query_ball_point(position, smoothing_length)
        for index in neighbor_indices:
            if index != i:
                neighbors[i].append(index)
                distance = np.linalg.norm(np.array(position) - np.array(fluid_positions[index]))
                distances[i].append((index, distance))

    return neighbors, distances

def calculate_density(density_factor, smoothing_length, distances):
    num_particles = len(distances)
    densities = [0.0] * num_particles

    self_density = density_factor * (smoothing_length**2)**3

    for i in range(num_particles):
        density_sum = 0.0
        for j, dij in distances[i]:
            density_sum += density_factor * (smoothing_length**2 - dij**2)**3
        densities[i] = self_density + density_sum

    return densities

def calculate_pressure(isentropic_exponent, initial_density, densities):
    pressures = [0.0] * len(densities)
    
    for i in range(len(densities)):
        pressures[i] = isentropic_exponent * (densities[i] - initial_density)

    return pressures

def calculate_pressure_force(pressure_factor, pressures, densities, distances, smoothing_length, fluid_positions):
    num_particles = len(fluid_positions)
    pressure_forces = [(0.0, 0.0) for _ in range(num_particles)]

    for i in range(num_particles):
        sum_force_x = 0.0
        sum_force_y = 0.0
        xi, yi = fluid_positions[i]
        pi = pressures[i]
        
        for j, dij in distances[i]:
            xj, yj = fluid_positions[j]
            pj = pressures[j]
            rhoj = densities[j]
            term = ((smoothing_length - dij)**2 * (pj + pi)) / ((2 * rhoj) * dij)
             
            sum_force_x += pressure_factor * -(xj - xi) * term
            sum_force_y += pressure_factor * -(yj - yi) * term
        
        force_x = sum_force_x
        force_y = sum_force_y
        
        pressure_forces[i] = (force_x, force_y)

    return pressure_forces

def calculate_viscous_force(viscosity_factor, distances, smoothing_length, densities, fluid_velocities):
    num_particles = len(distances)
    viscous_forces = [(0.0, 0.0) for _ in range(num_particles)]

    for i in range(num_particles):
        sum_force_x = 0.0
        sum_force_y = 0.0
        ui, vi = fluid_velocities[i]

        for j, dij in distances[i]:
            uj, vj = fluid_velocities[j]
            rhoj = densities[j]

            term = (smoothing_length - dij) / rhoj

            sum_force_x += viscosity_factor * (uj - ui) * term
            sum_force_y += viscosity_factor * (vj - vi) * term

        force_x = sum_force_x
        force_y = sum_force_y

        viscous_forces[i] = (force_x, force_y)

    return viscous_forces

def sum_up_forces(pressure_forces, viscous_forces, gravity):
    num_particles = len(pressure_forces)
    total_forces = [(0.0, 0.0) for _ in range(num_particles)]

    for i in range(num_particles):
        px, py = pressure_forces[i]
        vx, vy = viscous_forces[i]
        gx, gy = gravity

        total_force_x = px + vx + gx
        total_force_y = py + vx + gy

        total_forces[i] = (total_force_x, total_force_y)

    return total_forces

def integrate_acceleration(fluid_positions, fluid_velocities, densities, delta_t, total_forces):
    num_particles = len(fluid_positions)
    new_fluid_positions = [(0.0, 0.0) for _ in range(num_particles)]
    new_fluid_velocities = [(0.0, 0.0) for _ in range(num_particles)]
    
    for i in range(num_particles):
        xi, yi = fluid_positions[i]
        ui, vi = fluid_velocities[i]
        rhoi = densities[i]
        fx, fy = total_forces[i]

        new_ui = ui + delta_t * fx / rhoi
        new_vi = vi + delta_t * fy / rhoi

        new_xi = xi + delta_t * new_ui
        new_yi = yi + delta_t * new_vi

        new_fluid_positions[i] = (new_xi, new_yi)
        new_fluid_velocities[i] = (new_ui, new_vi)

    return new_fluid_positions, new_fluid_velocities

def enforce_boundary_condition(fluid_positions, fluid_velocities, box_length, box_height, spacing, boundary_damping):
    for i in range(len(fluid_positions)):
        x, y = fluid_positions[i]
        u, v = fluid_velocities[i]

        if x < spacing:
            x = spacing
            u *= boundary_damping
        elif x > box_length - spacing:
            x = box_length - spacing
            u *= boundary_damping

        if y < spacing:
            y = spacing
            v *= boundary_damping
        elif y > box_height - spacing:
            y = box_height - spacing
            v *= boundary_damping

        fluid_positions[i] = (x, y)
        fluid_velocities[i] = (u, v)
    
    return fluid_positions, fluid_velocities


def run_simulation(gravity, initial_density, num_time_steps, spacing, smoothing_length, isentropic_exponent, delta_t, box_length, box_height, fluid_length, fluid_height, boundary_damping, density_factor, pressure_factor, viscosity_factor, update_progress):
    fluid_positions, fluid_velocities = initialize_fluid(fluid_length, fluid_height, spacing)

    delta_t_collected = []  # List for collecting delta_t values
    positions_collected = [] # List for collecting position components
    velocities_collected = [] # List for collecting velocity components
    densities_collected = [] # List for collecting densities
    pressures_collected = [] # List for collecting pressures
    viscous_forces_collected = [] # List for collecting viscous forces

    iteration_start_time = time.perf_counter()  # Start time for measuring iterations

    for t in range(num_time_steps):
        iteration_step_start_time = time.perf_counter() # Start time for each iteration step

        # Get neighbors and their distances
        neighbors, distances = find_neighbors(fluid_positions, smoothing_length)
        # Calculate densities
        densities = calculate_density(density_factor, smoothing_length, distances)
        # Calculate pressures
        pressures = calculate_pressure(isentropic_exponent, initial_density, densities) 
        # Calculate forces due to pressure
        pressure_forces = calculate_pressure_force(pressure_factor, pressures, densities, distances, smoothing_length, fluid_positions)
        # Calculate forces due to viscosity
        viscous_forces = calculate_viscous_force(viscosity_factor, distances, smoothing_length, densities, fluid_velocities)
        # Calculate resultant viscous forces
        viscous_forces = np.array(viscous_forces)  # Ensure it is a NumPy array
        resultant_viscous_forces = np.sqrt(viscous_forces[:, 0]**2 + viscous_forces[:, 1]**2)
        # Sum up gravitational, viscous, and pressure forces
        total_forces = sum_up_forces(pressure_forces, viscous_forces, gravity)
        # Integrate the acceleration to get the velocities and positions
        fluid_positions, fluid_velocities = integrate_acceleration(fluid_positions, fluid_velocities, densities, delta_t, total_forces)
        # Enforce that the fluid particles stay within the defined boundary
        fluid_positions, fluid_velocities = enforce_boundary_condition(fluid_positions, fluid_velocities, box_length, box_height, spacing, boundary_damping)

        delta_t_collected.append(delta_t)  # Collect delta_t
        positions_collected.append(fluid_positions.copy()) # Collect positions
        velocities_collected.append(fluid_velocities.copy()) # Collect velocities
        densities_collected.append(densities.copy()) # Collect densities
        pressures_collected.append(pressures.copy()) # Collect pressures
        viscous_forces_collected.append(resultant_viscous_forces.copy()) # Collect resultant viscous forces

        iteration_step_end_time = time.perf_counter()  # End time for the iteration step
        step_time = iteration_step_end_time - iteration_step_start_time

        update_progress(t + 1, num_time_steps, step_time)  # Update progress

    iteration_end_time = time.perf_counter()  # End time for measuring iterations
    iteration_time = iteration_end_time - iteration_start_time  # Calculate time for iterations

    fluid_particles = [[], [], [], [], [], [], [], []]
    for t in range(num_time_steps):
        positions_x_collected = [pos[0] for pos in positions_collected[t]]
        positions_y_collected = [pos[1] for pos in positions_collected[t]]
        velocities_x_collected = [vel[0] for vel in velocities_collected[t]]
        velocities_y_collected = [vel[1] for vel in velocities_collected[t]]
        densities_collected_step = densities_collected[t]
        pressures_collected_step = pressures_collected[t]
        viscous_forces_collected_step = viscous_forces_collected[t]

        fluid_particles[0].append(positions_x_collected)
        fluid_particles[1].append(positions_y_collected)
        fluid_particles[2].append(velocities_x_collected)
        fluid_particles[3].append(velocities_y_collected)
        fluid_particles[4].append(densities_collected_step)
        fluid_particles[5].append(pressures_collected_step)
        fluid_particles[6].append(viscous_forces_collected_step)
        fluid_particles[7].append(delta_t_collected[t])

    array_build_end_time = time.perf_counter()  # End time for building the array
    array_build_time = array_build_end_time - iteration_end_time  # Calculate time for building the array

    total_time = iteration_time + array_build_time  # Calculate total time

    return fluid_particles, delta_t_collected, iteration_time, array_build_time