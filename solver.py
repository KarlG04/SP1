import numpy as np
import time
from scipy.spatial import KDTree

def initialize_fluid(fluid_width, fluid_height, spacing):
    # Initialize empty lists for the x and y coordinates of the inlet points
    inlet_points_x, inlet_points_y = [], []

    # Generate a uniform distribution of x-coordinates across the width of the fluid
    # The number of points is determined by fluid_width / spacing
    inlet_points_x = np.linspace(spacing, fluid_width, int(fluid_width / spacing))
    
    # Generate a uniform distribution of y-coordinates across the height of the fluid
    # The number of points is determined by fluid_height / spacing
    inlet_points_y = np.linspace(spacing, fluid_height, int(fluid_height / spacing))

    # Create a grid from the x and y coordinates
    inlet_points_x, inlet_points_y = np.meshgrid(inlet_points_x, inlet_points_y)
    
    # Combine the x and y coordinates into pairs representing the inlet points
    # np.vstack stacks the arrays vertically, ravel flattens the arrays, and .T transposes them
    inlet_points = np.vstack([inlet_points_x.ravel(), inlet_points_y.ravel()]).T

    # Add a random offset to each point to introduce variability
    # The offset is uniformly distributed within the range [0, 0.01 * spacing]
    random_shift = np.random.uniform(0, 0.01 * spacing, inlet_points.shape)
    inlet_points += random_shift
    
    # Convert inlet_points into a list of tuples for easier handling
    fluid_positions = [tuple(point) for point in inlet_points]
    
    # Initialize velocities to (0, 0) for each point, representing stationary fluid
    fluid_velocities = [(0.0, 0.0) for _ in fluid_positions]

    return fluid_positions, fluid_velocities

def find_neighbors(fluid_positions, smoothing_length):
    # Get the number of particles
    num_particles = len(fluid_positions)
    
    # Initialize lists to store the neighbors and their corresponding distances for each particle
    neighbors = [[] for _ in range(num_particles)]
    distances = [[] for _ in range(num_particles)]
    
    # Create a KDTree for efficient spatial searches
    tree = KDTree(fluid_positions)
    
    # Iterate over each particle to find its neighbors
    for i, position in enumerate(fluid_positions):
        # Find all particles within the smoothing_length radius of the current particle
        neighbor_indices = tree.query_ball_point(position, smoothing_length)
        
        # Iterate over each neighboring particle index
        for index in neighbor_indices:
            # Exclude the particle itself from its list of neighbors
            if index != i:
                # Append the neighbor's index to the neighbors list of the current particle
                neighbors[i].append(index)
                
                # Calculate the Euclidean distance between the current particle and the neighbor
                distance = np.linalg.norm(np.array(position) - np.array(fluid_positions[index]))
                
                # Append the distance along with the neighbor's index to the distances list
                distances[i].append((index, distance))

    # Return the list of neighbors and their corresponding distances for each particle
    return neighbors, distances

def calculate_density(mass_per_particle, smoothing_length, distances):
    # Get the number of particles
    num_particles = len(distances)
    
    # Initialize densities to zero for all particles
    densities = [0.0] * num_particles

    # Precompute the density factor constant
    density_factor = (315 * mass_per_particle) / (64 * np.pi * smoothing_length**9)

    # Calculate the self-density for each particle
    self_density = density_factor * (smoothing_length**2)**3

    # Loop over each particle to calculate its density
    for i in range(num_particles):
        density_sum = 0.0
        # Sum the contributions from neighboring particles
        for j, dij in distances[i]:
            density_sum += density_factor * (smoothing_length**2 - dij**2)**3
        # Add the self-density to the sum to get the total density for the particle
        densities[i] = self_density + density_sum

    return densities

def calculate_pressure(isentropic_exponent, rest_density, densities):
    # Initialize pressures to zero for all particles
    pressures = [0.0] * len(densities)
    
    # Loop over each particle to calculate its pressure
    for i in range(len(densities)):
        # Calculate pressure using the isentropic exponent and the difference between the current and rest density
        pressures[i] = isentropic_exponent * (densities[i] - rest_density)

    return pressures

def calculate_pressure_force(mass_per_particle, pressures, densities, distances, smoothing_length, fluid_positions):
    # Get the number of particles
    num_particles = len(fluid_positions)
    
    # Initialize the pressure forces to zero for all particles
    pressure_forces = [(0.0, 0.0) for _ in range(num_particles)]

    # Precompute the pressure factor constant
    pressure_factor = (-(45 * mass_per_particle) / (np.pi * smoothing_length**6))

    # Loop over each particle to calculate the pressure force
    for i in range(num_particles):
        sum_force_x = 0.0
        sum_force_y = 0.0
        xi, yi = fluid_positions[i]  # Position of particle i
        pi = pressures[i]            # Pressure of particle i
        
        # Sum the contributions from neighboring particles
        for j, dij in distances[i]:
            xj, yj = fluid_positions[j]  # Position of neighbor j
            pj = pressures[j]            # Pressure of neighbor j
            rhoj = densities[j]          # Density of neighbor j
            
            # Calculate the force contribution from neighbor j
            term = ((smoothing_length - dij)**2 * (pj + pi)) / ((2 * rhoj) * dij)
             
            sum_force_x += pressure_factor * -(xj - xi) * term
            sum_force_y += pressure_factor * -(yj - yi) * term
        
        # Store the calculated force for particle i
        pressure_forces[i] = (sum_force_x, sum_force_y)

    return pressure_forces

def calculate_viscous_force(mass_per_particle, dynamic_viscosity, distances, smoothing_length, densities, fluid_velocities):
    # Get the number of particles
    num_particles = len(distances)
    
    # Initialize the viscous forces to zero for all particles
    viscous_forces = [(0.0, 0.0) for _ in range(num_particles)]

    # Precompute the viscosity factor constant
    viscosity_factor = (45 * dynamic_viscosity * mass_per_particle) / (np.pi * smoothing_length**6)

    # Loop over each particle to calculate the viscous force
    for i in range(num_particles):
        sum_force_x = 0.0
        sum_force_y = 0.0
        ui, vi = fluid_velocities[i]  # Velocity of particle i

        # Sum the contributions from neighboring particles
        for j, dij in distances[i]:
            uj, vj = fluid_velocities[j]  # Velocity of neighbor j
            rhoj = densities[j]           # Density of neighbor j

            # Calculate the force contribution from neighbor j
            term = (smoothing_length - dij) / rhoj

            sum_force_x += viscosity_factor * (uj - ui) * term
            sum_force_y += viscosity_factor * (vj - vi) * term

        # Store the calculated force for particle i
        viscous_forces[i] = (sum_force_x, sum_force_y)

    return viscous_forces

def sum_up_forces(pressure_forces, viscous_forces, gravity, densities):
    # Get the number of particles
    num_particles = len(pressure_forces)
    
    # Initialize the total forces to zero for all particles
    total_forces = [(0.0, 0.0) for _ in range(num_particles)]

    # Loop over each particle to sum up the forces
    for i in range(num_particles):
        px, py = pressure_forces[i]  # Pressure force on particle i
        vx, vy = viscous_forces[i]   # Viscous force on particle i
        gx, gy = gravity             # Gravity force (assumed to be constant for all particles)
        rho = densities[i]           # Density of particle i

        # Multiply gravity by the density to get the gravitational force density
        gx *= rho
        gy *= rho

        # Sum the x and y components of the forces
        total_force_x = px + vx + gx
        total_force_y = py + vy + gy

        # Store the calculated total force for particle i
        total_forces[i] = (total_force_x, total_force_y)

    return total_forces

def integrate_acceleration(fluid_positions, fluid_velocities, densities, delta_t, total_forces):
    # Get the number of particles
    num_particles = len(fluid_positions)
    
    # Initialize the new positions and velocities to zero for all particles
    new_fluid_positions = [(0.0, 0.0) for _ in range(num_particles)]
    new_fluid_velocities = [(0.0, 0.0) for _ in range(num_particles)]
    
    # Loop over each particle to update its position and velocity
    for i in range(num_particles):
        xi, yi = fluid_positions[i]   # Current position of particle i
        ui, vi = fluid_velocities[i]  # Current velocity of particle i
        rhoi = densities[i]           # Density of particle i
        fx, fy = total_forces[i]      # Total force acting on particle i

        # Update velocity using the acceleration (force/density) and the time step delta_t
        new_ui = ui + delta_t * fx / rhoi
        new_vi = vi + delta_t * fy / rhoi

        # Update position using the new velocity and the time step delta_t
        new_xi = xi + delta_t * new_ui
        new_yi = yi + delta_t * new_vi

        # Store the new position and velocity for particle i
        new_fluid_positions[i] = (new_xi, new_yi)
        new_fluid_velocities[i] = (new_ui, new_vi)

    return new_fluid_positions, new_fluid_velocities

def enforce_boundary_condition(fluid_positions, fluid_velocities, box_width, box_height, spacing, boundary_damping):
    # Loop over each particle to enforce boundary conditions
    for i in range(len(fluid_positions)):
        x, y = fluid_positions[i]  # Current position of particle i
        u, v = fluid_velocities[i] # Current velocity of particle i

        # Check and enforce boundary condition on the left side
        if x < spacing:
            x = spacing
            u *= boundary_damping  # Apply damping to velocity upon collision
        
        # Check and enforce boundary condition on the right side
        elif x > box_width - spacing:
            x = box_width - spacing
            u *= boundary_damping  # Apply damping to velocity upon collision

        # Check and enforce boundary condition on the bottom side
        if y < spacing:
            y = spacing
            v *= boundary_damping  # Apply damping to velocity upon collision
        
        # Check and enforce boundary condition on the top side
        elif y > box_height - spacing:
            y = box_height - spacing
            v *= boundary_damping  # Apply damping to velocity upon collision

        # Update the position and velocity for particle i after enforcing boundaries
        fluid_positions[i] = (x, y)
        fluid_velocities[i] = (u, v)
    
    return fluid_positions, fluid_velocities

def run_simulation(gravity, rest_density, num_time_steps, spacing, smoothing_length, isentropic_exponent, delta_t, box_width, box_height, fluid_width, fluid_height, boundary_damping, mass_per_particle, dynamic_viscosity, update_progress):
    fluid_positions, fluid_velocities = initialize_fluid(fluid_width, fluid_height, spacing)

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
        densities = calculate_density(mass_per_particle, smoothing_length, distances)
        # Calculate pressures
        pressures = calculate_pressure(isentropic_exponent, rest_density, densities) 
        # Calculate forcedensity due to pressure
        pressure_forces = calculate_pressure_force(mass_per_particle, pressures, densities, distances, smoothing_length, fluid_positions)
        # Calculate forcedenstiy due to viscosity
        viscous_forces = calculate_viscous_force(mass_per_particle, dynamic_viscosity, distances, smoothing_length, densities, fluid_velocities)
        # Calculate resultant viscous forces
        viscous_forces = np.array(viscous_forces)  # Ensure it is a NumPy array
        resultant_viscous_forces = np.sqrt(viscous_forces[:, 0]**2 + viscous_forces[:, 1]**2)
        # Sum up gravitational, viscous, and pressure forces
        total_forces = sum_up_forces(pressure_forces, viscous_forces, gravity, densities)
        # Integrate the acceleration to get the velocities and positions
        fluid_positions, fluid_velocities = integrate_acceleration(fluid_positions, fluid_velocities, densities, delta_t, total_forces)
        # Enforce that the fluid particles stay within the defined boundary
        fluid_positions, fluid_velocities = enforce_boundary_condition(fluid_positions, fluid_velocities, box_width, box_height, spacing, boundary_damping)

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

    # Build the fluid_particles Array
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