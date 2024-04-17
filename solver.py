import numpy as np

def initialize_simulation(inlet_points, initial_velocity):
    # Initialize lists for positions and velocities
    positions_x, positions_y, velocities_x, velocities_y = [], [], [], []
    for i in range(len(inlet_points)):
        positions_x.append([inlet_points[i][0]])
        positions_y.append([inlet_points[i][1]])
        velocities_x.append([initial_velocity[0]])
        velocities_y.append([initial_velocity[1]])
    return positions_x, positions_y, velocities_x, velocities_y

def calculate_time_step(velocities_x, velocities_y, delta_t_coefficient, initial_particle_distance):
    velocities = np.sqrt(np.square(velocities_x) + np.square(velocities_y))
    v_max = np.max(velocities)
    delta_t = delta_t_coefficient * initial_particle_distance / v_max
    return delta_t

def add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity, t):
    # Extend existing lists with new particles
    for i in range(len(inlet_points)):
        positions_x.append([inlet_points[i][0]] * (t + 1))
        positions_y.append([inlet_points[i][1]] * (t + 1))
        velocities_x.append([initial_velocity[0]] * (t + 1))
        velocities_y.append([initial_velocity[1]] * (t + 1))

def calculate_particle_add_interval(initial_velocity, initial_particle_distance):
    initial_velocity_magnitude = np.linalg.norm(initial_velocity)
    return initial_particle_distance / initial_velocity_magnitude

def update_velocity_due_to_gravity(velocities_x, velocities_y, gravity, delta_t, t):
    # Update velocity lists for gravity effect
    for i in range(len(velocities_x)):
        velocities_x[i].append(velocities_x[i][t] + gravity[0] * delta_t)
        velocities_y[i].append(velocities_y[i][t] + gravity[1] * delta_t)

def update_positions(positions_x, positions_y, velocities_x, velocities_y, delta_t, t):
    # Update position lists based on new velocities
    for i in range(len(positions_x)):
        positions_x[i].append(positions_x[i][t] + velocities_x[i][t+1] * delta_t)
        positions_y[i].append(positions_y[i][t] + velocities_y[i][t+1] * delta_t)

def run_simulation(inlet_points, initial_velocity, gravity, delta_t_coefficient, initial_particle_distance, num_time_steps):
    # Initialize simulation
    positions_x, positions_y, velocities_x, velocities_y = initialize_simulation(inlet_points, initial_velocity)
    delta_ts = []  # List to store delta_t values for each time step
    
    # Calculate the interval for adding new particles
    particle_add_interval = calculate_particle_add_interval(initial_velocity, initial_particle_distance)
    time_since_last_addition = 0  # Initialize time counter for particle addition

    for t in range(num_time_steps - 1):
        print(f"Running iteration {t+2}/{num_time_steps}")  # Outputs the current iteration number
        # Calculate time step based on velocities
        delta_t = calculate_time_step([vx[t] for vx in velocities_x], [vy[t] for vy in velocities_y],
                                      delta_t_coefficient, initial_particle_distance)
        delta_ts.append(delta_t)  # Collect delta_t
        time_since_last_addition += delta_t  # Update time since last addition

        # Check if it's time to add new particles
        if time_since_last_addition >= particle_add_interval:
            add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity, t)
            time_since_last_addition = 0  # Reset the time counter after adding particles
        
        # Update velocities due to gravity
        update_velocity_due_to_gravity(velocities_x, velocities_y, gravity, delta_t, t)
        
        # Update positions
        update_positions(positions_x, positions_y, velocities_x, velocities_y, delta_t, t)
    
    # Convert lists to numpy arrays for output
    print("Build Fluid_Points array")
    Fluid_Points = np.array([positions_x, positions_y, velocities_x, velocities_y])
    print("done...")
    return Fluid_Points, delta_ts


