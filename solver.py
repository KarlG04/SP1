import numpy as np

def initialize_simulation(inlet_points, initial_velocity, num_time_steps):
    num_particles = len(inlet_points)
    Fluid_Properties = np.zeros((5, num_particles, num_time_steps))
    Fluid_Properties[0:2, :, 0] = np.array(inlet_points).T  # x, y Positionen
    Fluid_Properties[2:4, :, 0] = np.tile(initial_velocity, (num_particles, 1)).T  # x, y Geschwindigkeiten
    return Fluid_Properties

def calculate_time_step(velocities, delta_t_coefficient, initial_particle_distance):
    v_max = np.max(np.linalg.norm(velocities, axis=0))
    delta_t = delta_t_coefficient * initial_particle_distance / v_max
    return delta_t

def add_new_particles(Fluid_Properties, inlet_points, initial_velocity, t, num_time_steps):
    num_existing_particles = Fluid_Properties.shape[1]
    num_new_particles = len(inlet_points)
    new_particles = np.zeros((5, num_new_particles, num_time_steps))
    new_particles[0:2, :, t] = np.array(inlet_points).T  # x, y Positionen
    new_particles[2:4, :, t] = np.tile(initial_velocity, (num_new_particles, 1)).T  # x, y Geschwindigkeiten
    # Erweitern des Fluid_Properties-Arrays um neue Partikel
    Fluid_Properties = np.concatenate((Fluid_Properties, new_particles), axis=1)
    return Fluid_Properties

def update_velocity_due_to_gravity(Fluid_Properties, gravity, delta_t, time_step):
    Fluid_Properties[2:4, :, time_step] = Fluid_Properties[2:4, :, time_step - 1] + np.array(gravity)[:, None] * delta_t
    return Fluid_Properties

def update_positions(Fluid_Properties, delta_t, time_step):
    Fluid_Properties[0:2, :, time_step] = Fluid_Properties[0:2, :, time_step - 1] + Fluid_Properties[2:4, :, time_step] * delta_t
    return Fluid_Properties

def run_simulation(inlet_points, initial_velocity, gravity, delta_t_coefficient, initial_particle_distance, num_time_steps):
    Fluid_Properties = initialize_simulation(inlet_points, initial_velocity, num_time_steps)

    for t in range(1, num_time_steps):
        velocities = Fluid_Properties[2:4, :, t-1]
        delta_t = calculate_time_step(velocities, delta_t_coefficient, initial_particle_distance)
        
        # Prüfen, ob neue Partikel hinzugefügt werden sollen
        if t % int(1 / delta_t) == 0:  # Zeitintervall für das Hinzufügen neuer Partikel
            Fluid_Properties = add_new_particles(Fluid_Properties, inlet_points, initial_velocity, t, num_time_steps)

        Fluid_Properties = update_velocity_due_to_gravity(Fluid_Properties, gravity, delta_t, t)
        Fluid_Properties = update_positions(Fluid_Properties, delta_t, t)

    return Fluid_Properties