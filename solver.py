import numpy as np
import time
from tqdm import tqdm
from scipy.spatial import KDTree

def initialize_fluid(fluid_length, fluid_height, spacing):
    inlet_points_x, inlet_points_y = [], []

    inlet_points_x = np.linspace(spacing, fluid_length, int(fluid_length / spacing))
    inlet_points_y = np.linspace(spacing, fluid_height, int(fluid_height / spacing))

    inlet_points_x, inlet_points_y = np.meshgrid(inlet_points_x, inlet_points_y)
    inlet_points = np.vstack([inlet_points_x.ravel(), inlet_points_y.ravel()]).T

    # Hinzufügen der zufälligen Verschiebung
    random_shift = np.random.uniform(0, 0.1, inlet_points.shape)
    inlet_points += random_shift
    
    # Konvertiere inlet_points in eine Liste von Tupeln
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

        # Berechnung der neuen Geschwindigkeiten
        new_ui = ui + delta_t * fx / rhoi
        new_vi = vi + delta_t * fy / rhoi

        # Berechnung der neuen Positionen
        new_xi = xi + delta_t * new_ui
        new_yi = yi + delta_t * new_vi

        # Speichern der neuen Positionen und Geschwindigkeiten
        new_fluid_positions[i] = (new_xi, new_yi)
        new_fluid_velocities[i] = (new_ui, new_vi)

    return new_fluid_positions, new_fluid_velocities

def enforce_boundary_condition(fluid_positions, fluid_velocities, box_length, box_height, spacing, boundary_damping):
    for i in range(len(fluid_positions)):
        x, y = fluid_positions[i]
        u, v = fluid_velocities[i]

        # Überprüfe und korrigiere die x-Komponente der Position
        if x < spacing:
            x = spacing
            u *= boundary_damping
        elif x > box_length - spacing:
            x = box_length - spacing
            u *= boundary_damping

        # Überprüfe und korrigiere die y-Komponente der Position
        if y < spacing:
            y = spacing
            v *= boundary_damping
        elif y > box_height - spacing:
            y = box_height - spacing
            v *= boundary_damping

        # Speichere die aktualisierten Positionen und Geschwindigkeiten
        fluid_positions[i] = (x, y)
        fluid_velocities[i] = (u, v)
    
    return fluid_positions, fluid_velocities



# solver.py
import numpy as np
import time

def run_simulation(gravity, initial_density, num_time_steps, spacing, smoothing_length, isentropic_exponent, delta_t, box_length, box_height, fluid_length, fluid_height, boundary_damping, density_factor, pressure_factor, viscosity_factor, update_progress):
    fluid_positions, fluid_velocities = initialize_fluid(fluid_length, fluid_height, spacing)
    num_particles = len(fluid_positions)

    delta_t_collected = []  # Liste zum Speichern der delta_t Werte
    positions_collected = [] # Liste zum Speichern der Positionen
    velocities_collected = [] # Liste zum Speichern der Geschwindigkeitskomponenten

    iteration_start_time = time.perf_counter()  # Startzeit für Iterationen messen

    for t in range(num_time_steps):
        iteration_step_start_time = time.perf_counter() # Startzeit für jeden einzelnen Iterationsschritt

        neighbors, distances = find_neighbors(fluid_positions, smoothing_length)
        densities = calculate_density(density_factor, smoothing_length, distances)
        pressures = calculate_pressure(isentropic_exponent, initial_density, densities) 
        pressure_forces = calculate_pressure_force(pressure_factor, pressures, densities, distances, smoothing_length, fluid_positions)
        viscous_forces = calculate_viscous_force(viscosity_factor, distances, smoothing_length, densities, fluid_velocities)
        total_forces = sum_up_forces(pressure_forces, viscous_forces, gravity)
        fluid_positions, fluid_velocities = integrate_acceleration(fluid_positions, fluid_velocities, densities, delta_t, total_forces)
        fluid_positions, fluid_velocities = enforce_boundary_condition(fluid_positions, fluid_velocities, box_length, box_height, spacing, boundary_damping)

        delta_t_collected.append(delta_t)  # delta_t sammeln
        positions_collected.append(fluid_positions.copy()) # Positionen sammeln
        velocities_collected.append(fluid_velocities.copy()) # Geschwindigkeiten sammeln

        iteration_step_end_time = time.perf_counter()  # Endzeit für den Iterationsschritt
        step_time = iteration_step_end_time - iteration_step_start_time

        update_progress(t + 1, num_time_steps, step_time)  # Update progress

    iteration_end_time = time.perf_counter()  # Endzeit für Iterationen messen
    iteration_time = iteration_end_time - iteration_start_time  # Zeit für Iterationen berechnen

    fluid_particles = [[], [], [], []]
    for t in range(num_time_steps):
        positions_x_collected = [pos[0] for pos in positions_collected[t]]
        positions_y_collected = [pos[1] for pos in positions_collected[t]]
        velocities_x_collected = [vel[0] for vel in velocities_collected[t]]
        velocities_y_collected = [vel[1] for vel in velocities_collected[t]]

        fluid_particles[0].append(positions_x_collected)
        fluid_particles[1].append(positions_y_collected)
        fluid_particles[2].append(velocities_x_collected)
        fluid_particles[3].append(velocities_y_collected)

    array_build_end_time = time.perf_counter()  # Endzeit für das Erstellen des Arrays messen
    array_build_time = array_build_end_time - iteration_end_time  # Zeit für das Erstellen des Arrays berechnen

    total_time = iteration_time + array_build_time  # Gesamtzeit berechnen

    return fluid_particles, delta_t_collected, iteration_time, array_build_time

