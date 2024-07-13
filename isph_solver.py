import numpy as np
import time
from scipy.spatial import KDTree
import math

def initialize_fluid(inlet_points):
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

def calculate_intermediate_values(gravity, fluid_positions, fluid_velocities, delta_t):
    num_particles = len(fluid_positions)
    g_x, g_y = gravity

    for i in range(num_particles):
        u_i, v_i = fluid_velocities[i]
        x_i, y_i = fluid_positions[i]

        # Neue Geschwindigkeiten berechnen
        u_new = u_i + g_x * delta_t
        v_new = v_i + g_y * delta_t

        # Neue Positionen berechnen
        x_new = x_i + u_new * delta_t
        y_new = y_i + v_new * delta_t

        # Aktualisiere die Listen
        fluid_velocities[i] = (u_new, v_new)
        fluid_positions[i] = (x_new, y_new)

    return fluid_positions, fluid_velocities

def calculate_grad_w(smoothing_length, distances, fluid_positions):
    num_particles = len(fluid_positions)
    grad_w = [[] for _ in range(num_particles)]

    alpha_d = 10 / (7 * np.pi * smoothing_length ** 2)

    for i in range(num_particles):
        for j, dist in distances[i]:
            q = dist / smoothing_length
            if 0 <= q <= 1:
                grad_w_scalar = alpha_d * (-3 * q + (9 / 4) * q ** 2)
            elif 1 < q <= 2:
                grad_w_scalar = alpha_d * (-3 / 4) * (2 - q) ** 2
            else:
                grad_w_scalar = 0

            x_i, y_i = fluid_positions[i]
            x_j, y_j = fluid_positions[j]
            r_ij = np.array([x_i - x_j, y_i - y_j])
            r_ij_norm = np.linalg.norm(r_ij)

            if r_ij_norm > 0:
                grad_w_ij = grad_w_scalar * r_ij / (smoothing_length * r_ij_norm)
            else:
                grad_w_ij = np.array([0.0, 0.0])

            grad_w[i].append((j, tuple(grad_w_ij)))

    return grad_w

def calculate_viscosity(mass_per_particle, eta, kinematic_viscosity, fluid_positions, fluid_velocities, grad_w, densities):
    num_particles = len(fluid_positions)
    viscosities = [(0.0, 0.0) for _ in range(num_particles)]

    for i in range(num_particles):
        visc_acceleration_x = 0.0
        visc_acceleration_y = 0.0
        u_i, v_i = fluid_velocities[i]
        rho_i = densities[i]
        nu_i = kinematic_viscosity
        
        for j, grad_w_ij in grad_w[i]:
            m_j = mass_per_particle
            u_j, v_j = fluid_velocities[j]
            rho_j = densities[j]
            nu_j = kinematic_viscosity
            x_i, y_i = fluid_positions[i]
            x_j, y_j = fluid_positions[j]
            
            x_ij = np.array([x_i - x_j, y_i - y_j])
            x_ij_norm_sq = np.dot(x_ij, x_ij)  # This is |x_ij|^2
            u_diff = np.array([u_i - u_j, v_i - v_j])
            dot_product = np.dot(u_diff, x_ij)
            grad_w_x, grad_w_y = grad_w_ij

            common_term = m_j * (8 * (nu_i + nu_j) * dot_product) / ((rho_i + rho_j) * (x_ij_norm_sq + eta**2))
            visc_acceleration_x += common_term * grad_w_x
            visc_acceleration_y += common_term * grad_w_y
        
        viscosities[i] = (visc_acceleration_x, visc_acceleration_y)

    return viscosities

def calculate_densities(initial_density, mass_per_particle, fluid_velocities, grad_w):
    num_particles = len(fluid_velocities)
    densities = [0.0 for _ in range(num_particles)]

    for i in range(num_particles):
        u_i, v_i = fluid_velocities[i]
        sum_term = 0.0
        for j, grad_w_ij in grad_w[i]:
            m_j = mass_per_particle
            u_j, v_j = fluid_velocities[j]
            du = u_i - u_j
            dv = v_i - v_j
            grad_w_x, grad_w_y = grad_w_ij
            sum_term += m_j * (du * grad_w_x + dv * grad_w_y)
        
        calculated_density = initial_density + sum_term
        if calculated_density < 0.99 * initial_density:
            densities[i] = initial_density
        else:
            densities[i] = calculated_density

    return densities

def calculate_pressures(pressures, initial_density, mass_per_particle, densities, delta_t, fluid_positions, grad_w, eta):
    num_particles = len(fluid_positions)
    new_pressures = [0.0 for _ in range(num_particles)]

    for i in range(num_particles):
        rho_i = densities[i]
        p_i_term1 = (initial_density - rho_i) / (delta_t ** 2)
        sum_term1 = 0.0
        sum_term2 = 0.0

        for j, grad_w_ij in grad_w[i]:
            m_j = mass_per_particle
            p_j = pressures[j]
            rho_j = densities[j]
            x_i, y_i = fluid_positions[i]
            x_j, y_j = fluid_positions[j]
            
            r_ij = np.array([x_i - x_j, y_i - y_j])
            r_ij_norm_sq = np.dot(r_ij, r_ij)  # This is |r_ij|^2
            grad_w_x, grad_w_y = grad_w_ij

            numerator1 = 8 * m_j * p_j * np.dot(r_ij, [grad_w_x, grad_w_y])
            denominator1 = (rho_i + rho_j) ** 2 * (r_ij_norm_sq + eta ** 2)
            sum_term1 += numerator1 / denominator1

            numerator2 = 8 * m_j * np.dot(r_ij, [grad_w_x, grad_w_y])
            sum_term2 += numerator2 / denominator1


        p_i_term2 = sum_term1
        p_i = (p_i_term1 + p_i_term2) / sum_term2
        new_pressures[i] = p_i

    return new_pressures

def calculate_corrective_velocities(delta_t, grad_w, mass_per_particle, pressures, densities):
    num_particles = len(pressures)
    corrective_velocities = [(0.0, 0.0) for _ in range(num_particles)]

    for i in range(num_particles):
        pressure_i = pressures[i]
        density_i = densities[i]
        correction_x = 0.0
        correction_y = 0.0
        
        for j, grad_w_ij in grad_w[i]:
            pressure_j = pressures[j]
            density_j = densities[j]
            m_j = mass_per_particle
            grad_w_x, grad_w_y = grad_w_ij
            
            pressure_term = (pressure_j / (density_j ** 2)) + (pressure_i / (density_i ** 2))
            
            correction_x += m_j * pressure_term * grad_w_x
            correction_y += m_j * pressure_term * grad_w_y
        
        corrective_velocities[i] = (-delta_t * correction_x, -delta_t * correction_y)

    return corrective_velocities

def calculate_final_values(last_positions, last_velocities, fluid_velocities, corrective_velocities, delta_t):
    num_particles = len(last_positions)

    for i in range(num_particles):
        x_last, y_last = last_positions[i]
        u_last, v_last = last_velocities[i]
        u_corr, v_corr = corrective_velocities[i]
        u_current, v_current = fluid_velocities[i]

        # Berechne die finalen Geschwindigkeiten
        u_final = u_current + u_corr
        v_final = v_current + v_corr

        # Berechne die neuen Positionen basierend auf den finalen Geschwindigkeiten
        x_new = x_last + delta_t * 0.5 * (u_final + u_last)
        y_new = y_last + delta_t * 0.5 * (v_final + v_last)

        # Aktualisiere die Listen
        fluid_velocities[i] = (u_final, v_final)
        fluid_positions[i] = (x_new, y_new)

    return fluid_positions, fluid_velocities

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

def run_simulation(mass_per_particle, inlet_points, gravity, initial_density, num_time_steps, spacing, smoothing_length, eta, delta_t, box_length, box_height, boundary_damping):
    # Initialisieren der Simulation
    fluid_positions, fluid_velocities = initialize_fluid(inlet_points)
    num_particles = len(fluid_positions)
    pressures = [0.0 for _ in range(num_particles)]

    # Gesuchte Werte für jeden Zeitschritt initialisieren
    delta_t_collected = []  # Liste zum Speichern der delta_t Werte
    positions_collected = [] # Liste zum Speichern der Positionen
    velocities_collected = [] # Liste zum Speichern der Geschwindigkeitskomponenten
    mirror_positions_collected = [] # Liste zum Speichern der Positionen der Spiegelpartikel
    mirror_velocities_collected = [] # Liste zum Speichern der Geschwindigkeitskomponenten der Spiegelpartikel

    iteration_start_time = time.perf_counter()  # Startzeit für Iterationen messen

    for t in range(num_time_steps):
        print(f"Running iteration {t+1}/{num_time_steps} | ", end="")  # Ausgabe der aktuellen Iterationsnummer
        iteration_step_start_time = time.perf_counter() # Startzeit für jeden einzelnen Iterationsschritt

        last_positions = fluid_positions
        last_velocities = fluid_velocities

        neighbors, distances = find_neighbors(fluid_positions, smoothing_length)

        fluid_positions, fluid_velocities = calculate_intermediate_values(gravity, fluid_positions, fluid_velocities, delta_t)

        fluid_positions, fluid_velocities = enforce_boundary_condition(fluid_positions, fluid_velocities, box_length, box_height, spacing, boundary_damping)

        neighbors, distances = find_neighbors(fluid_positions, smoothing_length)

        grad_w = calculate_grad_w(smoothing_length, distances, fluid_positions)

        densities = calculate_densities(initial_density, mass_per_particle, fluid_velocities, grad_w)

        pressures = calculate_pressures(pressures, initial_density, mass_per_particle, densities, delta_t, fluid_positions, grad_w, eta)
        
        corrective_velocities = calculate_corrective_velocities(delta_t, grad_w, mass_per_particle, pressures, densities)

        fluid_positions, fluid_velocities = calculate_final_values(last_positions, last_velocities, fluid_velocities, corrective_velocities, delta_t)        

        # Ergebnisse für den aktuellen Zeitschritt speichern
        delta_t_collected.append(delta_t)  # delta_t sammeln
        positions_collected.append(fluid_positions.copy()) # Positionen sammeln
        velocities_collected.append(fluid_velocities.copy()) # Geschwindigkeiten sammeln

        # Zeit für iterationsschritt berechnen und ausgeben
        iteration_step_end_time = time.perf_counter()
        iteration_step_time = iteration_step_end_time - iteration_step_start_time 
        print(f"{iteration_step_time:.2f}s")

    # Iterationszeit berechnen und ausgeben
    iteration_end_time = time.perf_counter()  # Endzeit für Iterationen messen
    iteration_time = iteration_end_time - iteration_start_time  # Zeit für Iterationen berechnen
    # Zeit in Stunden, Minuten und Sekunden umwandeln
    iteration_time_hours = int(iteration_time // 3600)
    iteration_time_minutes = int((iteration_time % 3600) // 60)
    iteration_time_seconds = iteration_time % 60
    print(f"Iteration time: {iteration_time_hours}h {iteration_time_minutes}m {iteration_time_seconds:.2f}s")
    print(" ")

    # Fluid_particles in Listenform initialisieren
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

    # Array build time berechnen und ausgeben
    array_build_end_time = time.perf_counter()  # Endzeit für das Erstellen des Arrays messen
    array_build_time = array_build_end_time - iteration_end_time  # Zeit für das Erstellen des Arrays berechnen
    # Zeit in Stunden, Minuten und Sekunden umwandeln
    array_build_time_hours = int(array_build_time // 3600)
    array_build_time_minutes = int((array_build_time % 3600) // 60)
    array_build_time_seconds = array_build_time % 60
    print(f"Array build time: {array_build_time_hours}h {array_build_time_minutes}m {array_build_time_seconds:.2f}s")
    print(" ")

    # Gesamtzeit berechnen und ausgeben
    total_time = iteration_time + array_build_time  # Gesamtzeit berechnen
    # Zeit in Stunden, Minuten und Sekunden umwandeln
    total_time_hours = int(total_time // 3600)
    total_time_minutes = int((total_time % 3600) // 60)
    total_time_seconds = total_time % 60
    print(f"Simulation completed in: {total_time_hours}h {total_time_minutes}m {total_time_seconds:.2f}s")
    print(" ")

    return fluid_particles, delta_t_collected