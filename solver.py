import numpy as np
import time
from scipy.spatial import KDTree
import math

def initialize_fluid(inlet_points):
    # Konvertiere inlet_points in eine Liste von Tupeln
    fluid_positions = [tuple(point) for point in inlet_points]
    fluid_velocities = [(0.0, 0.0) for _ in fluid_positions]  # Initialize velocities to (0, 0) for each point
    
    return fluid_positions, fluid_velocities

def calculate_timestep(fluid_velocities, cfl, spacing):
    # Berechne die Gesamtgeschwindigkeit für jeden Partikel
    velocities = [math.sqrt(vx**2 + vy**2) for vx, vy in fluid_velocities]
    
    # Bestimme die maximale Geschwindigkeit
    vmax = max(velocities)
    
    # Berechne delta_t
    delta_t = cfl * spacing / vmax
    
    return delta_t

def find_neighbors(fluid_positions, h):
    num_particles = len(fluid_positions)
    neighbors = [[] for _ in range(num_particles)]
    tree = KDTree(fluid_positions)
    for i, position in enumerate(fluid_positions):
        neighbor_indices = tree.query_ball_point(position, h)
        neighbors[i] = [index for index in neighbor_indices if index != i]
    return neighbors

def compute_grad_w(fluid_positions, neighbors, h):
    alpha_d = 10 / (7 * np.pi * h**2)
    def dW(q):
        if 0 <= q <= 1:
            return alpha_d * (-3 * q + 9 / 4 * q**2)
        elif 1 < q <= 2:
            return alpha_d * (-3 / 4 * (2 - q)**2)
        else:
            return 0
    grad_w = []
    for i, position_i in enumerate(fluid_positions):
        for j in neighbors[i]:
            position_j = fluid_positions[j]
            r_ij = np.array(position_i) - np.array(position_j)
            distance = np.linalg.norm(r_ij)
            q = distance / h
            if distance != 0:
                grad_W_ij = dW(q) * (r_ij / (distance * h))
                grad_w.append((i, j, (grad_W_ij[0], grad_W_ij[1])))
    return grad_w

def calculate_viscosity_acceleration(fluid_positions, fluid_velocities, neighbors, grad_w, nu, eta, mass_per_particle, current_rho):
    num_particles = len(fluid_positions)
    viscosity_acceleration = [(0.0, 0.0) for _ in range(num_particles)]
    for i in range(num_particles):
        acc_viscosity_x = 0
        acc_viscosity_y = 0
        for j in neighbors[i]:
            grad_W_ij_x, grad_W_ij_y = next((grad for (pi, pj, grad) in grad_w if pi == i and pj == j), (0, 0))
            u_diff_x = fluid_velocities[i][0] - fluid_velocities[j][0]
            u_diff_y = fluid_velocities[i][1] - fluid_velocities[j][1]
            r_ij = np.array(fluid_positions[i]) - np.array(fluid_positions[j])
            distance = np.linalg.norm(r_ij)
            acc_viscosity_x += mass_per_particle * 8 * (nu + nu) * u_diff_x / ((distance**2 + eta**2) * (current_rho[i] + current_rho[j])) * grad_W_ij_x
            acc_viscosity_y += mass_per_particle * 8 * (nu + nu) * u_diff_y / ((distance**2 + eta**2) * (current_rho[i] + current_rho[j])) * grad_W_ij_y
        viscosity_acceleration[i] = (acc_viscosity_x, acc_viscosity_y)
    return viscosity_acceleration

def first_step_velocities(fluid_positions, fluid_velocities, delta_t, gravity, viscosity_acceleration):
    updated_velocities = []
    updated_positions = []
    g_x, g_y = gravity
    
    for (u, v), (visc_u, visc_v), (x, y) in zip(fluid_velocities, viscosity_acceleration, fluid_positions):
        # Berechnung der neuen Geschwindigkeiten
        u_new = u + g_x * delta_t + visc_u * delta_t
        v_new = v + g_y * delta_t + visc_v * delta_t
        updated_velocities.append((u_new, v_new))
        
        # Berechnung der neuen Positionen
        x_new = x + u_new * delta_t
        y_new = y + v_new * delta_t
        updated_positions.append((x_new, y_new))
    
    return updated_positions, updated_velocities


def calculate_density(current_rho, fluid_velocities, neighbors, grad_w, delta_t, mass_per_particle, initial_rho):
    num_particles = len(fluid_velocities)
    new_rho = [0.0 for _ in range(num_particles)]
    for i in range(num_particles):
        sum_term = 0
        for j in neighbors[i]:
            grad_W_ij_x, grad_W_ij_y = next((grad for (pi, pj, grad) in grad_w if pi == i and pj == j), (0, 0))
            u_diff_x = fluid_velocities[i][0] - fluid_velocities[j][0]
            u_diff_y = fluid_velocities[i][1] - fluid_velocities[j][1]
            sum_term += mass_per_particle * (u_diff_x * grad_W_ij_x + u_diff_y * grad_W_ij_y)
        new_rho[i] = initial_rho + sum_term
        if new_rho[i] < 0.99 * initial_rho:
            new_rho[i] = initial_rho
    print(new_rho)
    
    return new_rho

def calculate_pressure(initial_rho, current_rho, delta_t, mass_per_particle, current_pressure, fluid_positions, eta, grad_w):
    num_particles = len(current_pressure)
    new_pressure = [0.0] * num_particles
    
    for i in range(num_particles):
        rho_i = current_rho[i]

        term1 = (initial_rho - rho_i) / delta_t ** 2
        term2 = 0.0
        term3 = 0.0
        
        for (i_index, j_index, grad_w_ij) in grad_w:
            if i_index == i and j_index != i:
                rho_j = current_rho[j_index]
                p_j = current_pressure[j_index]

                # Berechnung von r_ij (Abstand zwischen Partikel i und j)
                position_i = np.array(fluid_positions[i])
                position_j = np.array(fluid_positions[j_index])
                r_ij = np.linalg.norm(position_i - position_j)
                
                # Berechnung der Gewichtung
                weight_x, weight_y = grad_w_ij
                grad_w_ij_val = np.array([weight_x, weight_y])
                
                # Berechnung der Differenz der Positionen
                position_diff = position_i - position_j
                
                # Berechnung des Skalarprodukts
                dot_product = position_diff[0] * grad_w_ij_val[0] + position_diff[1] * grad_w_ij_val[1]
                
                # Berechnung von term2
                numerator_term2 = 8 * mass_per_particle * p_j * dot_product
                denominator_term2 = (rho_i + rho_j) ** 2 * (r_ij ** 2 + eta ** 2)
                term2 += numerator_term2 / denominator_term2
                
                # Berechnung von term3
                numerator_term3 = 8 * mass_per_particle * dot_product
                denominator_term3 = (rho_i + rho_j) ** 2 * (r_ij ** 2 + eta ** 2)
                term3 += numerator_term3 / denominator_term3

        print(f"Partikel {i}: term1 = {term1}, term2 = {term2}, term3 = {term3}")
        new_pressure[i] = (term1 + term2) / term3

    return new_pressure

def calculate_corrective_velocities(delta_t, mass_per_particle, current_pressure, current_rho, grad_w, fluid_positions):
    num_particles = len(current_pressure)
    fluid_and_mirror_corrective_velocities = [(0.0, 0.0)] * num_particles
    
    for i in range(num_particles):
        rho_i = current_rho[i]
        p_i = current_pressure[i]
        
        velocity_sum = np.array([0.0, 0.0])
        
        for (i_index, j_index, grad_w_ij) in grad_w:
            if i_index == i and j_index != i:
                rho_j = current_rho[j_index]
                p_j = current_pressure[j_index]
                
                weight_x, weight_y = grad_w_ij
                grad_w_ij_val = np.array([weight_x, weight_y])
                
                pressure_term = (p_j / rho_j ** 2) + (p_i / rho_i ** 2)
                velocity_sum += mass_per_particle * pressure_term * grad_w_ij_val
        
        corrective_velocity = -delta_t * velocity_sum
        fluid_and_mirror_corrective_velocities[i] = (corrective_velocity[0], corrective_velocity[1])
    
    num_fluid_particles = len(fluid_positions)
    fluid_corrective_velocities = fluid_and_mirror_corrective_velocities[:num_fluid_particles]

    return fluid_corrective_velocities


def final_velocities_and_positions(fluid_positions, fluid_velocities, last_fluid_positions, last_fluid_velocities, fluid_corrective_velocities, delta_t, box_height, box_length, spacing):
    num_particles = len(fluid_positions)
    
    # Berechnung der neuen Geschwindigkeiten
    for i in range(num_particles):
        u_star, v_star = fluid_velocities[i]
        u_hat, v_hat = fluid_corrective_velocities[i]
        new_u = u_star + u_hat
        new_v = v_star + v_hat
        fluid_velocities[i] = (new_u, new_v)
    
    # Berechnung der neuen Positionen
    for i in range(num_particles):
        x_last, y_last = last_fluid_positions[i]
        u_t, v_t = fluid_velocities[i]
        u_last, v_last = last_fluid_velocities[i]
        
        new_x = x_last + delta_t / 2 * (u_t + u_last)
        new_y = y_last + delta_t / 2 * (v_t + v_last)
        fluid_positions[i] = (new_x, new_y)
    
    # Überprüfung der Partikelpositionen und -geschwindigkeiten
    for i in range(num_particles):
        x, y = fluid_positions[i]
        u, v = fluid_velocities[i]

        border = spacing
        # Überprüfung der x-Koordinate
        if x < border:
            x = border
            u = 0
        elif x > box_length-border:
            x = box_length-border
            u = 0
        
        # Überprüfung der y-Koordinate
        if y < border:
            y = border
            v = 0
        elif y > box_height-border:
            y = box_height-border
            v = 0
        
        # Aktualisierung der Positionen und Geschwindigkeiten
        fluid_positions[i] = (x, y)
        fluid_velocities[i] = (u, v)
    
    return fluid_velocities, fluid_positions

def run_simulation(inlet_points, gravity, initial_rho, nu, mass_per_particle, num_time_steps, spacing, h, eta, cfl, delta_t, box_length, box_height):
    # Initialisieren der Simulation
    fluid_positions, fluid_velocities = initialize_fluid(inlet_points)
    current_rho = [initial_rho for _ in inlet_points]  # Initiale Dichten für jedes Partikel
    current_pressure = [0 for _ in inlet_points]  # Initiale Drücke für jedes Partikel
    num_particles = len(fluid_positions)

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

        last_fluid_positions = fluid_positions
        last_fluid_velocities = fluid_velocities

        neighbors = find_neighbors(fluid_positions, h)

        grad_w = compute_grad_w(fluid_positions, neighbors, h)
        
        viscosity_acceleration = calculate_viscosity_acceleration(fluid_positions, fluid_velocities, neighbors, grad_w, nu, eta, mass_per_particle, current_rho)
            
        fluid_positions, fluid_velocities = first_step_velocities(fluid_positions, fluid_velocities, delta_t, gravity, viscosity_acceleration)

        neighbors = find_neighbors(fluid_positions, h)

        grad_w = compute_grad_w(fluid_positions, neighbors, h)

        new_rho = calculate_density(current_rho, fluid_velocities, neighbors, grad_w, delta_t, mass_per_particle, initial_rho)
        current_rho = new_rho

        new_pressure = calculate_pressure(initial_rho, current_rho, delta_t, mass_per_particle, current_pressure, fluid_positions, eta, grad_w)
        current_pressure = new_pressure

        fluid_corrective_velocities = calculate_corrective_velocities(delta_t, mass_per_particle, current_pressure, current_rho, grad_w, fluid_positions)

        fluid_velocities, fluid_positions = final_velocities_and_positions(fluid_positions, fluid_velocities, last_fluid_positions, last_fluid_velocities, fluid_corrective_velocities, delta_t, box_height, box_length, spacing)

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