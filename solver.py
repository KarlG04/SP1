import numpy as np
import time
from scipy.spatial import KDTree
import math

def initialize_fluid(inlet_points):
    # Konvertiere inlet_points in eine Liste von Tupeln
    fluid_positions = [tuple(point) for point in inlet_points]
    fluid_velocities = [(0.0, 0.0) for _ in fluid_positions]  # Initialize velocities to (0, 0) for each point
    
    return fluid_positions, fluid_velocities

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

def compute_w(fluid_positions, neighbors, h):
    alpha_d = 10 / (7 * np.pi * h**2)
    
    def W(q):
        if 0 <= q <= 1:
            return alpha_d * (1 - 3/2 * q**2 + 3/4 * q**3)
        elif 1 < q <= 2:
            return alpha_d * (1/4 * (2 - q)**3)
        else:
            return 0
    
    w = []
    for i, position_i in enumerate(fluid_positions):
        for j in neighbors[i]:
            position_j = fluid_positions[j]
            r_ij = np.array(position_i) - np.array(position_j)
            distance = np.linalg.norm(r_ij)
            q = distance / h
            W_ij = W(q)
            w.append((i, j, W_ij))
    
    return w

def calculate_density(fluid_velocities, neighbors, w, mass_per_particle):
    num_particles = len(fluid_velocities)
    new_density = [0.0 for _ in range(num_particles)]
    for i in range(num_particles):
        sum_term = 0
        for j in neighbors[i]:
            W_ij = next((weight for (pi, pj, weight) in w if pi == i and pj == j), 0)
            sum_term += mass_per_particle * W_ij
        new_density[i] = sum_term
    return new_density

def calculate_densityy(current_density, fluid_velocities, neighbors, grad_w, delta_t, mass_per_particle, initial_density):
    num_particles = len(fluid_velocities)
    new_density = [0.0 for _ in range(num_particles)]
    for i in range(num_particles):
        sum_term = 0
        for j in neighbors[i]:
            grad_W_ij_x, grad_W_ij_y = next((grad for (pi, pj, grad) in grad_w if pi == i and pj == j), (0, 0))
            u_diff_x = fluid_velocities[i][0] - fluid_velocities[j][0]
            u_diff_y = fluid_velocities[i][1] - fluid_velocities[j][1]
            sum_term += mass_per_particle * (u_diff_x * grad_W_ij_x + u_diff_y * grad_W_ij_y) / current_density[j]
        new_density[i] = current_density[i] + delta_t * 0.5 * (initial_density + sum_term)
    return new_density

def calculate_pressure(density, initial_density, gamma, c_0):
    num_particles = len(density)
    pressure = [0.0 for _ in range(num_particles)]
    for i in range(num_particles):
        pressure[i] = ((initial_density * c_0**2) / gamma) * (((density[i] / initial_density)**gamma) - 1)
    return pressure

def calculate_pressure_acceleration(fluid_positions, density, pressure, neighbors, grad_w, mass_per_particle):
    num_particles = len(fluid_positions)
    pressure_acceleration = [(0.0, 0.0) for _ in range(num_particles)]
    for i in range(num_particles):
        acc_pressure_x = 0
        acc_pressure_y = 0
        for j in neighbors[i]:
            grad_W_ij_x, grad_W_ij_y = next((grad for (pi, pj, grad) in grad_w if pi == i and pj == j), (0, 0))
            P_i = pressure[i]
            P_j = pressure[j]
            rho_i = density[i]
            rho_j = density[j]
            acc_pressure_x += mass_per_particle * (P_i / rho_i**2 + P_j / rho_j**2) * grad_W_ij_x
            acc_pressure_y += mass_per_particle * (P_i / rho_i**2 + P_j / rho_j**2) * grad_W_ij_y
        pressure_acceleration[i] = (mass_per_particle * acc_pressure_x, mass_per_particle * acc_pressure_y)
        #print(f"Particle {i}: acc (Fx: {acc_pressure_x}, Fy: {acc_pressure_y})")

    return pressure_acceleration

def calculate_viscosity_acceleration(fluid_positions, fluid_velocities, neighbors, grad_w, nu, eta, mass_per_particle, current_density):
    num_particles = len(fluid_positions)
    viscosity_acceleration = [(0.0, 0.0) for _ in range(num_particles)]
    #for i in range(num_particles):
    #    acc_viscosity_x = 0
    #    acc_viscosity_y = 0
    #    for j in neighbors[i]:
    #        grad_W_ij_x, grad_W_ij_y = next((grad for (pi, pj, grad) in grad_w if pi == i and pj == j), (0, 0))
    #        u_diff_x = fluid_velocities[i][0] - fluid_velocities[j][0]
    #        u_diff_y = fluid_velocities[i][1] - fluid_velocities[j][1]
    #        r_ij = np.array(fluid_positions[i]) - np.array(fluid_positions[j])
    #        distance = np.linalg.norm(r_ij)
    #        acc_viscosity_x += mass_per_particle * 8 * (nu + nu) * u_diff_x / ((distance**2 + eta**2) * (current_density[i] + current_density[j])) * grad_W_ij_x
    #        acc_viscosity_y += mass_per_particle * 8 * (nu + nu) * u_diff_y / ((distance**2 + eta**2) * (current_density[i] + current_density[j])) * grad_W_ij_y
    #    viscosity_acceleration[i] = (acc_viscosity_x, acc_viscosity_y)
    return viscosity_acceleration

def calculate_boundary_force(fluid_positions, fluid_velocities, box_height, box_length, n_1, n_2, r_0, boundary_factor):
    # Initialize boundary forces list with zeros
    boundary_force = [(0.0, 0.0) for _ in range(len(fluid_positions))]

    for i in range(len(fluid_positions)):
        force_x, force_y = 0.0, 0.0
        xi = np.array(fluid_positions[i])
        vi = np.array(fluid_velocities[i])
        
        # Check boundaries in x direction
        if xi[0] < 0 + r_0 and vi[0] < 0:
            distance = xi[0]
            term1 = (r_0 / distance) ** n_1
            term2 = (r_0 / distance) ** n_2
            force_magnitude = boundary_factor * (term1 - term2)
            force_x += force_magnitude * abs(vi[0])  # Multiplied by the absolute value of x velocity component
                
        if xi[0] > box_length - r_0 and vi[0] > 0:
            distance = box_length - xi[0]
            term1 = (r_0 / distance) ** n_1
            term2 = (r_0 / distance) ** n_2
            force_magnitude = boundary_factor * (term1 - term2)
            force_x -= force_magnitude * abs(vi[0])  # Multiplied by the absolute value of x velocity component
        
        # Check boundaries in y direction
        if xi[1] < 0 + r_0 and vi[1] < 0:
            distance = xi[1]
            term1 = (r_0 / distance) ** n_1
            term2 = (r_0 / distance) ** n_2
            force_magnitude = boundary_factor * (term1 - term2)
            force_y += force_magnitude * abs(vi[1])  # Multiplied by the absolute value of y velocity component
                
        if xi[1] > box_height - r_0 and vi[1] > 0:
            distance = box_height - xi[1]
            term1 = (r_0 / distance) ** n_1
            term2 = (r_0 / distance) ** n_2
            force_magnitude = boundary_factor * (term1 - term2)
            force_y -= force_magnitude * abs(vi[1])  # Multiplied by the absolute value of y velocity component

        boundary_force[i] = (force_x, force_y)
        # Print the boundary force for each particle
        #print(f"Particle {i}: Boundary Force = (Fx: {force_x}, Fy: {force_y})")
        
    return boundary_force

def integrate_substantial_acceleration(fluid_positions, fluid_velocities, pressure_acceleration, viscosity_acceleration, boundary_force, delta_t, gravity):
    num_particles = len(fluid_positions)
    for i in range(num_particles):
        total_acc_x = pressure_acceleration[i][0] + viscosity_acceleration[i][0] + gravity[0] + boundary_force[i][0]
        total_acc_y = pressure_acceleration[i][1] + viscosity_acceleration[i][1] + gravity[1] + boundary_force[i][1]
        new_fluid_velocities_x = fluid_velocities[i][0] + delta_t * total_acc_x
        new_fluid_velocities_y = fluid_velocities[i][1] + delta_t * total_acc_y

        new_fluid_positions_x = fluid_positions[i][0] + delta_t * new_fluid_velocities_x
        new_fluid_positions_y = fluid_positions[i][1] + delta_t * new_fluid_velocities_y

        # Update fluid_positions as tuples
        fluid_positions[i] = (new_fluid_positions_x, new_fluid_positions_y)



        # Update fluid_velocities as tuples
        fluid_velocities[i] = (new_fluid_velocities_x, new_fluid_velocities_y)
        # Print the boundary force for each particle
        #print(f"Particle {i}: acc (Fx: {new_fluid_positions_x}, Fy: {new_fluid_positions_y})")
    return fluid_velocities, fluid_positions


def run_simulation(inlet_points, gravity, initial_density, nu, mass_per_particle, num_time_steps, spacing, h, eta, delta_t, box_length, box_height, c_0, gamma, n_1, n_2, r_0, boundary_factor):
    # Initialisieren der Simulation
    fluid_positions, fluid_velocities = initialize_fluid(inlet_points)
    current_density = [initial_density for _ in inlet_points]  # Initiale Dichten für jedes Partikel
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

        w = compute_w(fluid_positions, neighbors, h)

        grad_w = compute_grad_w(fluid_positions, neighbors, h)
        
        density = calculate_density(fluid_velocities, neighbors, w, mass_per_particle)

        pressure = calculate_pressure(density, initial_density, gamma, c_0)

        pressure_acceleration = calculate_pressure_acceleration(fluid_positions, current_density, pressure, neighbors, grad_w, mass_per_particle)

        viscosity_acceleration = calculate_viscosity_acceleration(fluid_positions, fluid_velocities, neighbors, grad_w, nu, eta, mass_per_particle, current_density)

        boundary_force = calculate_boundary_force(fluid_positions, fluid_velocities, box_height, box_length, n_1, n_2, r_0, boundary_factor)

        fluid_velocities, fluid_positions = integrate_substantial_acceleration(fluid_positions, fluid_velocities, pressure_acceleration, boundary_force, viscosity_acceleration, delta_t, gravity)

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