import numpy as np
import time
from scipy.spatial import KDTree

def initialize_simulation(inlet_points, initial_velocity):
    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []
    for point in inlet_points:
        positions_x.append(point[0])
        positions_y.append(point[1])
        velocities_x.append(initial_velocity[0])
        velocities_y.append(initial_velocity[1])

    positions = list(zip(positions_x, positions_y))
    velocities = list(zip(velocities_x, velocities_y))
    
    return positions, velocities

def calculate_time_step(velocities, cfl, spacing, delta_t_diffusion):
    velocities_x = [vel[0] for vel in velocities]
    velocities_y = [vel[1] for vel in velocities]
    resulting_velocities = np.sqrt(np.square(velocities_x) + np.square(velocities_y))
    v_max = np.max(resulting_velocities)
    if v_max != 0:
        delta_t_courant = cfl * spacing / v_max
    else:
        delta_t_courant = delta_t_diffusion

    return min(delta_t_courant, delta_t_diffusion)

def add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity):
    for point in inlet_points:
        positions_x.append(point[0])
        positions_y.append(point[1])
        velocities_x.append(initial_velocity[0])
        velocities_y.append(initial_velocity[1])

def calculate_particle_add_interval(initial_velocity, spacing):
    initial_velocity_magnitude = np.linalg.norm(initial_velocity)
    return spacing / initial_velocity_magnitude

def update_velocities_step_1(all_velocities, gravity, delta_t):
    all_velocities = [(vx + gravity[0] * delta_t, vy + gravity[1] * delta_t) for vx, vy in all_velocities]
    return all_velocities

def update_positions_step_1(all_positions, all_velocities, delta_t):
    # Extrahiere x- und y-Komponenten
    all_positions_x = [pos[0] for pos in all_positions]
    all_positions_y = [pos[1] for pos in all_positions]
    all_velocities_x = [vel[0] for vel in all_velocities]
    all_velocities_y = [vel[1] for vel in all_velocities]
    
    # Aktualisiere die Positionen
    all_positions_x = [px + vx * delta_t for px, vx in zip(all_positions_x, all_velocities_x)]
    all_positions_y = [py + vy * delta_t for py, vy in zip(all_positions_y, all_velocities_y)]
    
    # Kombiniere die aktualisierten x- und y-Komponenten
    all_positions = list(zip(all_positions_x, all_positions_y))
    
    return all_positions


def merge_positions(positions, boundary_positions):
    # Extrahiere x und y Positionen aus den Fluid- und Boundary-Points
    fluid_positions_x = [pos[0] for pos in positions]
    fluid_positions_y = [pos[1] for pos in positions]
    
    # Extrahiere x und y Positionen aus dem Array boundary_positions
    boundary_positions_x = boundary_positions[:, 0]
    boundary_positions_y = boundary_positions[:, 1]
    
    # Füge Boundary-Positionen an den Anfang der Fluid-Positionen hinzu
    all_positions_x = np.concatenate((boundary_positions_x, fluid_positions_x))
    all_positions_y = np.concatenate((boundary_positions_y, fluid_positions_y))
    
    # Erstelle die Liste von Tupeln
    all_positions = list(zip(all_positions_x, all_positions_y))

    return all_positions

def merge_velocities(velocities, boundary_points):
    # Anzahl der Boundary-Points
    num_boundary_points = len(boundary_points)

    # Extrahiere x- und y-Komponenten der Geschwindigkeiten
    velocities_x = [vel[0] for vel in velocities]
    velocities_y = [vel[1] for vel in velocities]

    # Füge Nullen am Anfang der Fluid-Geschwindigkeiten hinzu
    all_velocities_x = np.concatenate((np.zeros(num_boundary_points), velocities_x))
    all_velocities_y = np.concatenate((np.zeros(num_boundary_points), velocities_y))

    # Erstelle die Liste von Tupeln
    all_velocities = list(zip(all_velocities_x, all_velocities_y))

    return all_velocities


def find_neighbors(all_positions, h):
    # Anzahl der Partikel bestimmen
    num_particles = len(all_positions)

    # Liste mit leeren Listen der Länge num_particles initialisieren
    neighbors = [[] for _ in range(num_particles)]

    # KDTree erstellen
    tree = KDTree(all_positions)

    # Nachbarn innerhalb des Radius h finden und in der Liste speichern
    for i, position in enumerate(all_positions):
        # Finde alle Nachbarn innerhalb des Radius h
        neighbor_indices = tree.query_ball_point(position, h)
        # Entferne den Partikel selbst aus der Liste der Nachbarn
        neighbors[i] = [index for index in neighbor_indices if index != i]
      
    return neighbors


def kernel_gradient(all_positions, h, neighbors):
    num_particles = len(all_positions)
    grad_w = [[] for _ in range(num_particles)]  # Initialisiere eine Liste von Listen für Gradienten

    alpha_D = 10 / (7 * np.pi * h**2)

    # Extrahiere x- und y-Komponenten
    all_positions_x = [pos[0] for pos in all_positions]
    all_positions_y = [pos[1] for pos in all_positions]

    for i in range(num_particles):
        for j in neighbors[i]:  # Nur über die Nachbarn iterieren
            if i != j:
                dx = abs(all_positions_x[j] - all_positions_x[i])
                dy = abs(all_positions_y[j] - all_positions_y[i])
                r = np.sqrt(dx**2 + dy**2)
                
                #print(f"Partikel i: {i}, Partikel j: {j}, dx: {dx}, dy: {dy}, r: {r}")
                
                if r == 0 or r > h:
                    #print(f"Partikel i: {i}, Partikel j: {j} - r ist Null oder größer als h, wird übersprungen")
                    continue  # Vermeide Division durch Null und entfernte Nachbarn

                s = r / h
                #print(f"Partikel i: {i}, Partikel j: {j}, s: {s}")
                
                if s <= 1:
                    factor = alpha_D * abs(1 - 3/2 * s**2 + 3/4 * s**3)
                elif s <= 2:
                    factor = alpha_D * (1/4 * (2 - s)**2)
                else:
                    #print(f"Partikel i: {i}, Partikel j: {j} - s ist größer oder gleich 2, wird übersprungen")
                    continue  # Dieses Element bleibt Null und wird nicht verwendet.

                grad_w_ij = factor * np.array([dx, dy])
                grad_w[i].append((j, grad_w_ij))
                
                #print(f"Partikel i: {i}, Partikel j: {j}, factor: {factor:.2f}, r: {r:.2f}, grad_w_ij: {[round(g, 2) for g in grad_w_ij]}")

    return grad_w

def calculate_tau(all_positions, all_velocities, mu, neighbors):
    num_particles = len(all_positions)
    tau = []

    # Extrahiere x- und y-Komponenten
    all_positions_x = [pos[0] for pos in all_positions]
    all_positions_y = [pos[1] for pos in all_positions]
    all_velocities_x = [vel[0] for vel in all_velocities]
    all_velocities_y = [vel[1] for vel in all_velocities]

    for i in range(num_particles):
        tau_ij = [[0, 0], [0, 0]]
        for j in neighbors[i]:  # neighbors[i] enthält die Indizes der Nachbarn von Partikel i
            if i != j:  # Vermeide Selbstbezug
                dx = all_positions_x[j] - all_positions_x[i]
                dy = all_positions_y[j] - all_positions_y[i]

                du_dx = (all_velocities_x[j] - all_velocities_x[i]) / dx if dx != 0 else 0
                du_dy = (all_velocities_x[j] - all_velocities_x[i]) / dy if dy != 0 else 0
                dv_dx = (all_velocities_y[j] - all_velocities_y[i]) / dx if dx != 0 else 0
                dv_dy = (all_velocities_y[j] - all_velocities_y[i]) / dy if dy != 0 else 0

                # Gradientenmatrix und Transponierte
                grad_v = [[du_dx, du_dy], [dv_dx, dv_dy]]
                grad_v_T = [[du_dx, dv_dx], [du_dy, dv_dy]]

                # Berechnung der Scherspannung
                tau_ij[0][0] += mu * (grad_v[0][0] + grad_v_T[0][0])
                tau_ij[0][1] += mu * (grad_v[0][1] + grad_v_T[0][1])
                tau_ij[1][0] += mu * (grad_v[1][0] + grad_v_T[1][0])
                tau_ij[1][1] += mu * (grad_v[1][1] + grad_v_T[1][1])

        tau.append((tau_ij[0][0], tau_ij[1][1]))  # Speichere die x- und y-Komponenten
        #for i, (tau_x, tau_y) in enumerate(tau):
        #    print(f"Partikel {i+1}: tau_x = {tau_x:.2f} | tau_y = {tau_y:.2f}")

    return tau

def calculate_stress_tensor(mass_per_particle, tau, rho, grad_w, neighbors):
    num_particles = len(tau)
    stress_tensor = [(0, 0)] * num_particles

    # Extract components
    tau_x = [item[0] for item in tau]
    tau_y = [item[1] for item in tau]

    for i in range(num_particles):
        sum_x = 0
        sum_y = 0
        for j in neighbors[i]:
            if i != j:
                # Compute the term for the summation
                tau_term_x = (tau_x[j] / rho**2) + (tau_x[i] / rho**2)
                tau_term_y = (tau_y[j] / rho**2) + (tau_y[i] / rho**2)

                # Ensure the index j is valid within grad_w[i]
                grad_w_ij = next((grad[1] for grad in grad_w[i] if grad[0] == j), [0, 0])
                
                sum_x += mass_per_particle * tau_term_x * grad_w_ij[0]
                sum_y += mass_per_particle * tau_term_y * grad_w_ij[1]
        
        stress_tensor[i] = (sum_x, sum_y)

    #for i, (S_x, S_y) in enumerate(stress_tensor):
    #    print(f"Partikel {i+1}: S_x = {S_x:.2f} | S_y = {S_y:.2f}")

    return stress_tensor

def update_velocities_step_2(all_velocities, stress_tensor, delta_t):
    # Extrahiere x- und y-Komponenten
    all_velocities_x = [vel[0] for vel in all_velocities]
    all_velocities_y = [vel[1] for vel in all_velocities]
    stress_tensor_x = [item[0] for item in stress_tensor]
    stress_tensor_y = [item[1] for item in stress_tensor]

    all_velocities_x = [vx + stress_tensor_x[i] * delta_t for i, vx in enumerate(all_velocities_x)]
    all_velocities_y = [vy + stress_tensor_y[i] * delta_t for i, vy in enumerate(all_velocities_y)]

    # Erstelle die Liste von Tupeln
    all_velocities = list(zip(all_velocities_x, all_velocities_y))

    return all_velocities

def update_positions_step_2(all_positions, all_velocities, delta_t):
    # Extrahiere x- und y-Komponenten
    all_positions_x = [pos[0] for pos in all_positions]
    all_positions_y = [pos[1] for pos in all_positions]
    all_velocities_x = [vel[0] for vel in all_velocities]
    all_velocities_y = [vel[1] for vel in all_velocities]

    all_positions_x = [px + vx * delta_t for px, vx in zip(all_positions_x, all_velocities_x)]  
    all_positions_y = [py + vy * delta_t for py, vy in zip(all_positions_y, all_velocities_y)]

    # Erstelle die Liste von Tupeln
    all_positions = list(zip(all_positions_x, all_positions_y))

    return all_positions


def calculate_temporary_density(rho, mass_per_particle, all_velocities, grad_w, neighbors):
    num_particles = len(all_velocities)
    rho_temp = [rho] * num_particles  # Initialisiere die temporäre Dichte mit rho

    # Extrahiere x- und y-Komponenten
    all_velocities_x = [vel[0] for vel in all_velocities]
    all_velocities_y = [vel[1] for vel in all_velocities]

    for i in range(num_particles):
        density_sum = 0.0
        for j in neighbors[i]:
            if i != j:
                u_diff = all_velocities_x[i] - all_velocities_x[j]
                v_diff = all_velocities_y[i] - all_velocities_y[j]
                #print(f"Partikel i: {i}, Partikel j: {j}, u_diff: {u_diff}, v_diff: {v_diff}")

                # Ensure the index j is valid within grad_w[i]
                grad_w_ij = next((grad[1] for grad in grad_w[i] if grad[0] == j), [0, 0])
                
                density_sum += mass_per_particle * (u_diff * grad_w_ij[0] + v_diff * grad_w_ij[1])
                #print(f"Partikel i: {i}, Partikel j: {j}, Dichtebeitrag: {density_sum}")

        rho_temp[i] += density_sum
        #print(f"Partikel i: {i}, rho_temp: {rho_temp[i]}")  # Ausgabe der Partikelnummern und der temporären Dichte
    return rho_temp

def calculate_pressure(all_positions, rho, rho_temp, delta_t, neighbors, grad_w, pressure, mass_per_particle, eta, initial_pressure):
    num_particles = len(all_positions)
    last_num_particles = len(pressure)

    # Ensure pressure is a list
    pressure = list(pressure)

    # Extract x- and y-components
    all_positions_x = [pos[0] for pos in all_positions]
    all_positions_y = [pos[1] for pos in all_positions]

    if num_particles != last_num_particles:  # Extend the pressure list for new particles
        new_particles = num_particles - last_num_particles
        pressure.extend([(0.0, 0.0)] * new_particles)  # Initialize new particles with initial pressure

    # Extract x- and y-components of pressure from the last time step
    pressure_x = [item[0] for item in pressure]
    pressure_y = [item[1] for item in pressure]

    new_pressure = []  # To store updated pressure values

    for i in range(num_particles):
        if rho - rho_temp[i] == 0:
            new_pressure_x_i = 0
            new_pressure_y_i = 0
        else:
            term1 = (rho - rho_temp[i]) / delta_t**2
            sum_term2_x = 0.0
            sum_term2_y = 0.0
            sum_term3_x = 0.0
            sum_term3_y = 0.0

            for j in neighbors[i]:
                if i != j:
                    mj = mass_per_particle

                    dx = all_positions_x[i] - all_positions_x[j]
                    dy = all_positions_y[i] - all_positions_y[j]
                    
                    # Avoid division by zero
                    if dx != 0 or dy != 0:
                        r_ij_x = (dx)**-2 if dx != 0 else 0
                        r_ij_y = (dy)**-2 if dy != 0 else 0

                    # Ensure the index j is valid within grad_w[i]
                    grad_w_ij = next((grad[1] for grad in grad_w[i] if grad[0] == j), [0, 0])

                    grad_w_ij_x = r_ij_x * grad_w_ij[0]
                    grad_w_ij_y = r_ij_y * grad_w_ij[1]

                    sum_term2_x += (8 * mj * pressure_x[j] * grad_w_ij_x) / ((rho_temp[i] + rho_temp[j]) * (r_ij_x ** 2 + eta**2))
                    sum_term2_y += (8 * mj * pressure_y[j] * grad_w_ij_y) / ((rho_temp[i] + rho_temp[j]) * (r_ij_y ** 2 + eta**2))

                    sum_term3_x += (8 * mj * grad_w_ij_x) / ((rho_temp[i] + rho_temp[j]) * (r_ij_x ** 2 + eta**2))
                    sum_term3_y += (8 * mj * grad_w_ij_y) / ((rho_temp[i] + rho_temp[j]) * (r_ij_y ** 2 + eta**2))

            new_pressure_x_i = (term1 + sum_term2_x) / sum_term3_x
            new_pressure_y_i = (term1 + sum_term2_y) / sum_term3_y
            #print(f"Partikel {i+1}: term1: {term1:.2f} | sum_term2_x: {sum_term2_x:.10f}| sum_term3_x: {sum_term3_x:.10f}")
        new_pressure.append((new_pressure_x_i, new_pressure_y_i))

    for i, (pressure_x, pressure_y) in enumerate(new_pressure):
        print(f"Partikel {i+1}: pressure_x = {pressure_x:.6f} | pressure_y = {pressure_y:.6f}")

    return new_pressure


def calculate_corrective_velocities(mass_per_particle, pressure, rho_temp, grad_w, delta_t):
    num_particles = len(rho_temp)
    corrective_velocities_x = [0.0] * num_particles
    corrective_velocities_y = [0.0] * num_particles

    for i in range(num_particles):
        sum_x = 0.0
        sum_y = 0.0
        for (j, grad_w_ij) in grad_w[i]:
            if i != j:
                Pj = pressure[j]
                Pi = pressure[i]
                rho_temp_j = rho_temp[j]
                rho_temp_i = rho_temp[i]

                correction_term = mass_per_particle * ((Pj / (rho_temp_j ** 2)) + (Pi / (rho_temp_i ** 2)))
                
                sum_x += correction_term * grad_w_ij[0]
                sum_y += correction_term * grad_w_ij[1]

        corrective_velocities_x[i] = -delta_t * sum_x
        corrective_velocities_y[i] = -delta_t * sum_y

    return corrective_velocities_x, corrective_velocities_y

def update_velocities_step_3(all_velocities_x, all_velocities_y, corrective_velocities_x, corrective_velocities_y):
    updated_velocities_x = [vx + cvx for vx, cvx in zip(all_velocities_x, corrective_velocities_x)]
    updated_velocities_y = [vy + cvy for vy, cvy in zip(all_velocities_y, corrective_velocities_y)]

    return updated_velocities_x, updated_velocities_y

def update_positions_step_3(all_positions_x_last_iteration, all_positions_y_last_iteration, all_velocities_x_last_iteration, all_velocities_y_last_iteration, all_velocities_x, all_velocities_y, delta_t):
    all_positions_x = [px_last + (delta_t / 2) * (vx + vx_last)
                       for px_last, vx, vx_last in zip(all_positions_x_last_iteration, all_velocities_x, all_velocities_x_last_iteration)]
    all_positions_y = [
        py_last + (delta_t / 2) * (vy + vy_last)for py_last, vy, vy_last in zip(all_positions_y_last_iteration, all_velocities_y, all_velocities_y_last_iteration)]
    
    return all_positions_x, all_positions_y

def separate_positions(all_positions, boundary_points):
    # Anzahl der Boundary-Points
    num_boundary_points = len(boundary_points)

    # Entferne die Boundary-Points von den Positionen
    positions = all_positions[num_boundary_points:]

    return positions

def separate_velocities(all_velocities, boundary_points):
    # Anzahl der Boundary-Points
    num_boundary_points = len(boundary_points)

    # Entferne die Boundary-Points von den Geschwindigkeiten
    velocities = all_velocities[num_boundary_points:]

    return velocities


def run_simulation(inlet_points, initial_velocity, gravity, cfl, rho, mu, mass_per_particle, num_time_steps, spacing, h, boundary_points, eta, initial_pressure, delta_t_diffusion):
    # Initialisieren der Simulation
    positions, velocities = initialize_simulation(inlet_points, initial_velocity)

    # Initialisieren des Drucks für jeden Partikel
    num_particles = len(inlet_points) + len(boundary_points)
    pressure = [(0.0, 0.0) for _ in range(num_particles)]


    # Gesuchte Werte für jeden Zeitschritt initialisieren
    delta_t_collected = []  # Liste zum Speichern der delta_t Werte
    positions_collected = [] # Liste zum Speichern der Positionen
    velocities_collected = [] # Liste zum Speichern der Geschwindigkeitskomponenten
    tau_collected = [] # Liste zum Speichern der tau Werte
    pressure_collected = [] # Liste zum Speichern der Druck Werte

    # Intervall für das Hinzufügen neuer Partikel berechnen
    #particle_add_interval = calculate_particle_add_interval(initial_velocity, spacing)  
    #time_since_last_addition = 0  # Initialisieren des Zeitzählers für Partikelzugabe
    
    iteration_start_time = time.perf_counter()  # Startzeit für Iterationen messen
    for t in range(num_time_steps):
        print(f"Running iteration {t+1}/{num_time_steps} | ", end="")  # Ausgabe der aktuellen Iterationsnummer
        iteration_step_start_time = time.perf_counter() # Startzeit für jeden einzelnen Iterationsschritt
        # Zeitschritt basierend auf den Geschwindigkeiten berechnen
        delta_t = calculate_time_step(velocities, cfl, spacing, delta_t_diffusion)
        #time_since_last_addition += delta_t  # Zeit seit der letzten Zugabe aktualisieren

        # Überprüfen, ob es Zeit ist, neue Partikel hinzuzufügen
        #if time_since_last_addition >= particle_add_interval:
            #add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity)
            #time_since_last_addition = 0  # Zeitgeber nach dem Hinzufügen von Partikeln zurücksetzen

        # Positionen und Geschwindigkeitendes Fluid und der Boundary zusammenführen
        all_positions = merge_positions(positions, boundary_points)
        all_velocities = merge_velocities(velocities, boundary_points)

        # Kopien von Positionen und Geschwindigkeiten der letzten Iteration erstellen für update_positions_step_3
        all_positions_last_iteration = all_positions.copy()
        all_velocities_last_iteration = all_velocities.copy()

        # Schritt 1 (Dreischrittalgorithmus)
        # Aktualisieren der Geschwindigkeiten aufgrund von Gravitation
        all_velocities = update_velocities_step_1(all_velocities, gravity, delta_t)
        
        # Aktualisieren der Positionen aufgrund von Gravitation
        #all_positions = update_positions_step_1(all_positions, all_velocities, delta_t)
        
        # Schritt 2 (Dreischrittalgorithmus)
        # Nachbarn finden
        neighbors = find_neighbors(all_positions, h)
        
        # Kernel-Gradient berechnen
        grad_w = kernel_gradient(all_positions, h, neighbors)
        
        # Tau berechnen
        tau = calculate_tau(all_positions, all_velocities, mu, neighbors)

        # Komponenten der Divergenz des Spannungstensors S_x und S_y berechnen
        stress_tensor = calculate_stress_tensor(mass_per_particle, tau, rho, grad_w, neighbors)

        # Geschwindigkeiten mittels S_x und S_y aktualisieren
        all_velocities = update_velocities_step_2(all_velocities, stress_tensor, delta_t)

        # Positionen mittels S_x und S_y aktualisieren
        all_positions = update_positions_step_2(all_positions, all_velocities, delta_t)

        # Schritt 3 (Dreischrittalgorithmus)
        # Nachbarn finden
        neighbors = find_neighbors(all_positions, h)
        
        # Kernel-Gradient berechnen
        grad_w = kernel_gradient(all_positions, h, neighbors)

        # temporären Dichte brechnen
        rho_temp = calculate_temporary_density(rho, mass_per_particle, all_velocities, grad_w, neighbors)

        # Druck berechnen
        pressure = calculate_pressure(all_positions, rho, rho_temp, delta_t, neighbors, grad_w, pressure, mass_per_particle, eta, initial_pressure)

        # Berechnung der Korrekturgeschwindigkeiten mittels pessure und rho_temp
        #corrective_velocities_x, corrective_velocities_y = calculate_corrective_velocities(mass_per_particle, pressure, rho_temp, grad_w, delta_t)

        # Berechnung der neuen Geschwindigkeiten aufgrund der Korrekturgeschwindigkeit
        #all_velocities_x, all_velocities_y = update_velocities_step_3(all_velocities_x, all_velocities_y, corrective_velocities_x, corrective_velocities_y)
            
        # Positionen aufgrund Geschwindigkeitskorrektur berechnen
        #all_positions_x, all_positions_y = update_positions_step_3(all_positions_x_last_iteration, all_positions_y_last_iteration, all_velocities_x_last_iteration, all_velocities_y_last_iteration, all_velocities_x,all_velocities_y ,delta_t)

        # Positionslisten und Geschwindigkeitslisten nach dem Entfernen der Boundary-Points wiederherstellen
        positions = separate_positions(all_positions, boundary_points)
        velocities = separate_velocities(all_velocities, boundary_points)

        #print("velocities_x_step_3:")
        #for vx in velocities_x:
        #    print(f"{vx:.2f} m/s")

        #print("\nvelocities_y_step_3:")
        #for vy in velocities_y:
        #    print(f"{vy:.2f} m/s")

        # Ergebnisse für den aktuellen Zeitschritt speichern
        delta_t_collected.append(delta_t)  # delta_t sammeln
        positions_collected.append(positions.copy()) # Positionen sammeln
        velocities_collected.append(velocities.copy()) # Geschwindigkeiten sammeln
        #tau_collected.append(tau)  # Scherspannungen sammeln
        #pressure_collected.append(pressure) # Drücke sammeln

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

    # Maximale Anzahl der Partikel zu jedem Zeitschritt bestimmen
    max_particles = max(len(px) for px in positions_collected)

    # Fluid_Points-Array mit der maximalen Anzahl von Partikeln erstellen
    Fluid_Points = np.zeros((4, max_particles, num_time_steps))

    for t in range(num_time_steps):
        num_particles = len(positions_collected[t])
        
        positions_x_collected = [pos[0] for pos in positions_collected[t]]
        positions_y_collected = [pos[1] for pos in positions_collected[t]]
        velocities_x_collected = [vel[0] for vel in velocities_collected[t]]
        velocities_y_collected = [vel[1] for vel in velocities_collected[t]]

        Fluid_Points[0, :num_particles, t] = positions_x_collected
        Fluid_Points[1, :num_particles, t] = positions_y_collected
        Fluid_Points[2, :num_particles, t] = velocities_x_collected
        Fluid_Points[3, :num_particles, t] = velocities_y_collected

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

    # gesamte Partikelzahl berechnen und ausgeben
    number_particle_additions = len(positions) / len(inlet_points)
    print(f"fluid points: {len(inlet_points)} x {number_particle_additions:.0f} = {len(positions)}")
    print(" ")

    # Partikelgröße ausgeben
    diameterµm = spacing * 1e6
    print(f"particle diameter: {diameterµm:.2f}µm")

    return Fluid_Points, delta_t_collected
