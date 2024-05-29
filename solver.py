import numpy as np
import time
from scipy.spatial import KDTree

# Fluid-Eigenschaften gegeben
rho = 1000  # Dichte des Wassers in kg/m³
diameter_particle = 5 * 1e-3 # Partikeldurchmesser in m
mu = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))
initial_pressure = 1.0 # Initialer Druck (für ersten Iterationsschritt) in N/m²

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2 # Fläche eines Partikels in m²
volume_per_particle = area_per_particle # Volumen in m³ (für 1D Tiefe)
mass_per_particle = volume_per_particle * rho # Masse eines Partikels in kg
h = 1.5 * spacing # Glättungsradius in m

# Weitere Simulationsparameter
num_time_steps = 100 # Anzahl an Berechnungsintervallen
delta_t = 0.01  # Zeitschritt in s
eta = 0.1 * h # Reulierungsparameter für den Dreischrittalgorythmus
cfl = 0.1 # Konstante damit der Zeitschritt nicht zu groß wird (gängig 0.1)
beta = 0.1  # Faktor für Diffusionsbedingung
delta_t_diffusion = (beta * rho * spacing**2)/mu
animation_interval = 1 # Faktor zur animationsgeschwindigkeit

#Anfangsbedingungen
initial_velocity = [-3.0, 0.0] # Anfangsgeschwindigkeit in m/s
gravity = [0.0, -9.81]  # Gravitationskraft in m/s² (x-Komponente, y-Komponente)

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

    # Liste mit Nullen der Länge num_particles initialisieren
    neighbors = [[] for _ in range(num_particles)]

    # Positionen zusammenführen
    positions = all_positions

    # KDTree erstellen
    tree = KDTree(positions)

    # Nachbarn innerhalb des Radius h finden und in der Liste speichern
    for i, position in enumerate(positions):
        neighbors[i] = tree.query_ball_point(position, h)

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

def calculate_tau(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, mu, neighbors):
    num_particles = len(all_positions_x)
    tau = np.zeros((num_particles, 2, 2))  # Initialisiere tau als Array von 2x2 Matrizen für jedes Partikel

    for i in range(num_particles):
        for j in neighbors[i]:  # neighbors[i] enthält die Indizes der Nachbarn von Partikel i
            if i != j:  # Vermeide Selbstbezug
                dx = all_positions_x[j] - all_positions_x[i]
                dy = all_positions_y[j] - all_positions_y[i]

                if dx != 0:
                    du_dx = (all_velocities_x[j] - all_velocities_x[i]) / dx
                    dv_dx = (all_velocities_y[j] - all_velocities_y[i]) / dx
                else:
                    du_dx = 0
                    dv_dx = 0

                if dy != 0:
                    du_dy = (all_velocities_x[j] - all_velocities_x[i]) / dy
                    dv_dy = (all_velocities_y[j] - all_velocities_y[i]) / dy
                else:
                    du_dy = 0
                    dv_dy = 0

                # Gradientenmatrix und Transponierte
                grad_v = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
                grad_v_T = grad_v.T

                # Berechnung der Scherspannung
                tau_ij = mu * (grad_v + grad_v_T)

                tau[i] += tau_ij  # Akkumuliere die Scherspannung für Partikel i
                
    return tau

def calculate_tau(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, mu, neighbors):
    num_particles = len(all_positions_x)
    tau = []

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

        tau.append(tau_ij)
        print(f"Partikel i: {i}, tau: {tau}")

    return tau

def calculate_S(mass_per_particle, tau, rho, grad_w, neighbors):
    num_particles = len(tau)
    S_x = [0] * num_particles
    S_y = [0] * num_particles

    for i in range(num_particles):
        sum_x = 0
        sum_y = 0
        for j in neighbors[i]:
            if i != j:
                # Compute the term for the summation
                tau_term_x = ((tau[j][0][0]) / rho**2) + ((tau[i][0][0]) / rho**2)
                tau_term_y = ((tau[j][1][1]) / rho**2) + ((tau[i][1][1]) / rho**2)

                # Ensure the index j is valid within grad_w[i]
                grad_w_ij_x = next((grad[1][0] for grad in grad_w[i] if grad[0] == j), 0)
                grad_w_ij_y = next((grad[1][1] for grad in grad_w[i] if grad[0] == j), 0)
                
                sum_x += mass_per_particle * tau_term_x * grad_w_ij_x
                sum_y += mass_per_particle * tau_term_y * grad_w_ij_y
        
        S_x[i] = sum_x
        S_y[i] = sum_y
       # print(f"Partikel i: {i}, S_x: {S_x[i]}, S_y: {S_y[i]}")

    return S_x, S_y


def update_velocities_step_2(all_velocities_x, all_velocities_y, S_x, S_y, delta_t):
    all_velocities_x = [vx + S_x[i] * delta_t for i, vx in enumerate(all_velocities_x)]
    all_velocities_y = [vy + S_y[i] * delta_t for i, vy in enumerate(all_velocities_y)]

    return all_velocities_x, all_velocities_y

def update_positions_step_2(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, delta_t):
    all_positions_x = [px + vx * delta_t for px, vx in zip(all_positions_x, all_velocities_x)]  
    all_positions_y = [py + vy * delta_t for py, vy in zip(all_positions_y, all_velocities_y)]

    return all_positions_x, all_positions_y



def calculate_temporary_density(rho, mass_per_particle, all_velocities_x, all_velocities_y, grad_w, neighbors):
    num_particles = len(all_velocities_x)
    rho_temp = [rho] * num_particles  # Initialisiere die temporäre Dichte mit rho

    for i in range(num_particles):
        density_sum = 0.0
        for j in neighbors[i]:
            if i != j:
                u_diff = all_velocities_x[i] - all_velocities_x[j]
                v_diff = all_velocities_y[i] - all_velocities_y[j]
                #print(f"Partikel i: {i}, Partikel j: {j}, u_diff: {u_diff}, v_diff: {v_diff}")

                grad_w_ij = next((grad for idx, grad in grad_w[i] if idx == j), np.zeros(2))
                
                density_sum += mass_per_particle * (u_diff * grad_w_ij[0] + v_diff * grad_w_ij[1])
                #print(f"Partikel i: {i}, Partikel j: {j}, Dichtebeitrag: {density_sum}")

        rho_temp[i] += density_sum
        #print(f"Partikel i: {i}, Partikel j: {j}, rho_temp_i: {rho_temp[i]}")  # Ausgabe der Partikelnummern und der temporären Dichte
    return rho_temp

def calculate_pressure(all_positions_x, all_positions_y, rho, rho_temp, delta_t, neighbors, grad_w, pressure, mass_per_particle, eta, initial_pressure):
    num_particles = len(all_positions_x)
    last_num_particles = len(pressure)

    if num_particles != last_num_particles:  # Erweiterung der Druckliste um die neuen Partikel wenn neue Partikel hinzugefügt wurden
        new_particles = num_particles - last_num_particles
        pressure.extend([initial_pressure] * new_particles)

    for i in range(num_particles):
        term1 = (rho - rho_temp[i]) / delta_t**2
        term2_x = 0.0
        term2_y = 0.0
        term3_x = 0.0
        term3_y = 0.0

        for j in neighbors[i]:
            if i != j:
                Pj = pressure[j]
                mj = mass_per_particle
                r_ij_x = (all_positions_x[i]**2 - all_positions_x[j]**2)**-2
                r_ij_y = (all_positions_y[i]**2 - all_positions_y[j]**2)**-2
                grad_W_ij = next((grad for idx, grad in grad_w[i] if idx == j), np.zeros(2))
                grad_W_ij_x = r_ij_x * grad_W_ij[0]
                grad_W_ij_y = r_ij_y * grad_W_ij[1]

                term2_x += (8 * mj * Pj * grad_W_ij_x) / ((rho_temp[i] + rho_temp[j]) ** 2 * (r_ij_x ** 2 + eta**2))
                term2_y += (8 * mj * Pj * grad_W_ij_y) / ((rho_temp[i] + rho_temp[j]) ** 2 * (r_ij_y ** 2 + eta**2))

                term3_x += (8 * mj * grad_W_ij_x) / ((rho_temp[i] + rho_temp[j]) ** 2 * (r_ij_x ** 2 + eta**2))
                term3_y += (8 * mj * grad_W_ij_y) / ((rho_temp[i] + rho_temp[j]) ** 2 * (r_ij_y ** 2 + eta**2))

        term2 = term2_x + term2_y
        term3 = term3_x + term3_y

        if term1 != 0 and term2 != 0 or term3 != 0:
            pressure[i] = (term1 + term2) / term3
        print(f"Partikel i: {i}, Partikel j: {j}, pressure: {pressure[i]}")

    return pressure


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


def run_simulation(inlet_points, initial_velocity, gravity, cfl, rho, num_time_steps, spacing, boundary_points, eta, initial_pressure):
    # Initialisieren der Simulation
    positions, velocities = initialize_simulation(inlet_points, initial_velocity)

    # Initialisieren des Drucks für jeden Partikel
    pressure = [initial_pressure] * (len(inlet_points) + len(boundary_points))

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
        all_positions = update_positions_step_1(all_positions, all_velocities, delta_t)
        
        # Schritt 2 (Dreischrittalgorithmus)
        # Nachbarn finden
        neighbors = find_neighbors(all_positions, h)
        
        # Kernel-Gradient berechnen
        #grad_w = kernel_gradient(all_positions_x, all_positions_y, h, neighbors)
        
        # Tau berechnen
        #tau = calculate_tau(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, mu, neighbors)

        # Komponenten der Divergenz des Spannungstensors S_x und S_y berechnen
        #S_x, S_y = calculate_S(mass_per_particle, tau, rho, grad_w, neighbors)

        # Geschwindigkeiten mittels S_x und S_y aktualisieren
        #all_velocities_x, all_velocities_y = update_velocities_step_2(all_velocities_x, all_velocities_y, S_x, S_y, delta_t)

        # Positionen mittels S_x und S_y aktualisieren
        #all_positions_x, all_positions_y = update_positions_step_2(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, delta_t)

        # Schritt 3 (Dreischrittalgorithmus)
        # temporäre Dichte berechnen
        # Nachbarn finden
        #neighbors = find_neighbors(all_positions_x, all_positions_y, h)
        
        # Kernel-Gradient berechnen
        #grad_w = kernel_gradient(all_positions_x, all_positions_y, h, neighbors)

        # temporären Dichte brechnen
        #rho_temp = calculate_temporary_density(rho, mass_per_particle, all_velocities_x, all_velocities_y, grad_w, neighbors)

        # Druck berechnen
        #pressure = calculate_pressure(all_positions_x, all_positions_y, rho, rho_temp, delta_t, neighbors, grad_w, pressure, mass_per_particle, eta, initial_pressure)

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
