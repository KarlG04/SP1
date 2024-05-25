import numpy as np
import time
from scipy.spatial import KDTree

# Rohrparameter
pipe_1_length = 0.2  # Länge des geraden Rohrabschnitt (Einlass) in m
pipe_2_length = 0.2  # Länge des geraden Rohrabschnitt (Auslass) in m
manifold_radius = 0.12  # Äußerer Krümmungsradius in m
pipe_diameter = 0.02  # Durchmesser des Rohres in m
wall_layers = 5  # Anzahl der Wandschichten

# Fluid-Eigenschaften gegeben
rho = 1000  # Dichte des Wassers in kg/m³
diameter_particle = 5 * 1e-3  # Partikeldurchmesser in m
mu = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))
delta_t = 0.01  # Zeitschritt in s
cfl = 0.1  # Konstante damit der Zeitschritt nicht zu groß wird (gängig 0.1)
beta = 0.1  # Faktor für Diffusionsbedingung

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2  # Fläche eines Partikels in m²
volume_per_particle = area_per_particle  # Volumen in m³ (für 1D Tiefe)
mass_per_particle = volume_per_particle * rho  # Masse eines Partikels in kg
h = 1.5 * spacing  # Glättungsradius in m

# Anfangsbedingungen
initial_velocity = [-3.0, 0.0]  # Anfangsgeschwindigkeit in m/s (x-Komponente, y-Komponente)
gravity = [0.0, -9.81]  # Gravitationskraft in mm/s² (x-Komponente, y-Komponente)

# Weitere Simulationsparameter
num_time_steps = 2000  # Anzahl an Berechnungsintervallen
animation_interval = 1  # Faktor zur animationsgeschwindigkeit
delta_t_diffusion = (beta * rho * spacing**2) / mu  # delta_t aufgrund der Diffusionsbedingung

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
    return positions_x, positions_y, velocities_x, velocities_y

def calculate_time_step(velocities_x, velocities_y, cfl, spacing, delta_t_diffusion):
    velocities = np.sqrt(np.square(velocities_x) + np.square(velocities_y))
    v_max = np.max(velocities)
    delta_t_courant = cfl * spacing / v_max
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

def update_velocity_step_1(velocities_x, velocities_y, gravity, delta_t):
    velocities_x = [vx + gravity[0] * delta_t for vx in velocities_x]
    velocities_y = [vy + gravity[1] * delta_t for vy in velocities_y]
    return velocities_x, velocities_y

def update_positions_step_1(positions_x, positions_y, velocities_x, velocities_y, delta_t):
    positions_x = [px + vx * delta_t for px, vx in zip(positions_x, velocities_x)]
    positions_y = [py + vy * delta_t for py, vy in zip(positions_y, velocities_y)]
    return positions_x, positions_y




def merge_positions(positions_x, positions_y, boundary_points):
    # Extrahiere x und y Positionen aus den Boundary-Points
    boundary_positions_x = np.array([point[0] for point in boundary_points])
    boundary_positions_y = np.array([point[1] for point in boundary_points])
    
    # Konvertiere die Positionslisten zu 1D-Arrays, falls sie es nicht sind
    positions_x = np.array(positions_x).flatten()
    positions_y = np.array(positions_y).flatten()
    
    # Füge Boundary-Positionen an den Anfang der Fluid-Positionen hinzu
    all_positions_x = np.concatenate((boundary_positions_x, positions_x))
    all_positions_y = np.concatenate((boundary_positions_y, positions_y))

    return all_positions_x, all_positions_y

def merge_velocities(velocities_x, velocities_y, boundary_points):
    # Anzahl der Boundary-Points
    num_boundary_points = len(boundary_points)

    # Konvertiere die Geschwindigkeitslisten zu 1D-Arrays, falls sie es nicht sind
    velocities_x = np.array(velocities_x).flatten()
    velocities_y = np.array(velocities_y).flatten()

    # Füge Nullen am Anfang der Fluid-Geschwindigkeiten hinzu
    all_velocities_x = np.concatenate((np.zeros(num_boundary_points), velocities_x))
    all_velocities_y = np.concatenate((np.zeros(num_boundary_points), velocities_y))

    return all_velocities_x, all_velocities_y

def find_neighbors(all_positions_x, all_positions_y, h):
    positions = np.array(list(zip(all_positions_x, all_positions_y)))
    tree = KDTree(positions)
    neighbors = [tree.query_ball_point(position, h) for position in positions]

    return neighbors

def kernel(all_positions_x, all_positions_y, h, neighbors):
    num_particles = len(all_positions_x)
    w = [[] for _ in range(num_particles)]  # Initialisiere eine Liste von Listen für die Gewichte

    alpha_D = 10 / (7 * np.pi * h**2)
    
    for i in range(num_particles):
        for j in neighbors[i]:  # Nur über die Nachbarn iterieren
            if i != j:  # Vermeiden der Berechnung für einen Partikel mit sich selbst
                dx = all_positions_x[j] - all_positions_x[i]
                dy = all_positions_y[j] - all_positions_y[i]
                r = np.sqrt(dx**2 + dy**2)
                s = r / h
                
                if r == 0 or r > h:
                    continue  # Vermeide Division durch Null und entfernte Nachbarn

                if s < 1:
                    weight = alpha_D * (1 - 1.5 * s**2 + 0.75 * s**3)
                elif s < 2:
                    weight = alpha_D * 0.25 * (2 - s)**3
                else:
                    continue  # Dieses Element bleibt Null und wird nicht verwendet.

                w[i].append((j, weight))

    return w

def kernel_gradient(all_positions_x, all_positions_y, h, neighbors):
    num_particles = len(all_positions_x)
    grad_w = [[] for _ in range(num_particles)]  # Initialisiere eine Liste von Listen für Gradienten

    alpha_D = 10 / (7 * np.pi * h**2)

    for i in range(num_particles):
        for j in neighbors[i]:  # Nur über die Nachbarn iterieren
            if i != j:
                dx = all_positions_x[j] - all_positions_x[i]
                dy = all_positions_y[j] - all_positions_y[i]
                r = np.sqrt(dx**2 + dy**2)
                
                if r == 0 or r > h:
                    continue  # Vermeide Division durch Null und entfernte Nachbarn

                s = r / h
                if s < 1:
                    factor = alpha_D * (-3 * s + 2.25 * s**2) / (r * h)
                elif s < 2:
                    factor = alpha_D * -0.75 * (2 - s)**2 / (r * h)
                else:
                    continue  # Dieses Element bleibt Null und wird nicht verwendet.

                grad_w[i].append((j, factor * np.array([dx, dy])))

    return grad_w

def calculate_tau(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, mu, neighbors):
    num_particles = len(all_positions_x)
    tau = []

    for i in range(num_particles):
        tau_ij = [[0, 0], [0, 0]]
        for j in neighbors[i]:  # neighbors[i] enthält die Indizes der Nachbarn von Partikel i
            if i != j:  # Vermeide Selbstbezug
                if j >= num_particles:
                    continue
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

    return tau


def run_simulation(inlet_points, initial_velocity, gravity, cfl, rho, num_time_steps, spacing, boundary_points):
    # Initialize simulation
    positions_x, positions_y, velocities_x, velocities_y = initialize_simulation(inlet_points, initial_velocity)

    # Gesuchte Werte für jeden Zeitschritt initialisieren
    delta_t_collected = []  # Liste zum speichern der delta_t Werte
    positions_x_collected = [] # Liste zum speichern der X-Positionen
    positions_y_collected = [] # Liste zum speichern der Y-Positionen
    velocities_x_collected = [] # Liste zum speichern der X-Geschwindikgeitskomponenten
    velocities_y_collected = [] # Liste zum speichern der Y-Geschwindigkeitskomponenten
    tau_collected = [] # Liste zum speichern der tau Werte

    # Calculate the interval for adding new particles
    particle_add_interval = calculate_particle_add_interval(initial_velocity, spacing)
    
    time_since_last_addition = 0  # Initialize time counter for particle addition
    iteration_start_time = time.perf_counter()  # Startzeit für Iterationen messen

    for t in range(num_time_steps):
        print(f"Running iteration {t+1}/{num_time_steps}")  # Outputs the current iteration number
        # Calculate time step based on velocities
        delta_t = calculate_time_step(velocities_x, velocities_y, cfl, spacing, delta_t_diffusion)
        time_since_last_addition += delta_t  # Update time since last addition

        # Check if it's time to add new particles
        if time_since_last_addition >= particle_add_interval:
            add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity)
            time_since_last_addition = 0  # Reset the time counter after adding particles

        # Erster Schritt (Dreischrittalgorythmus)
        # Update velocities due to gravity
        velocities_x, velocities_y = update_velocity_step_1(velocities_x, velocities_y, gravity, delta_t)
        
        # Update positions
        positions_x, positions_y = update_positions_step_1(positions_x, positions_y, velocities_x, velocities_y, delta_t)
        
        # Zweiter Schritt (Dreischrittalogarythmus)
        # Positionen zusammenführen
        all_positions_x, all_positions_y = merge_positions(positions_x, positions_y, boundary_points)
        
        # Geschwindigkeiten zusammenführen
        all_velocities_x, all_velocities_y = merge_velocities(velocities_x, velocities_y, boundary_points)
        
        # Nachbarn finden
        neighbors = find_neighbors(all_positions_x, all_positions_y, h)
     
        # Kernel berechnen
        w = kernel(all_positions_x, all_positions_y, h, neighbors)
        
        # Kernel Gradient berechnen
        grad_w = kernel_gradient(all_positions_x, all_positions_y, h, neighbors)
        
        # Tau berechnen
        tau = calculate_tau(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, mu, neighbors)


        # Ergebnisse für den akutellen Zeitschritt speichern
        delta_t_collected.append(delta_t)  # Collect delta_t
        positions_x_collected.append(positions_x.copy())
        positions_y_collected.append(positions_y.copy())
        velocities_x_collected.append(velocities_x.copy())
        velocities_y_collected.append(velocities_y.copy())
        tau_collected.append(tau)  # Speichert die Scherspannung für den aktuellen Zeitschritt



    iteration_end_time = time.perf_counter()  # Endzeit für Iterationen messen
    iteration_time = iteration_end_time - iteration_start_time  # Berechnung der Zeit für Iterationen
    print(f"Iteration time: {iteration_time:.2f}s")
    print(" ")

    # Determine the maximum number of particles at any time step
    max_particles = max(len(px) for px in positions_x_collected)

    # Build the Fluid_Points array with the maximum number of particles
    Fluid_Points = np.zeros((4, max_particles, num_time_steps))

    for t in range(num_time_steps):
        num_particles = len(positions_x_collected[t])
        Fluid_Points[0, :num_particles, t] = positions_x_collected[t]
        Fluid_Points[1, :num_particles, t] = positions_y_collected[t]
        Fluid_Points[2, :num_particles, t] = velocities_x_collected[t]
        Fluid_Points[3, :num_particles, t] = velocities_y_collected[t]

    array_build_end_time = time.perf_counter()  # Endzeit für das Erstellen des Arrays messen
    array_build_time = array_build_end_time - iteration_end_time  # Berechnung der Zeit für das Erstellen des Arrays
    print(f"Array build time: {array_build_time:.2f}s")
    print(" ")

    total_time = iteration_time + array_build_time  # Berechnung der Gesamtzeit
    print(f"Simulation completed in: {total_time:.2f}s")
    print(" ")

    number_particle_additions = len(positions_x) / len(inlet_points)
    print(f"fluid points: {len(inlet_points)} x {number_particle_additions:.0f} = {len(positions_x)}")
    print(" ")

    diameterµm = spacing * 1e6
    print(f"particle diameter: {diameterµm:.2f}µm")

    return Fluid_Points, delta_t_collected
