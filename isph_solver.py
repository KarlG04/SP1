import numpy as np
import time
from scipy.spatial import KDTree
import math

# Fluid-Eigenschaften gegeben
initial_density = 1000  # Dichte des Wassers in kg/m³
diameter_particle = 0.02  # Partikeldurchmesser in m
dynamic_viscosity = 0.01    # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2 # Fläche eines Partikels in m²
volume_per_particle = area_per_particle # Volumen in m³ (für 1D Tiefe)
mass_per_particle = 1
smoothing_length = 1.5 * spacing
kinematic_viscosity = dynamic_viscosity/initial_density # Kinematische viskosität berechnen

# Weitere Simulationsparameter
num_time_steps = 200 # Anzahl an Berechnungsintervallen
boundary_damping = -0.6
delta_t = 0.01

#Anfangsbedingungen
gravity = (0.0, -9.81)  # Gravitationskraft in m/s² (x-Komponente, y-Komponente)

#Boxparameter
box_length = 2 # Länge der Box in m
box_height = 2 # Höhe der Box in m
fluid_length = 0.5 # initiale Länge des Fluid-Blocks in m
fluid_height = 0.5 # initiale Höhe des Fluid-Blocks in m

boundary_spacing = 1*spacing # Abstand der Boundary Partikel
wall_layers = 1 # Anzahl der Wandschichten

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

def run_simulation(inlet_points, gravity, initial_density, num_time_steps, spacing, smoothing_length, isentropic_exponent, delta_t, box_length, box_height, boundary_damping, density_factor, pressure_factor, viscosity_factor):
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

        neighbors, distances = find_neighbors(fluid_positions, smoothing_length)

        grad_w = calculate_grad_w(smoothing_length, distances, fluid_positions)
        print(grad_w)

        fluid_positions, fluid_velocities = enforce_boundary_condition(fluid_positions, fluid_velocities, box_length, box_height, spacing, boundary_damping)

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