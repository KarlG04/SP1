import numpy as np
import time

# Rohrparameter
pipe_1_length = 0.2 # Länge des geraden Rohrabschnitt (Einlass) in m
pipe_2_length = 0.2 # Länge des geraden Rohrabschnitt (Auslass) in m
manifold_radius = 0.12 # Äußerer Krümmungsradius in m
pipe_diameter = 0.02 # Durchmesser des Rohres in m
wall_layers = 5 # Anzahl der Wandschichten

# Fluid-Eigenschaften gegeben
rho = 1000  # Dichte des Wassers in kg/m³
diameter_particle = 1 * 1e-3 # Partikeldurchmesser in m
mu = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))
delta_t = 0.01  # Zeitschritt in s
cfl = 0.1 # Konstante damit der Zeitschritt nicht zu groß wird (gängig 0.1)
beta = 0.1  # Faktor für Diffusionsbedingung

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2 # Fläche eines Partikels in m²
volume_per_particle = area_per_particle # Volumen in m³ (für 1D Tiefe)
mass_per_particle = volume_per_particle * rho # Masse eines Partikels in kg
h = 1.5 * spacing # Glättungsradius in m

#Anfangsbedingungen
initial_velocity = [-3.0, 0.0] # Anfangsgeschwindigkeit in m/s (x-Komponente, y-Komponente)
gravity = [0.0, -9.81]  # Gravitationskraft in mm/s² (x-Komponente, y-Komponente)

# Weitere Simulationsparameter
num_time_steps = 2000 # Anzahl an Berechnungsintervallen
animation_interval = 1 # Faktor zur animationsgeschwindigkeit
delta_t_diffusion = (beta * rho * spacing**2)/mu

def initialize_simulation(inlet_points, initial_velocity):
    # Initialize lists for positions and velocities
    positions_x, positions_y, velocities_x, velocities_y = [], [], [], []
    for i in range(len(inlet_points)):
        positions_x.append([inlet_points[i][0]])
        positions_y.append([inlet_points[i][1]])
        velocities_x.append([initial_velocity[0]])
        velocities_y.append([initial_velocity[1]])
    return positions_x, positions_y, velocities_x, velocities_y

def calculate_time_step(velocities_x, velocities_y, cfl, spacing, delta_t_diffusion):
    velocities = np.sqrt(np.square(velocities_x) + np.square(velocities_y))
    v_max = np.max(velocities)
    delta_t_courant = cfl * spacing / v_max
    return min(delta_t_courant, delta_t_diffusion)

def add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity, t):
    # Extend existing lists with new particles
    for i in range(len(inlet_points)):
        positions_x.append([inlet_points[i][0]] * (t + 1))
        positions_y.append([inlet_points[i][1]] * (t + 1))
        velocities_x.append([initial_velocity[0]] * (t + 1))
        velocities_y.append([initial_velocity[1]] * (t + 1))

def calculate_particle_add_interval(initial_velocity, spacing):
    initial_velocity_magnitude = np.linalg.norm(initial_velocity)
    return spacing / initial_velocity_magnitude 

def update_velocity_step_1(velocities_x, velocities_y, gravity, delta_t, t):
    # Update velocity lists for gravity effect
    for i in range(len(velocities_x)):
        velocities_x[i].append(velocities_x[i][t] + gravity[0] * delta_t)
        velocities_y[i].append(velocities_y[i][t] + gravity[1] * delta_t)

def update_positions_step_1(positions_x, positions_y, velocities_x, velocities_y, delta_t, t):
    # Update position lists based on new velocities
    for i in range(len(positions_x)):
        positions_x[i].append(positions_x[i][t] + velocities_x[i][t+1] * delta_t)
        positions_y[i].append(positions_y[i][t] + velocities_y[i][t+1] * delta_t)



def merge_positions(positions_x, positions_y, boundary_points):
    # Extrahiere x und y Positionen aus den Boundary-Points
    boundary_positions_x = [point[0] for point in boundary_points]
    boundary_positions_y = [point[1] for point in boundary_points]
    
    # Füge Boundary-Positionen an den Anfang der Fluid-Positionen hinzu
    all_positions_x = np.concatenate([boundary_positions_x, positions_x])
    all_positions_y = np.concatenate([boundary_positions_y, positions_y])
    
    return all_positions_x, all_positions_y

def kernel(all_positions_x, all_positions_y, h):
    num_particles = len(all_positions_x)
    w = [[] for _ in range(num_particles)]  # Initialisiere eine Liste von Listen für die Gewichte

    alpha_D = 10 / (7 * np.pi * h**2)
    
    for i in range(num_particles):
        for j in range(num_particles):
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

def kernel_gradient_optimized(all_positions_x, all_positions_y, h):
    num_particles = len(all_positions_x)
    grad_w = [[] for _ in range(num_particles)]  # Initialisiere eine Liste von Listen für Gradienten

    alpha_D = 10 / (7 * np.pi * h**2)

    for i in range(num_particles):
        for j in range(num_particles):
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


def run_simulation(inlet_points, initial_velocity, gravity, cfl, rho, num_time_steps, spacing):
    # Initialize simulation
    positions_x, positions_y, velocities_x, velocities_y = initialize_simulation(inlet_points, initial_velocity)
    delta_ts = []  # List to store delta_t values for each time step
    
    # Calculate the interval for adding new particles
    particle_add_interval = calculate_particle_add_interval(initial_velocity, spacing)
    time_since_last_addition = 0  # Initialize time counter for particle addition

    iteration_start_time = time.perf_counter()  # Startzeit für Iterationen messen
    for t in range(num_time_steps - 1):
        print(f"Running iteration {t+2}/{num_time_steps}")  # Outputs the current iteration number
        # Calculate time step based on velocities
        delta_t = calculate_time_step([vx[t] for vx in velocities_x], [vy[t] for vy in velocities_y],cfl, spacing, delta_t_diffusion)
        delta_ts.append(delta_t)  # Collect delta_t
        time_since_last_addition += delta_t  # Update time since last addition

        # Check if it's time to add new particles
        if time_since_last_addition >= particle_add_interval:
            add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity, t)
            time_since_last_addition = 0  # Reset the time counter after adding particles
        
        # First Step (three step algorythm)
        # Update velocities due to gravity
        update_velocity_step_1(velocities_x, velocities_y, gravity, delta_t, t)
        
        # Update positions
        update_positions_step_1(positions_x, positions_y, velocities_x, velocities_y, delta_t, t)

    iteration_end_time = time.perf_counter()  # Endzeit für Iterationen messen
    iteration_time = iteration_end_time - iteration_start_time  # Berechnung der Zeit für Iterationen
    print(f"Iteration time: {iteration_time:.2f}s")
    print(" ")
    
    # Convert lists to numpy arrays for output
    print("Build Fluid_Points array")
    array_build_start_time = time.perf_counter()  # Startzeit für das Erstellen des Arrays messen
    Fluid_Points = np.array([positions_x, positions_y, velocities_x, velocities_y])
    array_build_end_time = time.perf_counter()  # Endzeit für das Erstellen des Arrays messen
    array_build_time = array_build_end_time - array_build_start_time  # Berechnung der Zeit für das Erstellen des Arrays
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

    return Fluid_Points, delta_ts
