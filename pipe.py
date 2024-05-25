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
    print(f"Länge von all_positions_x: {len(all_positions_x)}, all_positions_y: {len(all_positions_y)}")

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
    print(f"Länge von all_velocities_x: {len(all_velocities_x)}, all_velocities_y: {len(all_velocities_y)}")

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

    print(f"Anzahl der Partikel: {num_particles}")
    print(f"Länge von all_velocities_x: {len(all_velocities_x)}, all_velocities_y: {len(all_velocities_y)}")

    for i in range(num_particles):
        tau_ij = [[0, 0], [0, 0]]
        for j in neighbors[i]:  # neighbors[i] enthält die Indizes der Nachbarn von Partikel i
            if i != j:  # Vermeide Selbstbezug
                if j >= num_particles:
                    print(f"Warnung: Nachbarindex {j} liegt außerhalb des gültigen Bereichs!")
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
    delta_ts = []  # List to store delta_t values for each time step
    tau_values = []  # List to store shear stress values for each time step
    
    # Calculate the interval for adding new particles
    particle_add_interval = calculate_particle_add_interval(initial_velocity, spacing)
    time_since_last_addition = 0  # Initialize time counter for particle addition

    iteration_start_time = time.perf_counter()  # Startzeit für Iterationen messen
    for t in range(num_time_steps - 1):
        print(f"Running iteration {t+2}/{num_time_steps}")  # Outputs the current iteration number
        # Calculate time step based on velocities
        delta_t = calculate_time_step([vx[t] for vx in velocities_x], [vy[t] for vy in velocities_y], cfl, spacing, delta_t_diffusion)
        delta_ts.append(delta_t)  # Collect delta_t
        time_since_last_addition += delta_t  # Update time since last addition

        # Check if it's time to add new particles
        if time_since_last_addition >= particle_add_interval:
            add_new_particles(positions_x, positions_y, velocities_x, velocities_y, inlet_points, initial_velocity, t)
            time_since_last_addition = 0  # Reset the time counter after adding particles
        
        # Erster Schritt (Dreischrittalogarythmus)
        # Update velocities due to gravity
        update_velocity_step_1(velocities_x, velocities_y, gravity, delta_t, t)
        
        # Update positions
        update_positions_step_1(positions_x, positions_y, velocities_x, velocities_y, delta_t, t)

        # Zweiter Schritt (Dreischrittalogarythmus)
        # Positionen zusammenführen
        #all_positions_x, all_positions_y = merge_positions(positions_x, positions_y, boundary_points)
        
        # Geschwindigkeiten zusammenführen
        #all_velocities_x, all_velocities_y = merge_velocities([vel[t+1] for vel in velocities_x], [vel[t+1] for vel in velocities_y], boundary_points)
        
        # Nachbarn finden
        #neighbors = find_neighbors(all_positions_x, all_positions_y, h)
     
        # Kernel berechnen
        #w = kernel(all_positions_x, all_positions_y, h, neighbors)
        
        # Kernel Gradient berechnen
        #grad_w = kernel_gradient(all_positions_x, all_positions_y, h, neighbors)
        
        # Tau berechnen
        #tau = calculate_tau(all_positions_x, all_positions_y, all_velocities_x, all_velocities_y, mu, neighbors)
        #tau_values.append(tau)  # Speichert die Scherspannung für den aktuellen Zeitschritt



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

    return Fluid_Points, delta_ts, tau_values