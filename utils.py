def run_simulation(inlet_points, gravity, initial_rho, nu, mass_per_particle, num_time_steps, spacing, h, boundary_positions, boundary_description, eta, cfl, delta_t, gamma, c_0, n_1, n_2, r_0, box_length, box_height):
    # Initialisieren der Simulation
    fluid_positions, fluid_velocities = initialize_fluid(inlet_points)
    current_rho = [initial_rho for _ in inlet_points]  # Initiale Dichten für jedes Partikel
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

        delta_t = calculate_timestep(fluid_velocities, cfl, spacing)

        neighbors = find_neighbors(fluid_positions, h)

        grad_w = compute_grad_w(fluid_positions, neighbors, h)

        viscosity_acceleration = calculate_viscosity_acceleration(fluid_positions, fluid_velocities, neighbors, grad_w, nu, eta, mass_per_particle, current_rho)
              
        boundary_neighbors = find_neighbors_boundary(fluid_positions, boundary_positions, r_0)

        boundary_force = calculate_boundary_force(fluid_positions, fluid_velocities, boundary_positions, boundary_neighbors, boundary_description, n_1, n_2, r_0, c_0)


        mirror_positions, mirror_velocities, mirrored_particle_indices = calculate_mirror_particles(fluid_positions, fluid_velocities, box_length, box_height, h)

        fluid_and_mirror_positions, fluid_and_mirror_velocities, fluid_and_mirror_current_rho = merge_fluid_and_mirror_particles(fluid_positions, mirror_positions, fluid_velocities, mirror_velocities, mirrored_particle_indices, current_rho)
        
        neighbors_fluid_and_mirror = find_neighbors_fluid_and_mirror(fluid_and_mirror_positions, h)

        grad_w_fluid_and_mirror = compute_grad_w_fluid_and_mirror(fluid_and_mirror_positions, neighbors_fluid_and_mirror, h) 

        new_rho, fluid_and_mirror_new_rho = calculate_density(fluid_positions, fluid_and_mirror_current_rho, fluid_and_mirror_velocities, neighbors_fluid_and_mirror, grad_w_fluid_and_mirror, delta_t, mass_per_particle, initial_rho)
        fluid_and_mirror_current_rho = fluid_and_mirror_new_rho  # Aktualisierung der Dichte für den nächsten Zeitschritt
        current_rho = new_rho

        fluid_and_mirror_pressure = calculate_pressure(fluid_and_mirror_current_rho, initial_rho, gamma, c_0)

        pressure_acceleration = calculate_pressure_acceleration(fluid_positions, fluid_and_mirror_positions, fluid_and_mirror_current_rho, fluid_and_mirror_pressure, neighbors_fluid_and_mirror, grad_w_fluid_and_mirror, mass_per_particle)   


        fluid_velocities, fluid_positions = integrate_substantial_acceleration(fluid_positions, fluid_velocities, pressure_acceleration, viscosity_acceleration, boundary_force, delta_t, gravity) 
        

        # Ergebnisse für den aktuellen Zeitschritt speichern
        delta_t_collected.append(delta_t)  # delta_t sammeln
        positions_collected.append(fluid_positions.copy()) # Positionen sammeln
        velocities_collected.append(fluid_velocities.copy()) # Geschwindigkeiten sammeln
        mirror_positions_collected.append(mirror_positions.copy()) # Positionen der Spiegelpartikel sammeln
        mirror_velocities_collected.append(mirror_velocities.copy()) # Geschwindigkeiten der Spiegelpartikel sammeln

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

    # Spiegelpartikel in Listenform initialisieren
    mirror_particles = [[], [], [], []]

    for t in range(num_time_steps):
        mirror_positions_x_collected = [pos[0] for pos in mirror_positions_collected[t]]
        mirror_positions_y_collected = [pos[1] for pos in mirror_positions_collected[t]]
        mirror_velocities_x_collected = [vel[0] for vel in mirror_velocities_collected[t]]
        mirror_velocities_y_collected = [vel[1] for vel in mirror_velocities_collected[t]]

        mirror_particles[0].append(mirror_positions_x_collected)
        mirror_particles[1].append(mirror_positions_y_collected)
        mirror_particles[2].append(mirror_velocities_x_collected)
        mirror_particles[3].append(mirror_velocities_y_collected)

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

    return fluid_particles, delta_t_collected, mirror_particles