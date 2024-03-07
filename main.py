# main.py
# Importiere benötigte Module
import pipe
import solver
import visualization

def main():
    print("Das Programm startet hier.")

# Rohrparameter
pipe_1_length = 100  # Länge des geraden Rohrabschnitt (Einlass) in mm
pipe_2_length = 50 # Länge des geraden Rohrabschnitt (Auslass) in mm
manifold_radius = 60  # Äußerer Krümmungsradius in mm
pipe_diameter = 15 # Durchmesser des Rohres in mm
point_density = 10  # Punkte pro mm für die Diskretisierung der Krümmergeometrie
wall_layers = 4 # Anzahl der Wandschichten

# Fluid-Eigenschaften
rho = 1000  # Dichte des Wassers in kg/m³
initial_particle_distance = 1 # Abstand der Punkte in m
mu = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))
gravity = [0.0, -9.81]  # Gravitationskraft in m/s² (x-Komponente, y-Komponente)


#Anfangsbedingungen
initial_velocity = [2.0, 0.0] # Anfangsgeschwindigkeit in m/s

# Weitere Simulationsparameter
num_steps = 4
delta_t = 0.01  # Zeitschritt in Sekunden
delta_t_coefficient = 0.1 # Konstante damit der Zeitschritt nicht zu groß wir (gängig 0.1)

# Krümmerpubkte berechnen 
pipe_points, inlet_points, outlet_points = pipe.calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, point_density, wall_layers, initial_particle_distance)


positions, velocities, pressures = solver.initialize_particles(inlet_points, initial_velocity)




for step in range(num_steps):
    # Zeitschritt berechnen
    delta_t = solver.calculate_time_step(velocities, delta_t_coefficient, initial_particle_distance)

    # Aktualisiere die Geschwindigkeiten (falls notwendig)
    velocities = solver.update_velocity_due_to_gravity(velocities, gravity, delta_t)
    
    # Aktualisiere die Positionen der Fluidpartikel
    fluid_positions = solver.update_positions(positions, velocities, delta_t)
    
    # Visualisiere die aktuelle Szene
    visualization.visualize(pipe_points, inlet_points, outlet_points, fluid_positions)
    
    # Hier könntest du die neuen Positionen als die aktuellen für den nächsten Schritt setzen
    positions = fluid_positions


# Dieser Block stellt sicher, dass main() nur ausgeführt wird, wenn diese Datei direkt ausgeführt wird,
# und nicht, wenn sie in einer anderen Datei importiert wird.
if __name__ == "__main__":
    main()