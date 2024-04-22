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
point_density = 0.8  # Punkte pro mm für die Diskretisierung der Krümmergeometrie
wall_layers = 3 # Anzahl der Wandschichten

# Fluid-Eigenschaften
rho = 1000  # Dichte des Wassers in kg/m³
initial_particle_distance = 1 # Abstand der Punkte in mm
mu = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))
gravity = [0.0, -9810.0]  # Gravitationskraft in mm/s² (x-Komponente, y-Komponente)


#Anfangsbedingungen
initial_velocity = [-3000.0, 0.0] # Anfangsgeschwindigkeit in mm/s

# Weitere Simulationsparameter
num_time_steps = 3000
delta_t = 0.01  # Zeitschritt in s
delta_t_coefficient = 0.1 # Konstante damit der Zeitschritt nicht zu groß wird (gängig 0.1)
animation_interval = 1 # Faktor zur animationsgeschwindigkeit

# Krümmerpubkte berechnen 
pipe_points, inlet_points, outlet_points = pipe.calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, point_density, wall_layers, initial_particle_distance)

#visualization.visualize(pipe_points, inlet_points, outlet_points)
Fluid_Points,delta_ts = solver.run_simulation(inlet_points, initial_velocity, gravity, delta_t_coefficient, initial_particle_distance, num_time_steps)

visualization.visualize_flow(pipe_points, inlet_points, outlet_points, Fluid_Points, delta_ts)

#visualization.visualize_flow_animation(pipe_points, inlet_points, outlet_points, Fluid_Points, delta_ts, animation_interval)
# Dieser Block stellt sicher, dass main() nur ausgeführt wird, wenn diese Datei direkt ausgeführt wird,
# und nicht, wenn sie in einer anderen Datei importiert wird.

if __name__ == "__main__":
    main()