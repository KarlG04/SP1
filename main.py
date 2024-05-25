# main.py
# Importiere benötigte Module
import numpy as np
import boundary
import solver
import visualization

def main():
    print("Das Programm startet hier.")

# Rohrparameter
pipe_1_length = 0.2 # Länge des geraden Rohrabschnitt (Einlass) in m
pipe_2_length = 0.2 # Länge des geraden Rohrabschnitt (Auslass) in m
manifold_radius = 0.12 # Äußerer Krümmungsradius in m
pipe_diameter = 0.02 # Durchmesser des Rohres in m
wall_layers = 5 # Anzahl der Wandschichten

# Fluid-Eigenschaften gegeben
rho = 1000  # Dichte des Wassers in kg/m³
diameter_particle = 5 * 1e-3 # Partikeldurchmesser in m
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
initial_velocity = [-3.0, 0.0] # Anfangsgeschwindigkeit in m/s
gravity = [0.0, -9.81]  # Gravitationskraft in mm/s² (x-Komponente, y-Komponente)

# Weitere Simulationsparameter
num_time_steps = 200 # Anzahl an Berechnungsintervallen
animation_interval = 1 # Faktor zur animationsgeschwindigkeit
delta_t_diffusion = (beta * rho * spacing**2)/mu

# Krümmerpunkte berechnen 
boundary_points, inlet_points, outlet_points = boundary.calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, spacing, wall_layers)
#visualization.visualize_boundary(boundary_points, inlet_points, outlet_points, diameter_particle)

Fluid_Points, delta_ts = solver.run_simulation(inlet_points, initial_velocity, gravity, cfl, rho, num_time_steps, spacing, boundary_points)

visualization.visualize_flow(boundary_points, inlet_points, outlet_points, Fluid_Points, delta_ts, diameter_particle)

#visualization.visualize_flow_animation(boundary_points, inlet_points, outlet_points, Fluid_Points, delta_ts, animation_interval)

if __name__ == "__main__":
    main()