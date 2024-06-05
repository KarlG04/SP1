# main.py
# Importiere benötigte Module
import numpy as np
import boundary
import solver
import visualization

def main():
    print("Das Programm startet hier.")

# Boundary Parameter
# Rohrparameter
pipe_1_length = 0.2 # Länge des geraden Rohrabschnitt (Einlass) in m
pipe_2_length = 0.01 # Länge des geraden Rohrabschnitt (Auslass) in m
manifold_radius = 0.015 # Äußerer Krümmungsradius in m
pipe_diameter = 0.01 # Durchmesser des Rohres in m

#Boxparameter
box_length = 0.05 # Länge der Box in m
box_height = 0.025 # Häche der Box in m

wall_layers = 1 # Anzahl der Wandschichten

# Fluid-Eigenschaften gegeben
rho = 1000  # Dichte des Wassers in kg/m³
diameter_particle = 1 * 1e-3 # Partikeldurchmesser in m
mu = 1 * 1e-3    # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))
initial_pressure = 0.0 # Initialer Druck (für ersten Iterationsschritt) in N/m²

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2 # Fläche eines Partikels in m²
volume_per_particle = area_per_particle # Volumen in m³ (für 1D Tiefe)
mass_per_particle = volume_per_particle * rho # Masse eines Partikels in kg
h = 1.5 * spacing # Glättungsradius in m

# Weitere Simulationsparameter
num_time_steps = 114 # Anzahl an Berechnungsintervallen
eta = 0.1 * h # Regulierungsparameter für den Dreischrittalgorythmus
cfl = 0.1 # Konstante damit der Zeitschritt nicht zu groß wird (gängig 0.1)
beta = 0.1  # Faktor für Diffusionsbedingung
delta_t_diffusion = (beta * rho * spacing**2)/mu
animation_interval = 1 # Faktor zur animationsgeschwindigkeit

#Anfangsbedingungen
initial_velocity = [2.0, 0.0] # Anfangsgeschwindigkeit in m/s
gravity = [0.0, -9.81]  # Gravitationskraft in m/s² (x-Komponente, y-Komponente)

# Krümmerpunkte berechnen 
#boundary_points, inlet_points, outlet_points = boundary.calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, spacing, wall_layers)
#visualization.visualize_boundary_outlet(boundary_points, inlet_points, outlet_points, diameter_particle)

# Boxpunkte berechnen 
boundary_points, inlet_points = boundary.calculate_box_points(box_length, box_height, spacing, wall_layers)
#visualization.visualize_boundary(boundary_points, inlet_points, diameter_particle)

Fluid_Points, delta_ts = solver.run_simulation(inlet_points, initial_velocity, gravity, cfl, rho, mu, mass_per_particle, num_time_steps, spacing, h, boundary_points, eta, initial_pressure, delta_t_diffusion)

visualization.visualize_flow(boundary_points, inlet_points, Fluid_Points, delta_ts, diameter_particle, h)


if __name__ == "__main__":
    main()