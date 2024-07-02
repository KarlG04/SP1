# main.py
# Importiere benötigte Module
import numpy as np
import boundary
import solver
import visualization

def main():
    print("Das Programm startet hier.")

# Fluid-Eigenschaften gegeben
initial_density = 1000.0  # Dichte des Wassers in kg/m³
diameter_particle = 0.015 # Partikeldurchmesser in m
mu = 1 * 1e-3    # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2 # Fläche eines Partikels in m²
volume_per_particle = area_per_particle # Volumen in m³ (für 1D Tiefe)
mass_per_particle = 0.1
h = 1.5 * spacing # Glättungsradius in m
nu = mu/initial_density # Kinematische viskosität berechnen

# Weitere Simulationsparameter
num_time_steps = 500 # Anzahl an Berechnungsintervallen
eta = 0.01 # Regulierungsparameter für den Dreischrittalgorythmus

#Anfangsbedingungen
gravity = (0.0, -9.81)  # Gravitationskraft in m/s² (x-Komponente, y-Komponente)

#Boxparameter
box_length = 0.4 # Länge der Box in m
box_height = 0.4 # Höhe der Box in m
fluid_length = 0.2 # initiale Länge des Fluid-Blocks in m
fluid_height = 0.2 # initiale Höhe des Fluid-Blocks in m

boundary_spacing = 1*spacing # Abstand der Boundary Partikel
wall_layers = 1 # Anzahl der Wandschichten


# Berechne c_0 basierend auf der größeren Komponente der Gravitation
max_gravity = max(abs(gravity[0]), abs(gravity[1]))
c_0 = 20 #10 * (2 * max_gravity * fluid_height) ** 0.5
n_1 = 12.0
n_2 = 4.0
r_0 = spacing
boundary_factor = 3 * 1e2


# Berechne delta_t
delta_t = 4 * 1e-4
gamma = 7.0


# Boxpunkte berechnen 
boundary_points, inlet_points, boundary_description = boundary.calculate_box_points(box_length, box_height, fluid_length, fluid_height, spacing, boundary_spacing, wall_layers)
#visualization.visualize_boundary(boundary_points, inlet_points, diameter_particle)

Fluid_Points, delta_ts = solver.run_simulation(inlet_points, gravity, initial_density, nu, mass_per_particle, num_time_steps, spacing, h, eta, delta_t, box_length, box_height, c_0, gamma, n_1, n_2, r_0, boundary_factor)

visualization.visualize_flow(boundary_points, inlet_points, Fluid_Points, delta_ts, diameter_particle, h)


if __name__ == "__main__":
    main()