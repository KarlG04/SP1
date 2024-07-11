# main.py
# Importiere benötigte Module
import numpy as np
import boundary
import solver
import visualization
import isph_solver

def main():
    print("Das Programm startet hier.")

# Fluid-Eigenschaften gegeben
initial_density = 1  # Dichte des Wassers in kg/m³
diameter_particle = 5  # Partikeldurchmesser in m
dynamic_viscosity = 0.5    # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))

# Fluid-Eigenschaften berechnet
spacing = diameter_particle  # Initialer Partikelabstand
area_per_particle = np.pi * (diameter_particle / 2) ** 2 # Fläche eines Partikels in m²
volume_per_particle = area_per_particle # Volumen in m³ (für 1D Tiefe)
mass_per_particle = 1
smoothing_length = 5
kinematic_viscosity = dynamic_viscosity/initial_density # Kinematische viskosität berechnen

# Kernel Factors
density_factor = (315 * mass_per_particle) / (64 * np.pi * smoothing_length**9)
pressure_factor = (-(65 * mass_per_particle) / (np.pi * smoothing_length**6)) # 65 war bei 45
viscosity_factor = (45 * dynamic_viscosity * mass_per_particle) / (np.pi * smoothing_length**6)

print (density_factor)
print (pressure_factor)
print (viscosity_factor)

# Weitere Simulationsparameter
num_time_steps = 2000 # Anzahl an Berechnungsintervallen
isentropic_exponent = 20 
boundary_damping = -0.6
delta_t = 0.01

#Anfangsbedingungen
gravity = (0.0, -0.0981)  # Gravitationskraft in m/s² (x-Komponente, y-Komponente)

#Boxparameter
box_length = 200 # Länge der Box in m
box_height = 200 # Höhe der Box in m
fluid_length = 50 # initiale Länge des Fluid-Blocks in m
fluid_height = 150 # initiale Höhe des Fluid-Blocks in m

boundary_spacing = 1*spacing # Abstand der Boundary Partikel
wall_layers = 1 # Anzahl der Wandschichten

# Boxpunkte berechnen 
boundary_points, inlet_points, boundary_description = boundary.calculate_box_points(box_length, box_height, fluid_length, fluid_height, spacing, boundary_spacing, wall_layers)
#visualization.visualize_boundary(boundary_points, inlet_points, diameter_particle)

Fluid_Points, delta_ts = solver.run_simulation(inlet_points, gravity, initial_density, num_time_steps, spacing, smoothing_length, isentropic_exponent, delta_t, box_length, box_height, boundary_damping, density_factor, pressure_factor, viscosity_factor)

visualization.visualize_flow(boundary_points, inlet_points, Fluid_Points, delta_ts, diameter_particle, smoothing_length)


if __name__ == "__main__":
    main()