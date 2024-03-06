# main.py
# Importiere benötigte Module
import pipe
import solver
import visualization

def main():
    print("Das Programm startet hier.")

# Rohrparameter
pipe_1_length = 200  # Länge des geraden Rohrabschnitt (Einlass) in mm
pipe_2_length = 100 # Länge des geraden Rohrabschnitt (Auslass) in mm
manifold_radius = 60  # Äußerer Krümmungsradius in mm
pipe_diameter = 15 # Durchmesser des Rohres in mm
point_density = 1000  # Punkte pro mm für die Diskretisierung der Krümmergeometrie
wall_layers = 6 # Anzahl der Wandschichten

# Fluid-Eigenschaften
rho = 1000  # Dichte des Wassers in kg/m^3
inlet_density = rho/2 # Dichte der Punkte für den Einlass
mu = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))

#Anfangsbedingungen
initial_velocitie = 2 # Anfangsgeschwindigkeit in m/s


# Krümmerpubkte berechnen 
pipe_points, inlet_points, outlet_points = pipe.calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, point_density, wall_layers, inlet_density)
# visualisieren
visualization.visualize_pipe_points(pipe_points, inlet_points, outlet_points)


# Visualisiere die Strömung




# Dieser Block stellt sicher, dass main() nur ausgeführt wird, wenn diese Datei direkt ausgeführt wird,
# und nicht, wenn sie in einer anderen Datei importiert wird.
if __name__ == "__main__":
    main()