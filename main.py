# main.py
# Importiere benötigte Module
import pipe
import solver


def main():
    print("Das Programm startet hier.")

# Rohrparameter
pipe_lengh = 100  # Länge der geraden Rohrabschnitte in mm
manifold_radius = 100  # Äußerer Krümmungsradius in mm
pipe_diameter = 60 # Durchmesser des Rohres in mm
point_density = 1  # Punkte pro mm für die Diskretisierung der Krümmergeometrie

# Fluid-Eigenschaften
fluid_density = 1000  # Dichte des Wassers in kg/m^3
fluid_viskosity = 0.001  # Dynamische Viskosität von Wasser bei Raumtemperatur in Pa·s (oder kg/(m·s))

# Anfangsbedingungen
initial_velocity = 1.0  # Anfangsgeschwindigkeit des Fluids in m/s (gleichförmiges Profil)
inlet_pressure = 101325  # Druck am Einlass in Pa (atmosphärischer Druck)

# Weitere Simulationseinstellungen

# Krümmerberechnungen durchführen
pipe_points_x, pipe_points_y = pipe.calculate_pipe_points(pipe_lengh, manifold_radius, pipe_diameter, point_density)

# Visualisierung der Krümmergeometrie
pipe.visualize_pipe_points(pipe_points_x, pipe_points_y)



# Dieser Block stellt sicher, dass main() nur ausgeführt wird, wenn diese Datei direkt ausgeführt wird,
# und nicht, wenn sie in einer anderen Datei importiert wird.
if __name__ == "__main__":
    main()