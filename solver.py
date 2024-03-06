import numpy as np
import matplotlib.pyplot as plt


def initialize_particles(inlet_points, initial_velocity):
    # Die Anzahl der Partikel bestimmen
    num_particles = len(inlet_points)
    
    # Initialisiere Arrays für die Eigenschaften der Partikel
    positions = np.array(inlet_points)  # Positionen aus den Einlasspunkten
    velocities = np.tile(initial_velocity, (num_particles, 1))  # Anfangsgeschwindigkeit für alle Partikel
    pressures = np.zeros(num_particles)  # Anfangsdruck für alle Partikel (kann initial 0 sein)

    return positions, velocities, densities, pressures, viscosities
