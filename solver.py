import numpy as np
import matplotlib.pyplot as plt


def initialize_particles(inlet_points, initial_velocity):
    # Die Anzahl der Partikel bestimmen
    num_particles = len(inlet_points)
    
    # Initialisiere Arrays für die Eigenschaften der Partikel
    positions = np.array(inlet_points)  # Positionen aus den Einlasspunkten
    velocities = np.tile(initial_velocity, (num_particles, 1))  # Anfangsgeschwindigkeit für alle Partikel
    pressures = np.zeros(num_particles)  # Anfangsdruck für alle Partikel (kann initial 0 sein)

    return positions, velocities, pressures

# Zeitschritt berechnen
def calculate_time_step(velocities, delta_t_coefficient, initial_particle_distance):

    # Maximale Geschwindigkeit ermitteln 
    v_max = np.max(np.linalg.norm(velocities, axis=1))    

    # Courant Bedingung
    delta_t = delta_t_coefficient * initial_particle_distance / v_max

    return delta_t


# First step of explict ISPH algorythm
def update_velocity_due_to_gravity(velocities, gravity, delta_t):
    # Erstelle eine Kopie der aktuellen Geschwindigkeiten, um die neuen Geschwindigkeiten zu berechnen
    velocities_due_to_gravity = velocities.copy()
    
    # Aktualisiere jede Komponente der Geschwindigkeit basierend auf der Gravitationskraft und der Zeit
    velocities_due_to_gravity[:, 0] += gravity[0] * delta_t  # Update u (Geschwindigkeit in x-Richtung)
    velocities_due_to_gravity[:, 1] += gravity[1] * delta_t  # Update v (Geschwindigkeit in y-Richtung)
    
    return velocities_due_to_gravity


def update_positions(positions, velocities, delta_t):

    # Berechne die neuen Positionen basierend auf den aktuellen Geschwindigkeiten und delta_t
    Fluid_positions = positions + velocities * delta_t

    return Fluid_positions