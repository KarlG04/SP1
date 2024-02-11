import numpy as np
import matplotlib.pyplot as plt

def calculate_pipe_points(pipe_lengh, manifold_radius, pipe_diameter, point_density):
    # Berechnungen 
    outer_inlet_points_x = np.linspace(0, -pipe_lengh, int(pipe_lengh * point_density))
    outer_inlet_points_y = np.zeros_like(outer_inlet_points_x)

    outer_outlet_points_y = np.linspace(manifold_radius, manifold_radius + pipe_lengh, int(pipe_lengh * point_density))
    outer_outlet_points_x = np.ones_like(outer_outlet_points_y) * (-manifold_radius -pipe_lengh)

    angle = np.linspace(3/2*np.pi, np.pi, int(np.pi * manifold_radius * point_density))
    outer_manifold_points_x = -pipe_lengh + manifold_radius * np.cos(angle)
    outer_manifold_points_y = manifold_radius + manifold_radius * np.sin(angle)


    inner_inlet_points_x = np.linspace(-pipe_lengh, 0, int(pipe_lengh * point_density))
    inner_inlet_points_y = np.linspace(pipe_diameter, pipe_diameter, int(pipe_lengh * point_density))

    inner_outlet_points_y = np.linspace(manifold_radius, manifold_radius + pipe_lengh, int(pipe_lengh * point_density))
    inner_outlet_points_x = np.linspace(pipe_diameter - pipe_lengh - manifold_radius, pipe_diameter - pipe_lengh - manifold_radius, int(pipe_lengh * point_density))

    inner_manifold_radius = manifold_radius - pipe_diameter
    angle = np.linspace(3/2*np.pi, np.pi, int(np.pi * inner_manifold_radius * point_density))
    krümmer_punkte_innen_x = -pipe_lengh + inner_manifold_radius * np.cos(angle)
    krümmer_punkte_innen_y = manifold_radius + inner_manifold_radius * np.sin(angle)


    pipe_points_x = np.concatenate([inner_inlet_points_x, outer_inlet_points_x, krümmer_punkte_innen_x, outer_manifold_points_x, inner_outlet_points_x, outer_outlet_points_x])
    pipe_points_y = np.concatenate([inner_inlet_points_y, outer_inlet_points_y, krümmer_punkte_innen_y, outer_manifold_points_y, inner_outlet_points_y, outer_outlet_points_y])

    #Ausgabe
    return pipe_points_x, pipe_points_y

def visualize_pipe_points(pipe_points_x, pipe_points_y):
    plt.figure(figsize=(10, 6))
    plt.plot(pipe_points_x, pipe_points_y, 'o', markersize=2)
    plt.axis('equal')
    plt.title('Visualisierung des Krümmers mit Punkten')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid(True)
    plt.show()

