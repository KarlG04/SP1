import numpy as np

def calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, point_density, wall_layers, initial_particle_distance):
    # Berechnungen der Schichten
    def calculate_layer(offset):
        # Äßere Schicht
        outer_manifold_radius = manifold_radius + pipe_diameter/2

        outer_straight_pipe_1_x = np.linspace(0, -pipe_1_length, int(pipe_1_length * point_density)) 
        outer_straight_pipe_1_y = np.zeros_like(outer_straight_pipe_1_x) - offset

        outer_straight_pipe_2_y = np.linspace(outer_manifold_radius, outer_manifold_radius + pipe_2_length, int(pipe_2_length * point_density))
        outer_straight_pipe_2_x = np.ones_like(outer_straight_pipe_2_y) * (-outer_manifold_radius - pipe_1_length - offset)

        angle = np.linspace(3/2*np.pi, np.pi, int(np.pi * (outer_manifold_radius + offset) * point_density /2))
        outer_manifold_x = -pipe_1_length + ((outer_manifold_radius + offset) * np.cos(angle)) 
        outer_manifold_y = outer_manifold_radius + ((outer_manifold_radius + offset) * np.sin(angle))

        # Innere Schicht
        inner_manifold_radius = manifold_radius - pipe_diameter/2

        inner_straight_pipe_1_x = np.linspace(-pipe_1_length, 0, int(pipe_1_length * point_density))
        inner_straight_pipe_1_y = np.linspace(pipe_diameter + offset, pipe_diameter + offset, int(pipe_1_length * point_density))

        inner_straight_pipe_2_y = np.linspace(outer_manifold_radius, outer_manifold_radius + pipe_2_length, int(pipe_2_length * point_density))
        inner_straight_pipe_2_x = np.linspace(pipe_diameter - pipe_1_length - outer_manifold_radius + offset, pipe_diameter - pipe_1_length - outer_manifold_radius + offset, int(pipe_2_length * point_density))

        angle = np.linspace(3/2*np.pi, np.pi, int(np.pi * (inner_manifold_radius - offset) * point_density /2))
        inner_manifold_x = -pipe_1_length + (inner_manifold_radius - offset) * np.cos(angle)
        inner_manifold_y = outer_manifold_radius + (inner_manifold_radius - offset) * np.sin(angle)

        return outer_straight_pipe_1_x, outer_straight_pipe_1_y, outer_manifold_x, outer_manifold_y, outer_straight_pipe_2_x, outer_straight_pipe_2_y, inner_straight_pipe_1_x, inner_straight_pipe_1_y, inner_manifold_x, inner_manifold_y, inner_straight_pipe_2_x, inner_straight_pipe_2_y

    # Startpunkte ohne Offset
    layers_points = []
    for i in range(wall_layers):
        offset = i * 1/point_density  # Offset basierend auf der aktuellen Schicht und Punkt Dichte
        layers_points.append(calculate_layer(offset))

    # inlet_points - Vertikale Linie
    inlet_points_y = np.linspace(0, pipe_diameter, int(pipe_diameter * initial_particle_distance))[1:-1]
    inlet_points_x = np.zeros_like(inlet_points_y)

    inlet_points = np.vstack([inlet_points_x, inlet_points_y]).T

    # outlet_points - Horizontale Linie
    outlet_points_x = np.linspace(-pipe_1_length - manifold_radius + pipe_diameter/2, -pipe_1_length - manifold_radius - pipe_diameter/2, int(pipe_diameter * initial_particle_distance))
    outlet_points_y = np.ones_like(outlet_points_x) * (manifold_radius + pipe_2_length + pipe_diameter/2)  # Y-Koordinate angepasst für die Positionierung

    outlet_points = np.vstack([outlet_points_x, outlet_points_y]).T

    # Kombinieren der Punkte aller Schichten
    combined_x, combined_y = [], []
    for layer in layers_points:
        for lx, ly in zip(layer[::2], layer[1::2]):  # Schrittweite von 2, um x und y Koordinaten zu paaren
            combined_x.extend(lx)
            combined_y.extend(ly)

    # Kombinieren der X- und Y-Koordinaten in einem Array
    pipe_points = np.vstack([combined_x, combined_y]).T

    return pipe_points, inlet_points, outlet_points