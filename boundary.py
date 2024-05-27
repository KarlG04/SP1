import numpy as np

def calculate_pipe_points(pipe_1_length, pipe_2_length, manifold_radius, pipe_diameter, spacing, wall_layers):
    # Berechnungen der Schichten
    def calculate_layer(offset):
        # Äßere Schicht
        outer_manifold_radius = manifold_radius + pipe_diameter/2

        outer_straight_pipe_1_x = np.linspace(0, -pipe_1_length, int( pipe_1_length / spacing)) 
        outer_straight_pipe_1_y = np.zeros_like(outer_straight_pipe_1_x) - offset

        outer_straight_pipe_2_y = np.linspace(outer_manifold_radius, outer_manifold_radius + pipe_2_length, int(pipe_2_length / spacing))
        outer_straight_pipe_2_x = np.ones_like(outer_straight_pipe_2_y) * (-outer_manifold_radius - pipe_1_length - offset)

        angle = np.linspace(3/2*np.pi, np.pi, int(np.pi * (outer_manifold_radius + offset) / spacing /2))
        outer_manifold_x = -pipe_1_length + ((outer_manifold_radius + offset) * np.cos(angle)) 
        outer_manifold_y = outer_manifold_radius + ((outer_manifold_radius + offset) * np.sin(angle))

        # Innere Schicht
        inner_manifold_radius = manifold_radius - pipe_diameter/2

        inner_straight_pipe_1_x = np.linspace(-pipe_1_length, 0, int(pipe_1_length / spacing))
        inner_straight_pipe_1_y = np.linspace(pipe_diameter + offset, pipe_diameter + offset, int(pipe_1_length / spacing))

        inner_straight_pipe_2_y = np.linspace(outer_manifold_radius, outer_manifold_radius + pipe_2_length, int(pipe_2_length / spacing))
        inner_straight_pipe_2_x = np.linspace(pipe_diameter - pipe_1_length - outer_manifold_radius + offset, pipe_diameter - pipe_1_length - outer_manifold_radius + offset, int(pipe_2_length / spacing))

        angle = np.linspace(3/2*np.pi, np.pi, int(np.pi * (inner_manifold_radius - offset) / spacing /2))
        inner_manifold_x = -pipe_1_length + (inner_manifold_radius - offset) * np.cos(angle)
        inner_manifold_y = outer_manifold_radius + (inner_manifold_radius - offset) * np.sin(angle)

        return outer_straight_pipe_1_x, outer_straight_pipe_1_y, outer_manifold_x, outer_manifold_y, outer_straight_pipe_2_x, outer_straight_pipe_2_y, inner_straight_pipe_1_x, inner_straight_pipe_1_y, inner_manifold_x, inner_manifold_y, inner_straight_pipe_2_x, inner_straight_pipe_2_y

    # Startpunkte ohne Offset
    layers_points = []
    for i in range(wall_layers):
        offset = i * spacing  # Offset basierend auf der aktuellen Schicht und Punkt Dichte
        layers_points.append(calculate_layer(offset))

    # inlet_points - Vertikale Linie
    inlet_points_y = np.linspace(0, pipe_diameter, int(pipe_diameter / spacing))[1:-1]
    inlet_points_x = np.zeros_like(inlet_points_y)

    inlet_points = np.vstack([inlet_points_x, inlet_points_y]).T

    # outlet_points - Horizontale Linie
    outlet_points_x = np.linspace(-pipe_1_length - manifold_radius + pipe_diameter/2, -pipe_1_length - manifold_radius - pipe_diameter/2, int(pipe_diameter / spacing))
    outlet_points_y = np.ones_like(outlet_points_x) * (manifold_radius + pipe_2_length + pipe_diameter/2)  # Y-Koordinate angepasst für die Positionierung

    outlet_points = np.vstack([outlet_points_x, outlet_points_y]).T

    # Kombinieren der Punkte aller Schichten
    combined_x, combined_y = [], []
    for layer in layers_points:
        for lx, ly in zip(layer[::2], layer[1::2]):  # Schrittweite von 2, um x und y Koordinaten zu paaren
            combined_x.extend(lx)
            combined_y.extend(ly)

    # Kombinieren der X- und Y-Koordinaten in einem Array
    boundary_points = np.vstack([combined_x, combined_y]).T

    return boundary_points, inlet_points, outlet_points


def calculate_box_points(box_length, box_height, spacing, wall_layers):
    # Berechnungen der Schichten
    boundary_points_x, boundary_points_y = [], []
    for i in range(wall_layers):
        offset = i * spacing  # Offset basierend auf der aktuellen Schicht und Punkt Dichte
        
        bottom_x = np.linspace(-spacing * (wall_layers-1), box_length + spacing * (wall_layers-1), int((box_length + spacing * (wall_layers-1) * 2) / spacing))
        bottom_y = np.full_like(bottom_x, -offset)
        boundary_points_x.extend(bottom_x)
        boundary_points_y.extend(bottom_y)

        left_wall_x = np.full(int(box_height / spacing), -offset)
        left_wall_y = np.linspace(spacing, box_height, int(box_height / spacing))
        boundary_points_x.extend(left_wall_x)
        boundary_points_y.extend(left_wall_y)

        right_wall_x = np.full(int(box_height / spacing), box_length + offset)
        right_wall_y = np.linspace(spacing, box_height, int(box_height / spacing))
        boundary_points_x.extend(right_wall_x)
        boundary_points_y.extend(right_wall_y)
    
    # Kombinieren der X- und Y-Koordinaten in einem Array für boundary_points
    boundary_points = np.vstack([boundary_points_x, boundary_points_y]).T

    # inlet_points
    inlet_points_layers = 1
    for i in range(inlet_points_layers):
        offset = i * spacing  # Offset basierend auf der aktuellen Schicht und Punkt Dichte
        box_length_2 = box_length / 2
        inlet_points_x = np.linspace(box_length/4, box_length*3/4, int(box_length_2 / spacing))
        inlet_points_y = np.full_like(inlet_points_x, box_height*3/4 - offset)

        # Kombinieren der X- und Y-Koordinaten in einem Array für inlet_points
        inlet_points = np.vstack([inlet_points_x, inlet_points_y]).T

    return boundary_points, inlet_points