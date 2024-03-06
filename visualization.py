import matplotlib.pyplot as plt

def visualize_pipe_points(pipe_points, inlet_points, outlet_points):
    plt.figure(figsize=(10, 6))
    # Zeichne die pipe_points in Dunkelgrau / Antrazit
    plt.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=0.5, color='#000000', label='Rohrpunkte')  # Dunkelgrau / Antrazit
    # Zeichne die inlet_points in hellem Grün
    plt.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=0.5, color='#55ff4a', label='Einlasspunkte')  # Helles Grün
    # Zeichne die outlet_points in hellem Rot
    plt.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=0.5, color='#ff4747', label='Auslasspunkte')  # Helles Rot
    
    plt.axis('equal')
    plt.title('Visualisierung des Krümmers mit Punkten')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid(True)
    plt.legend()
    plt.show()
