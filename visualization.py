import matplotlib.pyplot as plt

def visualize(pipe_points, inlet_points, outlet_points, fluid_positions):
    # Anpassen der figsize und dpi für eine angemessene Darstellung auf deinem Bildschirm
    plt.figure(figsize=(12, 6), dpi=100)  

    # Rohrpunkte
    plt.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=1, color='#000000', label='Rohrpunkte')
    # Einlasspunkte
    plt.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=1, color='#55ff4a', label='Einlasspunkte')
    # Auslasspunkte
    plt.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=1, color='#ff4747', label='Auslasspunkte')
    # Fluidpunkte
    plt.plot(fluid_positions[:, 0], fluid_positions[:, 1], 'o', markersize=2, color='#4a76ff', label='Fluidpunkte')  # Blau

    plt.axis('equal')
    plt.title('Visualisierung des Krümmers mit Punkten')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid(True)
    plt.legend()

    # Zeige das Plot-Fenster. Die Vollbildfunktion wird hier ausgelassen, um Überzoomen zu vermeiden.
    plt.show()