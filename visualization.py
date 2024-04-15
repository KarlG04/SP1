import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

def visualize(pipe_points, inlet_points, outlet_points):
    # Anpassen der figsize und dpi für eine angemessene Darstellung auf deinem Bildschirm
    plt.figure(figsize=(14, 6), dpi=100)  

    # Rohrpunkte
    plt.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=2, color='#000000', label='Rohrpunkte')
    # Einlasspunkte
    plt.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=2, color='#55ff4a', label='Einlasspunkte')
    # Auslasspunkte
    plt.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=2, color='#ff4747', label='Auslasspunkte')

    plt.axis('equal')
    plt.title('Visualisierung des Krümmers mit Punkten')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.grid(True)
    plt.legend()

    # Zeige das Plot-Fenster. Die Vollbildfunktion wird hier ausgelassen, um Überzoomen zu vermeiden.
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_flow(pipe_points, inlet_points, outlet_points, Fluid_Properties):
    # Anpassen der figsize und dpi für eine angemessene Darstellung auf deinem Bildschirm
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    plt.subplots_adjust(bottom=0.2)  # Platz für den Slider lassen

    # Rohrpunkte
    ax.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=2, color='#000000', label='Rohrpunkte')
    # Einlasspunkte
    ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=2, color='#55ff4a', label='Einlasspunkte')
    # Auslasspunkte
    ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=2, color='#ff4747', label='Auslasspunkte')

    # Initialzeichnung der Partikelpositionen
    points, = ax.plot(Fluid_Properties[0, :, 0], Fluid_Properties[1, :, 0], 'o', markersize=2, color='#42a7f5', label='Fluid Partikel')

    ax.axis('equal')
    ax.set_title('Visualisierung des Krümmers mit Partikeln')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.legend()

    # Slider
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Zeitschritt', 0, Fluid_Properties.shape[2] - 1, valinit=0, valfmt='%d')

    # Update-Funktion für den Slider
    def update(val):
        time_step = int(slider.val)
        points.set_data(Fluid_Properties[0, :, time_step], Fluid_Properties[1, :, time_step])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()



