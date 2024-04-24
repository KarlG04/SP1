import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

def visualize_pipe(boundary_points, inlet_points, outlet_points, diameter_particle):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)  # Erstellt ein Subplot für besseres Handling des zooms

    factor = 2000
    # Rohrpunkte
    ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', markersize=diameter_particle * factor, color='#000000', label='Rohrpunkte')
    # Einlasspunkte
    ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=diameter_particle * factor, color='#55ff4a', label='Einlasspunkte')
    # Auslasspunkte
    ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=diameter_particle * factor, color='#ff4747', label='Auslasspunkte')

    ax.axis('equal')
    ax.set_title('Visualisierung des Krümmers mit Punkten')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend()

    # Funktion um auf Mausrad-Events zu reagieren
    def on_scroll(event):
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:  # Falls der Mauszeiger außerhalb der Achsen ist
            return
        base_scale = 1.1
        if event.button == 'up':  # Zoom-in
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # Zoom-out
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Aktualisiere die Achsen-Skalierung
        ax.set_xlim([xdata - (xdata - ax.get_xlim()[0]) * scale_factor,
                     xdata + (ax.get_xlim()[1] - xdata) * scale_factor])
        ax.set_ylim([ydata - (ydata - ax.get_ylim()[0]) * scale_factor,
                     ydata + (ax.get_ylim()[1] - ydata) * scale_factor])
        plt.draw()  # Aktualisiere das Plot-Fenster

    fig.canvas.mpl_connect('scroll_event', on_scroll)  # Verbinde das scroll_event mit der on_scroll Funktion

    plt.show()



def visualize_flow(boundary_points, inlet_points, outlet_points, Fluid_Points, delta_ts, diameter_particle):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    plt.subplots_adjust(bottom=0.15)  # Vergrößern des unteren Randes für Slider

    factor = 2000
    ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', markersize=diameter_particle * factor, color='#000000', label='Boundary Points')
    ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=diameter_particle * factor, color='#55ff4a', label='Inlet Points')
    ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=diameter_particle * factor, color='#ff4747', label='Outlet Points')
    points, = ax.plot(Fluid_Points[0, :, 0], Fluid_Points[1, :, 0], 'o', markersize=diameter_particle * factor, color='#42a7f5', label='Fluid Particles')

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend()

    # Position und Größe des Sliders anpassen, um die gleiche Breite wie das Hauptdiagramm zu haben
    slider_ax = fig.add_axes([ax.get_position().x0, 0.05, ax.get_position().width, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Time Step', 1, Fluid_Points.shape[2], valinit=1, valfmt='%d')

    initial_time = sum(delta_ts[:1])
    time_text = ax.text(0.5, 1.05, f'Time: {initial_time:.6f} s', transform=ax.transAxes, ha='center')

    def update(val):
        time_step = int(slider.val) - 1  # Slider-Wert in Array-Index umwandeln
        points.set_data(Fluid_Points[0, :, time_step], Fluid_Points[1, :, time_step])
        current_time = sum(delta_ts[:time_step+1])  # Zeit-Index bleibt gleich, weil delta_ts bei 0 beginnt
        time_text.set_text(f'Time: {current_time:.6f} s')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_scroll(event):
        # Center zoom around the mouse position
        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None:
            return  # Abbruch, falls Klick außerhalb der Achsen

        scale_factor = 1.1 if event.button == 'up' else 0.9
        ax.set_xlim([x_click - (x_click - ax.get_xlim()[0]) * scale_factor,
                     x_click + (ax.get_xlim()[1] - x_click) * scale_factor])
        ax.set_ylim([y_click - (y_click - ax.get_ylim()[0]) * scale_factor,
                     y_click + (ax.get_ylim()[1] - y_click) * scale_factor])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()



def visualize_flow_animation(boundary_points, inlet_points, outlet_points, Fluid_Points, delta_ts, animation_interval):
    print("visualize flow animation")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', markersize=2, color='#000000', label='Boundary Points')
    ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=2, color='#55ff4a', label='Inlet Points')
    ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=2, color='#ff4747', label='Outlet Points')

    points, = ax.plot(Fluid_Points[0, :, 0], Fluid_Points[1, :, 0], 'o', markersize=2, color='#42a7f5', label='Fluid Particles')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)

    # Initial time display setup
    initial_time = sum(delta_ts[:1])  # Calculate initial time for display
    time_text = ax.text(0.5, 1.05, f'Time: {initial_time:.6f} s', transform=ax.transAxes, ha='center')

    def update(frame):
        points.set_data(Fluid_Points[0, :, frame], Fluid_Points[1, :, frame])
        current_time = sum(delta_ts[:frame+1])
        time_text.set_text(f'Time: {current_time:.6f} s')
        return points,

    # Calculation of the interval using the speed factor
    ani = FuncAnimation(fig, update, frames=len(delta_ts), interval=animation_interval, blit=True)

    plt.show()
