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

def visualize_flow(pipe_points, inlet_points, outlet_points, Fluid_Points, delta_ts):
    print("visualize flow ...")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    plt.subplots_adjust(bottom=0.25)  # Vergrößern des unteren Randes für Slider

    ax.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=2, color='#000000', label='Pipe Points')
    ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=2, color='#55ff4a', label='Inlet Points')
    ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=2, color='#ff4747', label='Outlet Points')
    points, = ax.plot(Fluid_Points[0, :, 0], Fluid_Points[1, :, 0], 'o', markersize=2, color='#42a7f5', label='Fluid Particles')

    ax.set_aspect('equal')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.legend()

    # Position und Größe des Sliders anpassen, um die gleiche Breite wie das Hauptdiagramm zu haben
    slider_ax = fig.add_axes([ax.get_position().x0, 0.05, ax.get_position().width, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Time Step', 0, Fluid_Points.shape[2] - 1, valinit=0, valfmt='%d')

    initial_time = sum(delta_ts[:1])
    time_text = ax.text(0.5, 1.05, f'Time: {initial_time:.6f} s', transform=ax.transAxes, ha='center')

    def update(val):
        time_step = int(slider.val)
        points.set_data(Fluid_Points[0, :, time_step], Fluid_Points[1, :, time_step])
        current_time = sum(delta_ts[:time_step+1])
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

def visualize_flow_animation(pipe_points, inlet_points, outlet_points, Fluid_Points, delta_ts, animation_interval):
    print("visualize flow animation")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=2, color='#000000', label='Pipe Points')
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
