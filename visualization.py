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
    print("visualize flow")
    # Adjust figsize and dpi for appropriate display on your screen
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    # Plot pipe points
    ax.plot(pipe_points[:, 0], pipe_points[:, 1], 'o', markersize=2, color='#000000', label='Pipe Points')
    # Plot inlet points
    ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', markersize=2, color='#55ff4a', label='Inlet Points')
    # Plot outlet points
    ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', markersize=2, color='#ff4747', label='Outlet Points')

    # Initial drawing of particle positions
    points, = ax.plot(Fluid_Points[0, :, 0], Fluid_Points[1, :, 0], 'o', markersize=2, color='#42a7f5', label='Fluid Particles')

    ax.axis('equal')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.grid(True)
    ax.legend()

    # Slider
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Time Step', 0, Fluid_Points.shape[2] - 1, valinit=0, valfmt='%d')

    # Current time display in the title
    initial_time = sum(delta_ts[:1])  # Calculate initial time for display
    time_text = ax.text(0.5, 1.05, f'Time: {initial_time:.6f} s', transform=ax.transAxes, ha='center')

    # Update function for the slider
    def update(val):
        time_step = int(slider.val)
        points.set_data(Fluid_Points[0, :, time_step], Fluid_Points[1, :, time_step])
        current_time = sum(delta_ts[:time_step+1])
        time_text.set_text(f'Time: {current_time:.6f} s')
        fig.canvas.draw_idle()

    slider.on_changed(update)

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
