import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

def visualize_boundary_outlet(boundary_points, inlet_points, outlet_points, diameter_particle):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    # Initiales Zeichnen der Punkte
    boundary_plot, = ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', color='#000000', label='Rohrpunkte')
    inlet_plot, = ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', color='#55ff4a', label='Einlasspunkte')
    outlet_plot, = ax.plot(outlet_points[:, 0], outlet_points[:, 1], 'o', color='#ff4747', label='Auslasspunkte')

    ax.axis('equal')
    title = f'Visualisierung der festen Strukturen\nPartikeldurchmesser: {diameter_particle*1e6}µm'
    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend()

    def update_marker_size():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Berechnung des Skalierungsfaktors basierend auf dem kleineren Achsenbereich und der Fenstergröße
        x_scale = ax_width / x_range
        y_scale = ax_height / y_range
        scale_factor = min(x_scale, y_scale)

        # Berechnung der Markergröße in Punkten
        marker_size = diameter_particle * scale_factor * 0.416
        boundary_plot.set_markersize(marker_size)
        inlet_plot.set_markersize(marker_size)
        outlet_plot.set_markersize(marker_size)
        plt.draw()

    def on_scroll(event):
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        ax.set_xlim([xdata - (xdata - ax.get_xlim()[0]) * scale_factor,
                     xdata + (ax.get_xlim()[1] - xdata) * scale_factor])
        ax.set_ylim([ydata - (ydata - ax.get_ylim()[0]) * scale_factor,
                     ydata + (ax.get_ylim()[1] - ydata) * scale_factor])

        update_marker_size()

    def on_resize(event):
        update_marker_size()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('resize_event', on_resize)

    update_marker_size()

    plt.show()

def visualize_boundary(boundary_points, inlet_points, diameter_particle, boundary_description):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    # Initiales Zeichnen der Punkte
    boundary_plot, = ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', color='#000000', label='Rohrpunkte')
    inlet_plot, = ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', color='#55ff4a', label='Einlasspunkte')

    # Anzeigen der Boundary-Beschreibungen
    for (x, y), desc in zip(boundary_points, boundary_description):
        ax.text(x, y, desc, fontsize=8, ha='center', va='center', color='red')

    ax.axis('equal')
    title = f'Visualisierung der festen Strukturen\nPartikeldurchmesser: {diameter_particle*1e6}µm'
    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend()

    def update_marker_size():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Berechnung des Skalierungsfaktors basierend auf dem kleineren Achsenbereich und der Fenstergröße
        x_scale = ax_width / x_range
        y_scale = ax_height / y_range
        scale_factor = min(x_scale, y_scale)

        # Berechnung der Markergröße in Punkten
        marker_size = diameter_particle * scale_factor * 0.416
        boundary_plot.set_markersize(marker_size)
        inlet_plot.set_markersize(marker_size)
        plt.draw()

    def on_scroll(event):
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        base_scale = 1.1
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        ax.set_xlim([xdata - (xdata - ax.get_xlim()[0]) * scale_factor,
                     xdata + (ax.get_xlim()[1] - xdata) * scale_factor])
        ax.set_ylim([ydata - (ydata - ax.get_ylim()[0]) * scale_factor,
                     ydata + (ax.get_ylim()[1] - ydata) * scale_factor])

        update_marker_size()

    def on_resize(event):
        update_marker_size()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('resize_event', on_resize)

    update_marker_size()

    plt.show()





def visualize_flow(boundary_points, inlet_points, fluid_particles, delta_ts, diameter_particle, smoothing_length):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    plt.subplots_adjust(bottom=0.35)  # Vergrößern des unteren Randes für Slider und Buttons

    # Initiales Zeichnen der Punkte
    boundary_plot, = ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', color='#000000', label='Boundary Points')
    points, = ax.plot(fluid_particles[0][0], fluid_particles[1][0], 'o', color='#42a7f5', label='Fluid Particles')

    # Nummern der Boundary-Punkte anzeigen, beginnend bei 0
    boundary_labels = [ax.text(x, y, str(i), fontsize=7, color='red', visible=False) for i, (x, y) in enumerate(boundary_points)]
    # Nummern der Fluid-Punkte anzeigen, beginnend bei 0
    fluid_labels = [ax.text(x, y, str(i), fontsize=7, color='green', visible=False) for i, (x, y) in enumerate(zip(fluid_particles[0][0], fluid_particles[1][0]))]

    # Glättungslänge als Kreise
    fluid_circles = [plt.Circle((x, y), smoothing_length, color='lightgray', fill=False, linestyle='-', linewidth=0.5, visible=False) for x, y in zip(fluid_particles[0][0], fluid_particles[1][0])]
    smoothing_circles = fluid_circles

    for circle in smoothing_circles:
        ax.add_artist(circle)

    labels_visible = False
    circles_visible = False
    velocities_visible = False
    mirror_visible = False
    velocities_vector_factor = 0.1

    # Initiales Zeichnen der Geschwindigkeitsvektoren (unsichtbar)
    velocities = [ax.annotate('', xy=(fluid_particles[0][0][i] + velocities_vector_factor * fluid_particles[2][0][i], 
                                       fluid_particles[1][0][i] + velocities_vector_factor * fluid_particles[3][0][i]), 
                               xytext=(fluid_particles[0][0][i], fluid_particles[1][0][i]),
                               arrowprops=dict(facecolor='red', edgecolor='red', width=0.5, headwidth=3),
                               visible=False) for i in range(len(fluid_particles[0][0]))]

    def toggle_labels(event):
        nonlocal labels_visible
        labels_visible = not labels_visible
        for label in boundary_labels + fluid_labels:
            label.set_visible(labels_visible)
        fig.canvas.draw_idle()

    def toggle_circles(event):
        nonlocal circles_visible
        circles_visible = not circles_visible
        for circle in smoothing_circles:
            circle.set_visible(circles_visible)
        fig.canvas.draw_idle()

    def toggle_velocities(event):
        nonlocal velocities_visible
        velocities_visible = not velocities_visible
        for velocity in velocities:
            velocity.set_visible(velocities_visible)
        fig.canvas.draw_idle()

    label_button_ax = fig.add_axes([0.45, 0.02, 0.1, 0.04])
    label_button = Button(label_button_ax, 'Toggle Labels', color='#ffffff', hovercolor='#f1f1f1')
    label_button.on_clicked(toggle_labels)

    circle_button_ax = fig.add_axes([0.45, 0.07, 0.1, 0.04])
    circle_button = Button(circle_button_ax, 'Toggle Circles', color='#ffffff', hovercolor='#f1f1f1')
    circle_button.on_clicked(toggle_circles)

    velocities_button_ax = fig.add_axes([0.45, 0.12, 0.1, 0.04])
    velocities_button = Button(velocities_button_ax, 'Velocities', color='#ffffff', hovercolor='#f1f1f1')
    velocities_button.on_clicked(toggle_velocities)

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend()

    # Position und Größe des Sliders anpassen, um die gleiche Breite wie das Hauptdiagramm zu haben
    slider_ax = fig.add_axes([ax.get_position().x0, 0.23, ax.get_position().width, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Time Step', 1, len(fluid_particles[0]), valinit=1, valfmt='%d')

    initial_time = sum(delta_ts[:1])
    time_text = ax.text(0.5, 1.04, f'Time: {initial_time:.6f} s', transform=ax.transAxes, ha='center')

    def update(val):
        time_step = int(slider.val) - 1  # Slider-Wert in Array-Index umwandeln
        points.set_data(fluid_particles[0][time_step], fluid_particles[1][time_step])

        # Aktualisieren der Fluid Labels
        for label, (x, y) in zip(fluid_labels, zip(fluid_particles[0][time_step], fluid_particles[1][time_step])):
            label.set_position((x, y))

        # Aktualisieren der Fluid Kreise
        for circle, (x, y) in zip(fluid_circles, zip(fluid_particles[0][time_step], fluid_particles[1][time_step])):
            circle.center = (x, y)

        # Aktualisieren der Geschwindigkeitsvektoren
        for velocity in velocities:
            velocity.remove()
        velocities[:] = [ax.annotate('', xy=(fluid_particles[0][time_step][i] + velocities_vector_factor * fluid_particles[2][time_step][i], 
                                             fluid_particles[1][time_step][i] + velocities_vector_factor * fluid_particles[3][time_step][i]), 
                                   xytext=(fluid_particles[0][time_step][i], fluid_particles[1][time_step][i]),
                                   arrowprops=dict(facecolor='red', edgecolor='red', width=0.5, headwidth=3),
                                   visible=velocities_visible, zorder=5) for i in range(len(fluid_particles[0][time_step]))]

        current_time = sum(delta_ts[:time_step + 1])  # Zeit-Index bleibt gleich, weil delta_ts bei 0 beginnt
        time_text.set_text(f'Time: {current_time:.6f} s')
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def update_marker_size():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Berechnung des Skalierungsfaktors basierend auf dem kleineren Achsenbereich und der Fenstergröße
        x_scale = ax_width / x_range
        y_scale = ax_height / y_range
        scale_factor = min(x_scale, y_scale)

        # Berechnung der Markergröße in Punkten
        marker_size = diameter_particle * scale_factor * 0.416
        boundary_plot.set_markersize(marker_size)
        points.set_markersize(marker_size)
        plt.draw()

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

        update_marker_size()

    def on_resize(event):
        update_marker_size()

    def next_time_step(event):
        current_val = slider.val
        if current_val < slider.valmax:
            slider.set_val(current_val + 1)

    def prev_time_step(event):
        current_val = slider.val
        if current_val > slider.valmin:
            slider.set_val(current_val - 1)

    # Button für vorwärts
    next_button_ax = fig.add_axes([0.8, 0.02, 0.1, 0.04])
    next_button = Button(next_button_ax, 'Next', color='#ffffff', hovercolor='#f1f1f1')
    next_button.on_clicked(next_time_step)

    # Button für rückwärts
    prev_button_ax = fig.add_axes([0.125, 0.02, 0.1, 0.04])
    prev_button = Button(prev_button_ax, 'Previous', color='#ffffff', hovercolor='#f1f1f1')
    prev_button.on_clicked(prev_time_step)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('resize_event', on_resize)

    update_marker_size()
    plt.show()