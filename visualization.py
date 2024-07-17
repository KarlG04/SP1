import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

def visualize_boundary(ax, diameter_particle, box_length, box_height, fluid_length, fluid_height, num_time_steps, delta_t):
    spacing = diameter_particle
    # Berechnen der inlet_points
    inlet_points_x = np.linspace(spacing, fluid_length, int(fluid_length / spacing))
    inlet_points_y = np.linspace(spacing, fluid_height, int(fluid_height / spacing))

    inlet_points_x, inlet_points_y = np.meshgrid(inlet_points_x, inlet_points_y)
    inlet_points = np.vstack([inlet_points_x.ravel(), inlet_points_y.ravel()]).T

    # Hinzufügen der zufälligen Verschiebung
    random_shift = np.random.uniform(0, 0.1, inlet_points.shape)
    inlet_points += random_shift

    # Zeichnen der Begrenzung als dicke schwarze Linien
    ax.plot([0, box_length], [0, 0], color='black', linewidth=10)  # untere Begrenzung
    ax.plot([0, box_length], [box_height, box_height], color='black', linewidth=10)  # obere Begrenzung
    ax.plot([0, 0], [0, box_height], color='black', linewidth=10)  # linke Begrenzung
    ax.plot([box_length, box_length], [0, box_height], color='black', linewidth=10)  # rechte Begrenzung

    # Zeichnen der inlet_points
    inlet_plot, = ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', color='#42a7f5')

    # Berechne die Anzahl der Flüssigkeitspartikel und die simulierte Zeit
    num_fluid_particles = len(inlet_points)
    simulated_time = num_time_steps * delta_t

    # Füge die Überschriften hinzu
    ax.text(0.5, 1.05, f"Simulated time: {simulated_time:.2f} s", transform=ax.transAxes, ha="center", fontsize=16)
    ax.text(0.5, 1.02, f"Number of fluid particles: {num_fluid_particles}", transform=ax.transAxes, ha="center", fontsize=16)

    ax.axis('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend().remove()  # Entferne das Legendenelement

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
        marker_size = diameter_particle * scale_factor * 0.5
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

    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)
    ax.figure.canvas.mpl_connect('resize_event', on_resize)

    update_marker_size()







def visualize_flow(ax, fluid_particles, delta_ts, diameter_particle, smoothing_length, box_length, box_height):
    ax.clear()
    
    # Zeichnen der Begrenzung als schwarze Linien
    ax.plot([0, box_length], [0, 0], color='black')  # untere Begrenzung
    ax.plot([0, box_length], [box_height, box_height], color='black')  # obere Begrenzung
    ax.plot([0, 0], [0, box_height], color='black')  # linke Begrenzung
    ax.plot([box_length, box_length], [0, box_height], color='black')  # rechte Begrenzung

    # Initiales Zeichnen der Fluid-Partikel
    points, = ax.plot(fluid_particles[0][0], fluid_particles[1][0], 'o', color='#42a7f5', label='Fluid Particles')

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
        for label in fluid_labels:
            label.set_visible(labels_visible)
        ax.figure.canvas.draw_idle()

    def toggle_circles(event):
        nonlocal circles_visible
        circles_visible = not circles_visible
        for circle in smoothing_circles:
            circle.set_visible(circles_visible)
        ax.figure.canvas.draw_idle()

    def toggle_velocities(event):
        nonlocal velocities_visible
        velocities_visible = not velocities_visible
        for velocity in velocities:
            velocity.set_visible(velocities_visible)
        ax.figure.canvas.draw_idle()

    label_button_ax = ax.figure.add_axes([0.45, 0.02, 0.1, 0.04])
    label_button = Button(label_button_ax, 'Toggle Labels', color='#ffffff', hovercolor='#f1f1f1')
    label_button.on_clicked(toggle_labels)

    circle_button_ax = ax.figure.add_axes([0.45, 0.07, 0.1, 0.04])
    circle_button = Button(circle_button_ax, 'Toggle Circles', color='#ffffff', hovercolor='#f1f1f1')
    circle_button.on_clicked(toggle_circles)

    velocities_button_ax = ax.figure.add_axes([0.45, 0.12, 0.1, 0.04])
    velocities_button = Button(velocities_button_ax, 'Velocities', color='#ffffff', hovercolor='#f1f1f1')
    velocities_button.on_clicked(toggle_velocities)

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend()

    # Position und Größe des Sliders anpassen, um die gleiche Breite wie das Hauptdiagramm zu haben
    slider_ax = ax.figure.add_axes([ax.get_position().x0, 0.23, ax.get_position().width, 0.03], facecolor='lightgoldenrodyellow')
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
        ax.figure.canvas.draw_idle()

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
        marker_size = diameter_particle * scale_factor * 0.4
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
    next_button_ax = ax.figure.add_axes([0.8, 0.02, 0.1, 0.04])
    next_button = Button(next_button_ax, 'Next', color='#ffffff', hovercolor='#f1f1f1')
    next_button.on_clicked(next_time_step)

    # Button für rückwärts
    prev_button_ax = ax.figure.add_axes([0.125, 0.02, 0.1, 0.04])
    prev_button = Button(prev_button_ax, 'Previous', color='#ffffff', hovercolor='#f1f1f1')
    prev_button.on_clicked(prev_time_step)

    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)
    ax.figure.canvas.mpl_connect('resize_event', on_resize)

    update_marker_size()
    ax.figure.canvas.draw()