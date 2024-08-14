import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

def visualize_boundary(ax, diameter_particle, box_width, box_height, fluid_width, fluid_height, num_time_steps, delta_t):
    spacing = diameter_particle
    # Calculate the inlet_points
    inlet_points_x = np.linspace(spacing, fluid_width, int(fluid_width / spacing))
    inlet_points_y = np.linspace(spacing, fluid_height, int(fluid_height / spacing))

    inlet_points_x, inlet_points_y = np.meshgrid(inlet_points_x, inlet_points_y)
    inlet_points = np.vstack([inlet_points_x.ravel(), inlet_points_y.ravel()]).T

    # Adding the random shift
    random_shift = np.random.uniform(0, 0.01 * spacing, inlet_points.shape)
    inlet_points += random_shift

    # Drawing the boundary as thick black lines
    ax.plot([0, box_width], [0, 0], color='black', linewidth=8)  # lower boundary
    ax.plot([0, box_width], [box_height, box_height], color='black', linewidth=8)  # upper boundary
    ax.plot([0, 0], [0, box_height], color='black', linewidth=8)  # left boundary
    ax.plot([box_width, box_width], [0, box_height], color='black', linewidth=8)  # right boundary

    # Drawing the inlet_points
    inlet_plot, = ax.plot(inlet_points[:, 0], inlet_points[:, 1], 'o', color='#42a7f5')

    # Calculate the number of fluid particles and the simulated time
    num_fluid_particles = len(inlet_points)
    simulated_time = num_time_steps * delta_t

    # Adding the titles
    ax.text(0.5, 1.05, f"Simulated time: {simulated_time:.2f} s", transform=ax.transAxes, ha="center", fontsize=16)
    ax.text(0.5, 1.02, f"Number of fluid particles: {num_fluid_particles}", transform=ax.transAxes, ha="center", fontsize=16)

    ax.axis('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend().remove()  # Remove the legend

    def update_marker_size():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # Calculation of the scaling factor based on the smaller axis range and the window size
        x_scale = ax_width / x_range
        y_scale = ax_height / y_range
        scale_factor = min(x_scale, y_scale)

        # Calculation of the marker size in points
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

def visualize_flow(fluid_particles, delta_ts, diameter_particle, smoothing_length, box_width, box_height, delta_t):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    plt.subplots_adjust(bottom=0.3)  # Vergrößern des unteren Randes für Slider und Buttons

    # Drawing the boundary as thick black lines
    ax.plot([0, box_width], [0, 0], color='black', linewidth=8)  # lower boundary
    ax.plot([0, box_width], [box_height, box_height], color='black', linewidth=8)  # upper boundary
    ax.plot([0, 0], [0, box_height], color='black', linewidth=8)  # left boundary
    ax.plot([box_width, box_width], [0, box_height], color='black', linewidth=8)  # right boundary

    # Initiales Zeichnen der Fluid-Partikel
    points = ax.scatter(fluid_particles[0][0], fluid_particles[1][0], c='#42a7f5', picker=True)

    # Nummern der Fluid-Punkte anzeigen, beginnend bei 0
    fluid_labels = [ax.text(x, y, str(i), fontsize=9, color='black', ha='center', va='center', visible=False) for i, (x, y) in enumerate(zip(fluid_particles[0][0], fluid_particles[1][0]))]

    # Glättungslänge als Kreise
    fluid_circles = [plt.Circle((x, y), smoothing_length, color='lightgray', fill=False, linestyle='-', linewidth=0.5, visible=False) for x, y in zip(fluid_particles[0][0], fluid_particles[1][0])]
    smoothing_circles = fluid_circles

    for circle in smoothing_circles:
        ax.add_artist(circle)

    labels_visible = False
    circles_visible = False
    velocities_visible = False
    density_colored = False
    colorbar_visible = False
    velocities_colored = False
    velocities_vector_factor = 0.1

    # Berechnung der globalen Min- und Max-Werte der Dichte und Geschwindigkeit über alle Zeitschritte
    global_min_density = np.min([np.min(densities) for densities in fluid_particles[4]])
    global_max_density = np.max([np.max(densities) for densities in fluid_particles[4]])
    global_min_speed = np.min([np.min(np.sqrt(np.array(fluid_particles[2][t])**2 + np.array(fluid_particles[3][t])**2)) for t in range(len(fluid_particles[0]))])
    global_max_speed = np.max([np.max(np.sqrt(np.array(fluid_particles[2][t])**2 + np.array(fluid_particles[3][t])**2)) for t in range(len(fluid_particles[0]))])
    norm_density = plt.Normalize(global_min_density, global_max_density)
    norm_speed = plt.Normalize(global_min_speed, global_max_speed)
    cmap = cm.jet
    mappable_density = cm.ScalarMappable(norm=norm_density, cmap=cmap)
    mappable_speed = cm.ScalarMappable(norm=norm_speed, cmap=cmap)

    # Initiales Zeichnen der Geschwindigkeitsvektoren (unsichtbar)
    fixed_vector_length = 0.1  # Definiere eine feste Länge für die Geschwindigkeitsvektoren

    velocities = [ax.annotate('', 
                            xy=(fluid_particles[0][0][i] + fixed_vector_length * (fluid_particles[2][0][i] / np.sqrt(fluid_particles[2][0][i]**2 + fluid_particles[3][0][i]**2)), 
                                fluid_particles[1][0][i] + fixed_vector_length * (fluid_particles[3][0][i] / np.sqrt(fluid_particles[2][0][i]**2 + fluid_particles[3][0][i]**2))), 
                            xytext=(fluid_particles[0][0][i], fluid_particles[1][0][i]),
                            arrowprops=dict(facecolor='black', edgecolor='black', width=0.5, headwidth=3),
                            visible=False) for i in range(len(fluid_particles[0][0]))]

    colorbar = None

    def toggle_labels(event):
        nonlocal labels_visible
        labels_visible = not labels_visible
        for label in fluid_labels:
            label.set_visible(labels_visible)
        fig.canvas.draw_idle()

    def toggle_velocities(event):
        nonlocal velocities_visible, velocities_colored, colorbar_visible, colorbar
        velocities_visible = not velocities_visible
        velocities_colored = not velocities_colored
        time_step = int(slider.val) - 1
        if velocities_colored:
            speeds = np.sqrt(np.array(fluid_particles[2][time_step])**2 + np.array(fluid_particles[3][time_step])**2)
            colors = cmap(norm_speed(speeds))
            points.set_facecolor(colors)
            if not colorbar_visible:
                cax = fig.add_axes([0.75, 0.35, 0.02, 0.5])
                colorbar = fig.colorbar(mappable_speed, cax=cax, orientation='vertical')
                colorbar.set_label('Speed (m/s)')
                colorbar.set_ticks(np.linspace(global_min_speed, global_max_speed, 5))
                colorbar_visible = True
        else:
            points.set_facecolor('#42a7f5')
            if colorbar_visible and colorbar is not None:
                colorbar.remove()
                colorbar_visible = False

        for velocity in velocities:
            velocity.set_visible(velocities_visible)
        fig.canvas.draw_idle()

    def toggle_density_coloring(event):
        nonlocal density_colored, colorbar_visible, colorbar
        density_colored = not density_colored
        time_step = int(slider.val) - 1
        if density_colored:
            densities = fluid_particles[4][time_step]
            colors = cmap(norm_density(densities))
            points.set_facecolor(colors)
            if not colorbar_visible:
                cax = fig.add_axes([0.75, 0.35, 0.02, 0.5])
                colorbar = fig.colorbar(mappable_density, cax=cax, orientation='vertical')
                colorbar.set_label('Density (kg/m^3)')
                colorbar.set_ticks(np.linspace(global_min_density, global_max_density, 5))
                colorbar_visible = True
        else:
            points.set_facecolor('#42a7f5')
            if colorbar_visible and colorbar is not None:
                colorbar.remove()
                colorbar_visible = False
        fig.canvas.draw_idle()

    ax.set_aspect('equal')
    ax.set_xlabel('X [m]', fontsize=14)
    ax.set_ylabel('Y [m]', fontsize=14)
    ax.grid(True)

    # Position und Größe des Sliders anpassen
    slider_ax = fig.add_axes([0.125, 0.2, 0.775, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(slider_ax, 'Time Step', 1, len(fluid_particles[0]), valinit=1, valfmt='%d')
    slider.label.set_fontsize(12)

    initial_time = sum(delta_ts[:1])
    time_text = ax.text(0.5, 1.04, f'Time: {initial_time:.3f} s', transform=ax.transAxes, ha='center', fontsize=16)

    selected_particle = None
    particle_info_text = []

    def calculate_max_values():
        max_speed = 0
        max_speed_info = (0, 0)
        max_pressure = 0
        max_pressure_info = (0, 0)
        max_density = 0
        max_density_info = (0, 0)

        for t in range(len(fluid_particles[0])):
            for i in range(len(fluid_particles[0][t])):
                speed = (fluid_particles[2][t][i]**2 + fluid_particles[3][t][i]**2)**0.5
                if speed > max_speed:
                    max_speed = speed
                    max_speed_info = (i, t+1)
                if fluid_particles[5][t][i] < max_pressure:
                    max_pressure = fluid_particles[5][t][i]
                    max_pressure_info = (i, t+1)
                if fluid_particles[4][t][i] > max_density:
                    max_density = fluid_particles[4][t][i]
                    max_density_info = (i, t+1)

        return max_speed, max_speed_info, max_pressure, max_pressure_info, max_density, max_density_info

    max_speed, max_speed_info, max_pressure, max_pressure_info, max_density, max_density_info = calculate_max_values()

    ax.text(-0.9, 1.02, 'Max Values', transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.text(-0.9, 0.98, f'Max Speed: {max_speed:.6f} m/s (Particle {max_speed_info[0]}, Step {max_speed_info[1]})', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(-0.9, 0.94, f'Min Pressure: {max_pressure:.6f} Pa (Particle {max_pressure_info[0]}, Step {max_pressure_info[1]})', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(-0.9, 0.90, f'Max Density: {max_density:.6f} kg/m^3 (Particle {max_density_info[0]}, Step {max_density_info[1]})', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.text(-0.9, 0.75, 'Particle Informations (click Particle)', transform=ax.transAxes, fontsize=14, verticalalignment='top')

    def update_particle_info(particle_idx, time_step):
        nonlocal particle_info_text
        for text in particle_info_text:
            text.remove()
        particle_info_text = []

        if particle_idx is not None:
            particle_info_text.append(ax.text(-0.9, 0.70, f'Particle: {particle_idx}', transform=ax.transAxes, fontsize=12, verticalalignment='top'))
            particle_info_text.append(ax.text(-0.9, 0.65, f'X: {fluid_particles[0][time_step][particle_idx]:.6f} m', transform=ax.transAxes, fontsize=12, verticalalignment='top'))
            particle_info_text.append(ax.text(-0.9, 0.60, f'Y: {fluid_particles[1][time_step][particle_idx]:.6f} m', transform=ax.transAxes, fontsize=12, verticalalignment='top'))
            speed = (fluid_particles[2][time_step][particle_idx]**2 + fluid_particles[3][time_step][particle_idx]**2)**0.5
            particle_info_text.append(ax.text(-0.9, 0.55, f'Speed: {speed:.6f} m/s', transform=ax.transAxes, fontsize=12, verticalalignment='top'))
            particle_info_text.append(ax.text(-0.9, 0.50, f'Density: {fluid_particles[4][time_step][particle_idx]:.6f} kg/m^3', transform=ax.transAxes, fontsize=12, verticalalignment='top'))
            particle_info_text.append(ax.text(-0.9, 0.45, f'Pressure: {fluid_particles[5][time_step][particle_idx]:.6f} Pa', transform=ax.transAxes, fontsize=12, verticalalignment='top'))

        fig.canvas.draw_idle()

    def update(val):
        time_step = int(slider.val) - 1  # Slider-Wert in Array-Index umwandeln
        points.set_offsets(np.c_[fluid_particles[0][time_step], fluid_particles[1][time_step]])

        # Aktualisieren der Fluid Labels
        for label, (x, y) in zip(fluid_labels, zip(fluid_particles[0][time_step], fluid_particles[1][time_step])):
            label.set_position((x, y))

        # Aktualisieren der Fluid Kreise
        for circle, (x, y) in zip(fluid_circles, zip(fluid_particles[0][time_step], fluid_particles[1][time_step])):
            circle.center = (x, y)

        # Aktualisieren der Geschwindigkeitsvektoren
        for velocity in velocities:
            velocity.remove()

        velocities[:] = [ax.annotate('', 
                                    xy=(fluid_particles[0][time_step][i] + fixed_vector_length * (fluid_particles[2][time_step][i] / np.sqrt(fluid_particles[2][time_step][i]**2 + fluid_particles[3][time_step][i]**2)), 
                                        fluid_particles[1][time_step][i] + fixed_vector_length * (fluid_particles[3][time_step][i] / np.sqrt(fluid_particles[2][time_step][i]**2 + fluid_particles[3][time_step][i]**2))), 
                                    xytext=(fluid_particles[0][time_step][i], fluid_particles[1][time_step][i]),
                                    arrowprops=dict(facecolor='black', edgecolor='black', width=0.5, headwidth=3),
                                    visible=velocities_visible, zorder=5) for i in range(len(fluid_particles[0][time_step]))]

        # Umwandlung der Geschwindigkeitskomponenten in NumPy-Arrays für die Berechnung der Geschwindigkeit
        speeds = np.sqrt(np.array(fluid_particles[2][time_step])**2 + np.array(fluid_particles[3][time_step])**2)

        if density_colored:
            densities = fluid_particles[4][time_step]
            colors = cmap(norm_density(densities))
            points.set_facecolor(colors)
        elif velocities_colored:
            colors = cmap(norm_speed(speeds))
            points.set_facecolor(colors)
        else:
            points.set_facecolor('#42a7f5')

        if selected_particle is not None:
            update_particle_info(selected_particle, time_step)

        # Aktualisieren der Zeit
        current_time = sum(delta_ts[:time_step + 1])
        time_text.set_text(f'Time: {current_time:.3f} s')

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
        marker_size = diameter_particle * scale_factor * 10
        points.set_sizes([marker_size] * len(fluid_particles[0][0]))
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

    def on_pick(event):
        nonlocal selected_particle
        if event.artist != points:
            return
        mouse_event = event.mouseevent
        x_mouse, y_mouse = mouse_event.xdata, mouse_event.ydata
        if x_mouse is None or y_mouse is None:
            return  # Abbruch, falls Klick außerhalb der Achsen

        distances = [(i, (x - x_mouse)**2 + (y - y_mouse)**2) for i, (x, y) in enumerate(zip(fluid_particles[0][int(slider.val) - 1], fluid_particles[1][int(slider.val) - 1]))]
        selected_particle = min(distances, key=lambda x: x[1])[0]
        update_particle_info(selected_particle, int(slider.val) - 1)

    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('resize_event', on_resize)

    # Play/Pause Button (Funktion nur zum Umschalten des Labels)
    def toggle_play_pause(event):
        if play_button.label.get_text() == 'Play':
            play_button.label.set_text('Pause')
        else:
            play_button.label.set_text('Play')
        fig.canvas.draw_idle()

    # Play/Pause Button
    play_button_ax = fig.add_axes([0.4625, 0.125, 0.1, 0.06])
    play_button = Button(play_button_ax, 'Play', color='#ffffff', hovercolor='#f1f1f1')
    play_button.label.set_fontsize(12)
    play_button.on_clicked(toggle_play_pause)

    # Button für vorwärts
    next_button_ax = fig.add_axes([0.8, 0.125, 0.1, 0.06])
    next_button = Button(next_button_ax, 'Next', color='#ffffff', hovercolor='#f1f1f1')
    next_button.label.set_fontsize(12)
    next_button.on_clicked(next_time_step)

    # Button für rückwärts
    prev_button_ax = fig.add_axes([0.125, 0.125, 0.1, 0.06])
    prev_button = Button(prev_button_ax, 'Previous', color='#ffffff', hovercolor='#f1f1f1')
    prev_button.label.set_fontsize(12)
    prev_button.on_clicked(prev_time_step)

    # Zusätzliche Buttons unter den Vor- und Zurück-Buttons
    label_button_ax = fig.add_axes([0.125, 0.02, 0.1, 0.06])
    label_button = Button(label_button_ax, 'Toggle Labels', color='#ffffff', hovercolor='#f1f1f1')
    label_button.label.set_fontsize(12)
    label_button.on_clicked(toggle_labels)

    # Neuen Button für Densities hinzufügen
    density_button_ax = fig.add_axes([0.235, 0.02, 0.1, 0.06])
    density_button = Button(density_button_ax, 'Densities', color='#ffffff', hovercolor='#f1f1f1')
    density_button.label.set_fontsize(12)
    density_button.on_clicked(toggle_density_coloring)

    # Neuen Button für Velocities hinzufügen
    velocities_button_ax = fig.add_axes([0.345, 0.02, 0.1, 0.06])
    velocities_button = Button(velocities_button_ax, 'Velocities', color='#ffffff', hovercolor='#f1f1f1')
    velocities_button.label.set_fontsize(12)
    velocities_button.on_clicked(toggle_velocities)

    update_marker_size()
    plt.show()