# main.py
# Importiere benötigte Module
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import solver
import visualization

def main():
    root = tk.Tk()
    app = DambreakApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Handle window close event
    root.mainloop()

class DambreakApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dambreak Simulation")
        
        # Set up GUI layout
        self.setup_ui()
        
    def setup_ui(self):
        # Create frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(self.root, padding="10")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add control elements
        self.setup_controls()
        
        # Add plot
        self.setup_plot()
        
    def setup_controls(self):
        label_font = ("Helvetica", 14)
        entry_font = ("Helvetica", 14)
        button_font = ("Helvetica", 14, "bold")
        
        ttk.Label(self.control_frame, text="Simulationsparameter", font=label_font).pack(anchor=tk.W)
        
        # Add entries for parameters
        self.fluid_height = tk.DoubleVar(value=130.0)
        ttk.Label(self.control_frame, text="Fluid Height", font=label_font).pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.fluid_height, font=entry_font).pack(anchor=tk.W)
        
        self.fluid_length = tk.DoubleVar(value=50.0)
        ttk.Label(self.control_frame, text="Fluid Length", font=label_font).pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.fluid_length, font=entry_font).pack(anchor=tk.W)
        
        self.box_height = tk.DoubleVar(value=200.0)
        ttk.Label(self.control_frame, text="Box Height", font=label_font).pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.box_height, font=entry_font).pack(anchor=tk.W)
        
        self.box_length = tk.DoubleVar(value=200.0)
        ttk.Label(self.control_frame, text="Box Length", font=label_font).pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.box_length, font=entry_font).pack(anchor=tk.W)
        
        self.particle_diameter = tk.DoubleVar(value=5.0)
        ttk.Label(self.control_frame, text="Particle Diameter", font=label_font).pack(anchor=tk.W)
        ttk.Entry(self.control_frame, textvariable=self.particle_diameter, font=entry_font).pack(anchor=tk.W)
        
        # Add start button
        ttk.Button(self.control_frame, text="Start Simulation", command=self.start_simulation, font=button_font).pack(anchor=tk.W, pady=20)
        
    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.visualize_boundary()
        
    def visualize_boundary(self):
        # Initial visualization of the boundary
        self.ax.clear()
        visualization.visualize_boundary(self.ax, 
                                         self.particle_diameter.get(), 
                                         self.box_length.get(), 
                                         self.box_height.get(), 
                                         self.fluid_length.get(), 
                                         self.fluid_height.get())
        self.canvas.draw()
        
    def start_simulation(self):
        # Gather parameters
        gravity = (0.0, -0.0981)
        initial_density = 1
        num_time_steps = 2000
        spacing = self.particle_diameter.get()
        smoothing_length = 5
        isentropic_exponent = 20
        delta_t = 0.01
        boundary_damping = -0.6
        density_factor = (315 * 1) / (64 * np.pi * smoothing_length**9)
        pressure_factor = (-(65 * 1) / (np.pi * smoothing_length**6))
        viscosity_factor = (45 * 0.5 * 1) / (np.pi * smoothing_length**6)
        
        fluid_height = self.fluid_height.get()
        fluid_length = self.fluid_length.get()
        box_height = self.box_height.get()
        box_length = self.box_length.get()
        
        # Run the simulation with the specified parameters
        fluid_particles, delta_ts = solver.run_simulation(gravity, initial_density, num_time_steps, spacing, smoothing_length, isentropic_exponent, delta_t, box_length, box_height, fluid_length, fluid_height, boundary_damping, density_factor, pressure_factor, viscosity_factor)
        
        # Update the plot with simulation results
        self.update_plot(fluid_particles, delta_ts)
        
    def update_plot(self, fluid_particles, delta_ts):
        self.ax.clear()
        visualization.visualize_flow(self.ax, fluid_particles, delta_ts, self.particle_diameter.get(), 5, self.box_length.get(), self.box_height.get())
        self.canvas.draw()

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    main()