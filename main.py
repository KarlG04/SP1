# main.py
# Importiere benötigte Module
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import solver
import visualization

def main():
    root = tk.Tk()
    root.state("zoomed")  # Setze das Fenster auf Vollbild
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
        self.fluid_height_prev = self.fluid_height.get()
        ttk.Label(self.control_frame, text="Fluid Height", font=label_font).pack(anchor=tk.W)
        fluid_height_entry = ttk.Entry(self.control_frame, textvariable=self.fluid_height, font=entry_font)
        fluid_height_entry.pack(anchor=tk.W)
        fluid_height_entry.bind("<FocusOut>", self.on_parameter_change)
        fluid_height_entry.bind("<Return>", self.on_parameter_change)
        
        self.fluid_length = tk.DoubleVar(value=50.0)
        self.fluid_length_prev = self.fluid_length.get()
        ttk.Label(self.control_frame, text="Fluid Length", font=label_font).pack(anchor=tk.W)
        fluid_length_entry = ttk.Entry(self.control_frame, textvariable=self.fluid_length, font=entry_font)
        fluid_length_entry.pack(anchor=tk.W)
        fluid_length_entry.bind("<FocusOut>", self.on_parameter_change)
        fluid_length_entry.bind("<Return>", self.on_parameter_change)
        
        self.box_height = tk.DoubleVar(value=200.0)
        self.box_height_prev = self.box_height.get()
        ttk.Label(self.control_frame, text="Box Height", font=label_font).pack(anchor=tk.W)
        box_height_entry = ttk.Entry(self.control_frame, textvariable=self.box_height, font=entry_font)
        box_height_entry.pack(anchor=tk.W)
        box_height_entry.bind("<FocusOut>", self.on_parameter_change)
        box_height_entry.bind("<Return>", self.on_parameter_change)
        
        self.box_length = tk.DoubleVar(value=200.0)
        self.box_length_prev = self.box_length.get()
        ttk.Label(self.control_frame, text="Box Length", font=label_font).pack(anchor=tk.W)
        box_length_entry = ttk.Entry(self.control_frame, textvariable=self.box_length, font=entry_font)
        box_length_entry.pack(anchor=tk.W)
        box_length_entry.bind("<FocusOut>", self.on_parameter_change)
        box_length_entry.bind("<Return>", self.on_parameter_change)
        
        self.particle_diameter = tk.DoubleVar(value=5.0)
        self.particle_diameter_prev = self.particle_diameter.get()
        ttk.Label(self.control_frame, text="Particle Diameter", font=label_font).pack(anchor=tk.W)
        particle_diameter_entry = ttk.Entry(self.control_frame, textvariable=self.particle_diameter, font=entry_font)
        particle_diameter_entry.pack(anchor=tk.W)
        particle_diameter_entry.bind("<FocusOut>", self.on_parameter_change)
        particle_diameter_entry.bind("<Return>", self.on_parameter_change)
        
        # Add start button
        tk.Button(self.control_frame, text="Start Simulation", command=self.start_simulation, font=button_font).pack(anchor=tk.W, pady=20)
        
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

    def on_parameter_change(self, event=None):
        # Validate parameters
        fluid_height = self.fluid_height.get()
        fluid_length = self.fluid_length.get()
        box_height = self.box_height.get()
        box_length = self.box_length.get()
        particle_diameter = self.particle_diameter.get()
        
        if (fluid_height + particle_diameter) > box_height:
            self.fluid_height.set(self.fluid_height_prev)
            messagebox.showerror("Invalid Input", "(Fluid Height + Particle Diameter) darf nicht größer als Box Height sein.")
            return
        elif box_height < (fluid_height + particle_diameter):
            self.box_height.set(self.box_height_prev)
            messagebox.showerror("Invalid Input", "Box Height darf nicht kleiner als (Fluid Height + Particle Diameter) sein.")
            return
        
        if (fluid_length + particle_diameter) > box_length:
            self.fluid_length.set(self.fluid_length_prev)
            messagebox.showerror("Invalid Input", "(Fluid Length + Particle Diameter) darf nicht größer als Box Length sein.")
            return
        elif box_length < (fluid_length + particle_diameter):
            self.box_length.set(self.box_length_prev)
            messagebox.showerror("Invalid Input", "(Box Length + Particle Diameter) darf nicht kleiner als Fluid Length sein.")
            return
        
        if particle_diameter > fluid_length or particle_diameter > fluid_height:
            self.particle_diameter.set(self.particle_diameter_prev)
            messagebox.showerror("Invalid Input", "Particle Diameter darf nicht größer als Fluid Length oder Fluid Height sein.")
            return

        # Save current values as previous values
        self.fluid_height_prev = fluid_height
        self.fluid_length_prev = fluid_length
        self.box_height_prev = box_height
        self.box_length_prev = box_length
        self.particle_diameter_prev = particle_diameter
        
        self.visualize_boundary()

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    main()
