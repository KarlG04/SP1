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
        style = ttk.Style()
        style.configure("TSeparator", background="gray")
        
        # Create frames
        self.control_frame = ttk.Frame(self.root, padding="14")
        self.control_frame.place(relx=1.0, rely=0.5, anchor="e", relheight=1.0)
        
        self.plot_frame = ttk.Frame(self.root, padding="0")
        self.plot_frame.place(relx=0.0, rely=0.0, anchor="nw", relheight=1.0, relwidth=0.875)
        
        # Add a thick separator between the plot frame and the control frame
        self.separator_right = ttk.Frame(self.root, width=4, style="TSeparator")
        self.separator_right.place(relx=0.875, rely=0.0, relheight=1.0)
        
        # Add control elements
        self.setup_controls()
        
        # Add plot
        self.setup_plot()

    def setup_controls(self):
        label_font = ("Helvetica", 18)
        entry_font = ("Helvetica", 18)
        button_font = ("Helvetica", 18, "bold")
        entry_width = 20
        
        ttk.Label(self.control_frame, text="Simulationsparameter", font=("Helvetica", 19)).pack(anchor=tk.W)
        
        # Add entries for parameters
        self.fluid_height = tk.DoubleVar(value=130.0)
        self.fluid_height_prev = self.fluid_height.get()
        ttk.Label(self.control_frame, text="Fluid Height [m]", font=label_font).pack(anchor=tk.W, pady=(20, 2))
        fluid_height_entry = ttk.Entry(self.control_frame, textvariable=self.fluid_height, font=entry_font, width=entry_width)
        fluid_height_entry.pack(anchor=tk.W, pady=(0, 24))
        fluid_height_entry.bind("<FocusOut>", self.on_parameter_change)
        fluid_height_entry.bind("<Return>", self.on_parameter_change)
        
        self.fluid_length = tk.DoubleVar(value=50.0)
        self.fluid_length_prev = self.fluid_length.get()
        ttk.Label(self.control_frame, text="Fluid Length [m]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        fluid_length_entry = ttk.Entry(self.control_frame, textvariable=self.fluid_length, font=entry_font, width=entry_width)
        fluid_length_entry.pack(anchor=tk.W, pady=(0, 20))
        fluid_length_entry.bind("<FocusOut>", self.on_parameter_change)
        fluid_length_entry.bind("<Return>", self.on_parameter_change)
        
        self.box_height = tk.DoubleVar(value=200.0)
        self.box_height_prev = self.box_height.get()
        ttk.Label(self.control_frame, text="Box Height [m]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        box_height_entry = ttk.Entry(self.control_frame, textvariable=self.box_height, font=entry_font, width=entry_width)
        box_height_entry.pack(anchor=tk.W, pady=(0, 20))
        box_height_entry.bind("<FocusOut>", self.on_parameter_change)
        box_height_entry.bind("<Return>", self.on_parameter_change)
        
        self.box_length = tk.DoubleVar(value=200.0)
        self.box_length_prev = self.box_length.get()
        ttk.Label(self.control_frame, text="Box Length [m]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        box_length_entry = ttk.Entry(self.control_frame, textvariable=self.box_length, font=entry_font, width=entry_width)
        box_length_entry.pack(anchor=tk.W, pady=(0, 20))
        box_length_entry.bind("<FocusOut>", self.on_parameter_change)
        box_length_entry.bind("<Return>", self.on_parameter_change)
        
        self.particle_diameter = tk.DoubleVar(value=5.0)
        self.particle_diameter_prev = self.particle_diameter.get()
        ttk.Label(self.control_frame, text="Particle Diameter [m]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        particle_diameter_entry = ttk.Entry(self.control_frame, textvariable=self.particle_diameter, font=entry_font, width=entry_width)
        particle_diameter_entry.pack(anchor=tk.W, pady=(0, 20))
        particle_diameter_entry.bind("<FocusOut>", self.on_parameter_change)
        particle_diameter_entry.bind("<Return>", self.on_parameter_change)
        
        # Add entries for new parameters
        self.initial_density = tk.DoubleVar(value=1.0)
        ttk.Label(self.control_frame, text="Initial Density [kg/m³]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        initial_density_entry = ttk.Entry(self.control_frame, textvariable=self.initial_density, font=entry_font, width=entry_width)
        initial_density_entry.pack(anchor=tk.W, pady=(0, 20))

        self.smoothing_length = tk.DoubleVar(value=5.0)
        ttk.Label(self.control_frame, text="Smoothing Length [m]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        smoothing_length_entry = ttk.Entry(self.control_frame, textvariable=self.smoothing_length, font=entry_font, width=entry_width)
        smoothing_length_entry.pack(anchor=tk.W, pady=(0, 20))

        self.isentropic_exponent = tk.DoubleVar(value=20.0)
        ttk.Label(self.control_frame, text="Isentropic Exponent [1]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        isentropic_exponent_entry = ttk.Entry(self.control_frame, textvariable=self.isentropic_exponent, font=entry_font, width=entry_width)
        isentropic_exponent_entry.pack(anchor=tk.W, pady=(0, 20))

        self.dynamic_viscosity = tk.DoubleVar(value=0.5)
        ttk.Label(self.control_frame, text="dynamic viscosity [Ns/m²]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        dynamic_viscosity_entry = ttk.Entry(self.control_frame, textvariable=self.dynamic_viscosity, font=entry_font, width=entry_width)
        dynamic_viscosity_entry.pack(anchor=tk.W, pady=(0, 20))

        self.delta_t = tk.DoubleVar(value=0.01)
        ttk.Label(self.control_frame, text="Δt [s]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        delta_t_entry = ttk.Entry(self.control_frame, textvariable=self.delta_t, font=entry_font, width=entry_width)
        delta_t_entry.pack(anchor=tk.W, pady=(0, 20))
        delta_t_entry.bind("<FocusOut>", self.on_parameter_change)
        delta_t_entry.bind("<Return>", self.on_parameter_change)

        self.boundary_damping = tk.DoubleVar(value=-0.6)
        ttk.Label(self.control_frame, text="Boundary Damping [1]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        boundary_damping_entry = ttk.Entry(self.control_frame, textvariable=self.boundary_damping, font=entry_font, width=entry_width)
        boundary_damping_entry.pack(anchor=tk.W, pady=(0, 20))

        # Add Gravity entries as a combined section with X and Y inputs
        ttk.Label(self.control_frame, text="Gravity [m/s²]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        gravity_frame = ttk.Frame(self.control_frame)
        gravity_frame.pack(anchor=tk.W, pady=(0, 20))
        
        ttk.Label(gravity_frame, text="X", font=label_font).pack(side=tk.LEFT)
        self.gravity_x = tk.DoubleVar(value=0.0)
        gravity_x_entry = ttk.Entry(gravity_frame, textvariable=self.gravity_x, font=entry_font, width=7)
        gravity_x_entry.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(gravity_frame, text="Y", font=label_font).pack(side=tk.LEFT)
        self.gravity_y = tk.DoubleVar(value=-0.0981)
        gravity_y_entry = ttk.Entry(gravity_frame, textvariable=self.gravity_y, font=entry_font, width=7)
        gravity_y_entry.pack(side=tk.LEFT)
        
        self.num_time_steps = tk.IntVar(value=2000)
        ttk.Label(self.control_frame, text="Time Steps [n]", font=label_font).pack(anchor=tk.W, pady=(10, 2))
        num_time_steps_entry = ttk.Entry(self.control_frame, textvariable=self.num_time_steps, font=entry_font, width=entry_width)
        num_time_steps_entry.pack(anchor=tk.W, pady=(0, 20))
        num_time_steps_entry.bind("<FocusOut>", self.on_parameter_change)
        num_time_steps_entry.bind("<Return>", self.on_parameter_change)
        
        # Add start button
        start_button = tk.Button(self.control_frame, text="Start Simulation", command=self.start_simulation, font=button_font, bg="#1bde1b", fg="white")
        start_button.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=20)
        start_button.config(relief="raised", bd=3, highlightthickness=2, highlightbackground="#1bde1b")

        # Adjust the button style for rounded corners
        style = ttk.Style()
        style.configure("RoundedButton.TButton", relief="flat", borderwidth=1, padding=4, background="#1bde1b", foreground="white")
        style.map("RoundedButton.TButton", background=[('active', '#76c776')])

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
                                         self.fluid_height.get(),
                                         self.num_time_steps.get(),
                                         self.delta_t.get())
        self.canvas.draw()
        
    def start_simulation(self):
        # Gather parameters
        gravity = (self.gravity_x.get(), self.gravity_y.get())
        initial_density = self.initial_density.get()
        num_time_steps = self.num_time_steps.get()
        spacing = self.particle_diameter.get()
        smoothing_length = self.smoothing_length.get()
        isentropic_exponent = self.isentropic_exponent.get()
        delta_t = self.delta_t.get()
        boundary_damping = self.boundary_damping.get()
        dynamic_viscosity = self.dynamic_viscosity.get()
        mass_per_particle = 1

        # Kernel Factors
        density_factor = (315 * mass_per_particle) / (64 * np.pi * smoothing_length**9)
        pressure_factor = (-(45 * mass_per_particle) / (np.pi * smoothing_length**6))
        viscosity_factor = (45 * dynamic_viscosity * mass_per_particle) / (np.pi * smoothing_length**6)
        
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
        delta_t = self.delta_t.get()
        num_time_steps = self.num_time_steps.get()
        
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
        
        # Aktualisiere den Plot
        self.visualize_boundary()

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    main()