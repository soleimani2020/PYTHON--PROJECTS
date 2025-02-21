This code post-processes molecular dynamics (MD) simulation data to calculate pressure and refines it using the Irving-Kirkwood method. The simulation is performed using GROMACS, which generates trajectory files that are analyzed with MDAnalysis.

In MD simulations, pressure arises from both particle motion and interactions:

    Kinetic Contribution – Comes from the thermal motion of particles, where higher velocities lead to increased pressure.
    Virial Contribution – Accounts for intermolecular forces such as van der Waals and electrostatic interactions, which influence the overall pressure distribution.

Since the system contains interfaces, a simple global pressure calculation is insufficient. Instead, the Irving-Kirkwood method is used to compute a spatially resolved pressure profile by distributing intermolecular forces along the interaction paths between particles. This approach is particularly useful for studying membranes, confined fluids, and other inhomogeneous systems.
