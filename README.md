# Dynamics Engine for 2D Kinematic Chain

## Objective
This project involves implementing a dynamics engine (a.k.a. physics simulator) for a 2D kinematic chain with an arbitrary number of links. The core components of this physics engine are a solver and an integrator, designed to handle external forces and internal constraints, compute accelerations, and update the positions and velocities of the links over time.

| 3-Link Robot | 2-Link Robot | 1-Link Robot |
|---|---|---|
| [![YouTube Video 1](https://img.youtube.com/vi/vBX-3_Odsxo/0.jpg)](https://www.youtube.com/watch?v=vBX-3_Odsxo) | [![YouTube Video 2](https://img.youtube.com/vi/mm9u_s_-hEk/0.jpg)](https://www.youtube.com/watch?v=mm9u_s_-hEk) | [![YouTube Video 3](https://img.youtube.com/vi/hCSBlupd65Y/0.jpg)](https://www.youtube.com/watch?v=hCSBlupd65Y) |

## Key Features

1. **Solver Implementation:**
   - Solver to compute accelerations of all bodies in the kinematic chain at each time step using the Newton-Euler algorithm.
   - Handled both external forces and internal constraints effectively.

2. **Numerical Integration:**
   - Euler integration method to update the positions and velocities of the links based on the computed accelerations.

3. **Simulation Environment:**
   - Custom GUI for real-time visualization of the robot's state during simulation.
   - Allowed for interactive testing and debugging.

4. **Testing and Validation:**
   - Testing with 1-link, 2-link, and 3-link robots.
   - Adjusted parameters such as friction, time step size, and applied torques to validate the dynamics engine's accuracy and robustness.

## Project Structure
- `arm_dynamics_engine.py`: Main script for implementing the dynamics engine for the 2D kinematic chain.
