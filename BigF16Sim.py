import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Plane:
    def __init__(self, mass, inertia_tensor, wing_area, air_density, initial_state):
        # Mass and inertia
        self.mass = mass  # kg
        self.inertia_tensor = inertia_tensor  # 3x3 numpy array

        # Aerodynamic properties
        self.wing_area = wing_area  # m^2
        self.air_density = air_density  # kg/m^3

        # State variables
        self.position = np.array(initial_state['position'])  # m
        self.velocity = np.array(initial_state['velocity'])  # m/s in body frame
        self.orientation = R.from_quat(initial_state['orientation'])  # Quaternion
        self.angular_velocity = np.array(initial_state['angular_velocity'])  # rad/s in body frame

        # Control inputs
        self.thrust = initial_state.get('thrust', 0.0)  # N
        self.control_surfaces = initial_state.get('control_surfaces', {'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0})  # Degrees

        self.CX_updated_data_full = np.array([
        [-0.099, -0.081, -0.081, -0.063, -0.025, 0.044, 0.097, 0.113, 0.145, 0.167, 0.174, 0.166],
        [-0.048, -0.038, -0.040, -0.021, 0.016, 0.083, 0.127, 0.137, 0.162, 0.177, 0.179, 0.167],
        [-0.022, -0.020, -0.021, -0.004, 0.032, 0.094, 0.128, 0.130, 0.154, 0.161, 0.155, 0.138],
        [-0.040, -0.038, -0.039, -0.025, 0.006, 0.062, 0.087, 0.085, 0.100, 0.110, 0.104, 0.091],
        [-0.083, -0.073, -0.076, -0.072, -0.046, 0.012, 0.024, 0.025, 0.043, 0.053, 0.047, 0.040]])

        self.CZ_updated_data_full = np.array([
            0.770, 0.241, -0.100, -0.416, -0.731, -1.053,
            -1.366, -1.646, -1.917, -2.120, -2.248, -2.229
        ])

    def compute_angle_of_attack(self):
        # Transform velocity to inertial frame
        v_inertial = self.orientation.apply(self.velocity)
        vx, vy, vz = v_inertial

        # Compute angle of attack (alpha)
        alpha = np.arctan2(vz, vx)
        return alpha

    def compute_sideslip_angle(self):
        # Transform velocity to inertial frame
        v_inertial = self.orientation.apply(self.velocity)
        vx, vy, vz = v_inertial

        # Compute sideslip angle (beta)
        beta = np.arctan2(vy, np.sqrt(vx**2 + vz**2))
        return beta
    
    def compute_cx_fixed(self,el):
        alpha = self.compute_angle_of_attack()
        beta = self.compute_sideslip_angle()
        # Map Alpha and EL into indices and interpolate
        data=self.CX_updated_data_full
        scaled_alpha = 0.2 * alpha
        k = int(np.floor(scaled_alpha))
        k = max(-2, min(k, 8))  # Constrain within bounds
        da = scaled_alpha - k
        l = k + int(np.sign(1.1 * da))
        l = max(-2, min(l, 8))  # Constrain within bounds

        k_index = k + 2  # Adjust index for array (0-based)
        l_index = l + 2  # Adjust index for array (0-based)

        scaled_el = el / 12.5
        m = int(np.floor(scaled_el))
        m = max(-2, min(m, 2))  # Constrain within bounds
        de = scaled_el - m
        n = m + int(np.sign(1.1 * de))
        n = max(-2, min(n, 2))  # Constrain within bounds

        m_index = m + 2  # Adjust index for array (0-based)
        n_index = n + 2  # Adjust index for array (0-based)

        # Ensure indices are within valid ranges

        # Interpolate values
        t = data[m_index, k_index]
        u = data[n_index, k_index]
        v = t + abs(da) * (data[m_index, l_index] - t)
        w = u + abs(da) * (data[n_index, l_index] - u)

        return v + (w - v) * abs(de) 
    
    def compute_cy(self,AIL,RDR):
        beta = self.compute_sideslip_angle()
        # Map Alpha into indices and interpolate
        c_y = (-0.02) * beta + (0.021) * (AIL / 20.0) + (0.086) * (RDR / 30.0)
        return c_y

    def compute_cz(self,el):
        alpha = self.compute_angle_of_attack()
        beta = self.compute_sideslip_angle()    
        data= self.CZ_updated_data_full
        # Map Alpha into indices and interpolate
        scaled_alpha = 0.2 * alpha
        k = int(np.floor(scaled_alpha))
        k = max(-2, min(k, 8))  # Constrain within bounds
        da = scaled_alpha - k
        l = k + int(np.sign(1.1 * da))
        l = max(-2, min(l, 8))  # Constrain within bounds
        
        k_index = k + 2  # Adjust index for array (0-based)
        l_index = l + 2  # Adjust index for array (0-based)
        
        s = data[k_index] + abs(da) * (data[l_index] - data[k_index])
        return s * (1 - (beta / 57.3) ** 2) - 0.19 * (el / 25.0)  

    def compute_aerodynamic_forces(self):
        alpha = self.compute_angle_of_attack()
        beta = self.compute_sideslip_angle()
        Cx = self.compute_cx_fixed(self.control_surfaces['elevator'])
        Cz = self.compute_cz(self.control_surfaces['elevator'])
        Cy = self.compute_cy(self.control_surfaces['aileron'],self.control_surfaces['rudder'])
        airspeed = np.linalg.norm(self.velocity)

        # Simplified lift and drag coefficients
        # CL = 2 * np.pi * alpha  # Lift coefficient
        # CD = 0.02 + 0.04 * (alpha ** 2)  # Drag coefficient

        # # Dynamic pressure
        q = 0.5 * self.air_density * airspeed ** 2

        # # Aerodynamic forces in body frame
        # lift = q * self.wing_area * CL
        # drag = q * self.wing_area * CD

        # # Thrust force in body frame (assuming along x-axis)
        # thrust_force = np.array([self.thrust, 0, 0])

        # # Total aerodynamic force
        # aerodynamic_force = np.array([
        #     -drag * np.cos(alpha) + lift * np.sin(alpha),
        #     0,  # Neglecting side force for simplicity
        #     -drag * np.sin(alpha) - lift * np.cos(alpha)
        # ])

        aerodynamic_force = np.array([
            q*Cx*self.wing_area,
            q*Cy*self.wing_area,  # Neglecting side force for simplicity
            q*Cz*self.wing_area
        ])
        thrust_force = np.array([self.thrust, 0, 0])
        total_force = aerodynamic_force + thrust_force
        return total_force

    def compute_aerodynamic_moments(self):
        # Placeholder for aerodynamic moments due to control surfaces
        moments = np.array([
            self.control_surfaces['aileron'],  # Roll moment
            self.control_surfaces['elevator'],  # Pitch moment
            self.control_surfaces['rudder']     # Yaw moment
        ])
        # Convert degrees to radians and apply effectiveness factors
        moments = np.deg2rad(moments) * self.get_control_effectiveness()
        return moments

    def get_control_effectiveness(self):
        # Placeholder for control effectiveness coefficients
        return np.array([1000.0, 1000.0, 1000.0])  # Adjust as needed

    def update_state(self, dt):
        # Compute forces and moments
        forces_body = self.compute_aerodynamic_forces()
        moments_body = self.compute_aerodynamic_moments()

        # Linear acceleration in body frame
        acceleration_body = forces_body / self.mass

        # Update velocities
        self.velocity += acceleration_body * dt

        # Update positions (convert velocity to inertial frame)
        velocity_inertial = self.orientation.apply(self.velocity)
        self.position += velocity_inertial * dt

        # Angular acceleration
        angular_acceleration = np.linalg.inv(self.inertia_tensor).dot(
            moments_body - np.cross(self.angular_velocity, self.inertia_tensor.dot(self.angular_velocity))
        )

        # Update angular velocities
        self.angular_velocity += angular_acceleration * dt

        # Update orientation
        delta_orientation = R.from_rotvec(self.angular_velocity * dt)
        self.orientation = delta_orientation * self.orientation  # Order matters

    def set_controls(self, thrust, aileron, elevator, rudder):
        self.thrust = thrust
        self.control_surfaces = {
            'aileron': aileron,
            'elevator': elevator,
            'rudder': rudder
        }

initial_state = {
    'position': [0.0, 0.0, -1000.0],  # Starting at 1000 meters altitude (negative z for upward)
    'velocity': [50.0, 0.0, 0.0],    # Forward speed of 250 m/s
    'orientation': [0.0, 0.0, 0.0, 1.0],  # Quaternion [x, y, z, w]
    'angular_velocity': [0.0, 0.0, 0.0],
    'thrust': 50000.0,  # N
    'control_surfaces': {'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}
}

mass = 9300.0  # kg
inertia_tensor = np.diag([12875.0, 75673.6, 85552.1])  # kg*m^2
wing_area = 27.87  # m^2
air_density = 1.225  # kg/m^3 at sea level


F_16 = Plane(mass, inertia_tensor, wing_area, air_density, initial_state)

dt = 0.01  # Time step in seconds
simulation_time = 10.0  # Total simulation time in seconds
num_steps = int(simulation_time / dt)

positions = []
velocities = []
orientations = []
times = []

for i in range(num_steps):
    # Example control input: small elevator deflection after 2 seconds
    if i * dt > 0.0:
        F_16.set_controls(
            thrust=50000.0,
            aileron=0.0,
            elevator=25.0,  # Deflect elevator by 5 degrees
            rudder=0.0
        )
    else:
        F_16.set_controls(
            thrust=50000.0,
            aileron=0.0,
            elevator=0.0,
            rudder=0.0
        )

    # Update state
    F_16.update_state(dt)

    # Record data
    positions.append(F_16.position.copy())
    velocities.append(F_16.velocity.copy())
    orientations.append(F_16.orientation.as_euler('xyz', degrees=True))
    times.append(i * dt)

# Convert lists to arrays
positions = np.array(positions)
velocities = np.array(velocities)
orientations = np.array(orientations)
times = np.array(times)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(times, -positions[:, 2])  # Negative Z for altitude
plt.title('Altitude over Time')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.grid(True)
plt.show()











