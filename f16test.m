% Define parameters
clc; clear;

% Aerodynamic coefficient data
% CX Data (5x12)
CX_updated_data_full = [
    -0.099, -0.081, -0.081, -0.063, -0.025, 0.044, 0.097, 0.113, 0.145, 0.167, 0.174, 0.166;
    -0.048, -0.038, -0.040, -0.021, 0.016, 0.083, 0.127, 0.137, 0.162, 0.177, 0.179, 0.167;
    -0.022, -0.020, -0.021, -0.004, 0.032, 0.094, 0.128, 0.130, 0.154, 0.161, 0.155, 0.138;
    -0.040, -0.038, -0.039, -0.025, 0.006, 0.062, 0.087, 0.085, 0.100, 0.110, 0.104, 0.091;
    -0.083, -0.073, -0.076, -0.072, -0.046, 0.012, 0.024, 0.025, 0.043, 0.053, 0.047, 0.040
];

% CZ Data (1x12)
CZ_updated_data_full = [
     0.770, 0.241, -0.100, -0.416, -0.731, -1.053, ...
    -1.366, -1.646, -1.917, -2.120, -2.248, -2.229
];

% CM Data (5x12)
CM_updated_data_full = [
    0.205, 0.168, 0.186, 0.196, 0.213, 0.251, 0.245, 0.238, 0.252, 0.231, 0.198, 0.192;
    0.081, 0.077, 0.107, 0.110, 0.110, 0.141, 0.127, 0.119, 0.133, 0.108, 0.081, 0.093;
    -0.046, -0.020, -0.009, -0.005, -0.006, 0.010, 0.006, -0.001, 0.014, 0.000, -0.013, 0.032;
    -0.174, -0.145, -0.121, -0.127, -0.129, -0.102, -0.097, -0.113, -0.087, -0.084, -0.069, -0.006;
    -0.259, -0.202, -0.184, -0.193, -0.199, -0.150, -0.160, -0.167, -0.104, -0.076, -0.041, -0.005
];

% CL Data (6x12)
CL_updated_data_full = [
    -0.001, -0.004, -0.008, -0.012, -0.016, -0.019, -0.020, -0.020, -0.015, -0.008, -0.013, -0.015;
    -0.003, -0.009, -0.017, -0.024, -0.030, -0.034, -0.040, -0.037, -0.016, -0.002, -0.010, -0.019;
    -0.001, -0.010, -0.020, -0.030, -0.039, -0.044, -0.050, -0.049, -0.023, -0.006, -0.014, -0.027;
    0.000, -0.010, -0.022, -0.034, -0.047, -0.046, -0.059, -0.061, -0.033, -0.036, -0.035, -0.035;
    0.007, -0.010, -0.023, -0.034, -0.049, -0.046, -0.068, -0.071, -0.060, -0.058, -0.062, -0.059;
    0.009, -0.011, -0.023, -0.037, -0.050, -0.047, -0.074, -0.079, -0.091, -0.076, -0.077, -0.076
];

% CN Data (6x12)
CN_updated_data_full = [
    0.018, 0.019, 0.018, 0.019, 0.019, 0.018, 0.013, 0.007, 0.004, -0.014, -0.017, -0.033;
    0.038, 0.042, 0.042, 0.042, 0.043, 0.039, 0.030, 0.017, 0.004, -0.035, -0.047, -0.057;
    0.056, 0.057, 0.059, 0.058, 0.058, 0.053, 0.032, 0.012, 0.002, -0.046, -0.071, -0.073;
    0.064, 0.077, 0.076, 0.074, 0.073, 0.057, 0.029, 0.007, 0.012, -0.034, -0.065, -0.041;
    0.074, 0.086, 0.093, 0.089, 0.080, 0.062, 0.049, 0.022, 0.028, -0.012, -0.002, -0.013;
    0.079, 0.090, 0.106, 0.106, 0.096, 0.080, 0.068, 0.030, 0.064, 0.015, 0.011, -0.001
];

% Aircraft parameters
S = 27.87;    % Reference area (m^2)
m = 9295.412844036697; % Mass (kg)
rho = 1.225;  % Air density (kg/m^3)
g = 9.81;     % Gravity (m/s^2)
Ixx = 12875;  % Moment of inertia (kg·m^2)
Iyy = 75673;  % Moment of inertia (kg·m^2)
Izz = 85552;  % Moment of inertia (kg·m^2)

% Initial conditions
u = 200; v = 0; w = 0; % Body velocities (m/s)
p = 0; q = 0; r = 0; % Angular rates (rad/s)
phi = 0; theta = 0; psi = 0; % Orientation (rad)
x = 0; y = 0; z = 0; % Position (m)

% Time parameters
dt = 0.01; sim_time = 10; % Time step and duration (s)
n_steps = sim_time / dt;

% Control surface deflections and thrust
AIL = 0; RDR = 0; ELE = 0; Thrust = 100000; % Initial values

% Trajectory storage
trajectory = zeros(n_steps, 15);

% Grids for interpolation
alpha_grid = linspace(-10, 25, 12);
beta_grid_CX_CM = linspace(-20, 20, 5);
beta_grid_CL_CN = linspace(-20, 20, 6);

% Simulation loop
for t = 1:n_steps
    % Dynamic adjustments (example: after 5 seconds)
    if t * dt > 5
        ELE = -5; % Elevator deflection
        Thrust = 40000; % Reduce thrust
    end
    
    % Compute angles
    aoa = atan2(w, u); % Angle of attack (rad)
    beta = asin(v / sqrt(u^2 + v^2 + w^2)); % Sideslip angle (rad)
    
    % Interpolate aerodynamic coefficients
    CX = interp2(alpha_grid, beta_grid_CX_CM, CX_updated_data_full, rad2deg(aoa), rad2deg(beta), 'linear', 0);
    CZ = interp1(alpha_grid, CZ_updated_data_full, rad2deg(aoa), 'linear', 0) + 0.1 * (ELE / 10);
    CM = interp2(alpha_grid, beta_grid_CX_CM, CM_updated_data_full, rad2deg(aoa), rad2deg(beta), 'linear', 0) + 0.02 * (ELE / 10);
    CL = interp2(alpha_grid, beta_grid_CL_CN, CL_updated_data_full, rad2deg(aoa), rad2deg(beta), 'linear', 0);
    CN = interp2(alpha_grid, beta_grid_CL_CN, CN_updated_data_full, rad2deg(aoa), rad2deg(beta), 'linear', 0);
    
    % Forces and moments
    q_dyn = 0.5 * rho * (u^2 + v^2 + w^2);
    Fx = q_dyn * S * CX + Thrust;
    Fy = q_dyn * S * (-0.02 * beta + 0.021 * (AIL / 20.0) + 0.086 * (RDR / 30.0));
    Fz = q_dyn * S * CZ - m * g;
    Mx = q_dyn * S * CL;
    My = q_dyn * S * CM;
    Mz = q_dyn * S * CN;
    
    % Translational motion
    du = r * v - q * w + Fx / m;
    dv = p * w - r * u + Fy / m;
    dw = q * u - p * v + Fz / m;
    
    % Rotational motion
    dp = (Iyy - Izz) / Ixx * q * r + Mx / Ixx;
    dq = (Izz - Ixx) / Iyy * p * r + My / Iyy;
    dr = (Ixx - Iyy) / Izz * p * q + Mz / Izz;
    
    % Update state variables
    u = u + du * dt; v = v + dv * dt; w = w + dw * dt;
    p = p + dp * dt; q = q + dq * dt; r = r + dr * dt;
    dx = u * cos(theta) * cos(psi) + ...
         v * (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) + ...
         w * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi));
    dy = u * cos(theta) * sin(psi) + ...
         v * (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) + ...
         w * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi));
    dz = -u * sin(theta) + v * sin(phi) * cos(theta) + w * cos(phi) * cos(theta);
    x = x + dx * dt; y = y + dy * dt; z = z + dz * dt;
    dphi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta);
    dtheta = q * cos(phi) - r * sin(phi);
    dpsi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta);
    phi = phi + dphi * dt; theta = theta + dtheta * dt; psi = psi + dpsi * dt;
    
    % Store results
    trajectory(t, :) = [t * dt, x, y, z, u, v, w, rad2deg(aoa), rad2deg(beta), p, q, r, rad2deg(phi), rad2deg(theta), rad2deg(psi)];
end

% Extract data for plotting
time = trajectory(:, 1); x_earth = trajectory(:, 2); y_earth = trajectory(:, 3); z_earth = -trajectory(:, 4);
uvw = trajectory(:, 5:7); pqr = trajectory(:, 10:12); euler_angles = trajectory(:, 13:15);
alpha_deg = trajectory(:, 8); beta_deg = trajectory(:, 9);

% Plot results
figure;
subplot(5, 1, 1); plot(time, alpha_deg, 'b', time, beta_deg, 'r'); grid on;
xlabel('Time (s)'); ylabel('Angle (deg)'); title('Angle of Attack (\alpha) and Sideslip Angle (\beta)'); legend('\alpha (deg)', '\beta (deg)');
subplot(5, 1, 2); plot(time, pqr(:, 1), 'r', time, pqr(:, 2), 'g', time, pqr(:, 3), 'b'); grid on;
xlabel('Time (s)'); ylabel('Angular Rate (rad/s)'); title('Angular Rates (P, Q, R)'); legend('p', 'q', 'r');
subplot(5, 1, 3); plot(time, uvw(:, 1), 'r', time, uvw(:, 2), 'g', time, uvw(:, 3), 'b'); grid on;
xlabel('Time (s)'); ylabel('Velocity (m/s)'); title('Body Velocities (U, V, W)'); legend('u', 'v', 'w');
subplot(5, 1, 4); plot(time, euler_angles(:, 1), 'r', time, euler_angles(:, 2), 'g', time, euler_angles(:, 3), 'b'); grid on;
xlabel('Time (s)'); ylabel('Angle (deg)'); title('Euler Angles (\phi, \theta, \psi)'); legend('\phi (deg)', '\theta (deg)', '\psi (deg)');
subplot(5, 1, 5); plot(time, x_earth, 'r', time, y_earth, 'g', time, z_earth, 'b'); grid on;
xlabel('Time (s)'); ylabel('Position (m)'); title('Earth Frame Positions (X, Y, Z)'); legend('X (m)', 'Y (m)', 'Z (altitude, m)');
