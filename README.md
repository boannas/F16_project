# F16 Fighting Falcon SIMULATION
This project simulates the dynamics of a F16 Fighting Falcon
#### This project is a part of FRA333 Robot Kinematics @ Institute of Field Robotics, King Mongkut’s University of Technology Thonburi

## Table of contents
1. [Overview](#overview)
2. [Getting started](#getting-started)
3. [User guide](#user-guide)
4. [Methodology](#methodology)
    - [Navigation Equation](#1-navigation-equation)
    - [Kinematic Equation](#2-kinematic-equation)
    - [Force Equation](#3-force-equation)
    - [Moment Equation](#4-moment-equation)
    - [Aerodynamics Equation](#5-aerodynamics)
    - [Force Equation New State](#6-force-equation-new-state)
    - [Linear Velocity Equation](#7-linear-velocity)
    - [Dynamic Pressure Equation](#8-dynamic-pressure)
    - [Coefficient Equation](#9-coefficient)
5. [Validation](#validation)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Acknowledgements](#acknowledgements)


## Overview
![Image of system diagram](image/system_diagram.png)

### Feature
- **Simulation :** Can simulates dynamics of F16 Fighting Falcon by asjust control surface.

## Getting started
Can use only on window OS.
### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/boannas/F16_project.git
cd F16_project/Simulation
```
### Open Simulation
```bash
F-16.exe
```

## User guide
You can config parameter of F16.
- `Thrust` - For control X axis of airplane. 
- `Rudder` - For control yaw axis of airplane. 
- `Elevator` - For control pitch axis of airplane. 
- `Ailerons` - For control row axis of airplane. 

## Methodology
**Control Surfaces**
![Image of system diagram](image/control_surface.png)
- `Aileron` - Ailerons are a primary flight control surface which control movement about the longitudinal axis of an aircraft. This movement is referred to as "roll".
- `Elevator` - An elevator is a primary flight control surface that controls movement about the lateral axis of an aircraft.This movement is referred to as "pitch".
- `Rudder` -  The rudder is a primary flight control surface which controls rotation about the vertical axis of an aircraft.This movement is referred to as "yaw"
- `Thrust` - Thrust is the force which moves an aircraft through the air. Thrust is generated by the engines of the airplane.


### **F-16 Parameters**
| Parameter    | Description                                          | Value        | Unit       |
|--------------|------------------------------------------------------|--------------|------------|
| 𝑚           | Aircraft mass                                        | 9295.44      | kg         |
| 𝐵           | Wing span                                            | 9.144        | m          |
| 𝑆           | Planform area                                        | 27.87        | m²         |
| c̄           | Mean aerodynamic chord                               | 3.45         | m          |
| 𝑥₍c.g.,r₎    | Reference center of gravity as a fraction of mean aerodynamic chord | 0.35         | -          |
| 𝑥₍c.g.₎      | Center of gravity as a fraction of mean aerodynamic chord | 0.3          | -          |
| 𝐼ₓₓ         | Moment of inertia around \(x\)-axis                  | 12874.847366 | kg·m²      |
| 𝐼ᵧᵧ         | Moment of inertia around \(y\)-axis                  | 75673.623725 | kg·m²      |
| 𝐼𝓏𝓏         | Moment of inertia around \(z\)-axis                  | 85552.113395 | kg·m²      |
| 𝐼ₓ𝓏         | Product moment of inertia between \(x\) and \(z\) axes | 1331.4132386 | kg·m²      |

### **F-16 Variables**
| Variable      | Description                                     | Unit      |
|---------------|-------------------------------------------------|-----------|
| 𝑣ₜ           | Aircraft airspeed in the wind coordinate system | m/s       |
| α             | Angle of attack                                | rad       |
| β             | Side slip angle                                | rad       |
| ϕ, θ, ψ       | Roll, pitch, and yaw angles (Euler angles)      | rad       |
| 𝑃, 𝑄, 𝑅      | Roll, pitch, and yaw angular rates              | rad/s     |
| 𝑈, 𝑉, 𝑊      | Axial linear velocities                        | m/s       |
| 𝑥ₑ, 𝑦ₑ, ℎ = −𝑧ₑ | Position in the earth reference frame         | m         |

### **Constant Variables**
| Constant     | Description                    | Value     | Unit          |
|--------------|--------------------------------|-----------|---------------|
| 𝑔           | Gravitational acceleration     | 9.81      | m/s²          |
| 𝜌           | Air density (at sea level)     | 1.225     | kg/m³         |


### **F-16 Fixed and Free States**
| Fixed Quantities          | Free States/Controls                            |
|---------------------------|------------------------------------------------|
| 𝑉ₜ = 500 ft/s            | 𝑥ₑ = 0 ft                                    |
| 𝑝 = 0 rad/s              | 𝑦ₑ = 0 ft                                    |
| 𝑞 = 0 rad/s              | ℎ = 10,000 ft                                 |
| 𝑟 = 0 rad/s              | 𝛾 = 0.349                                    |
| ψ = 0 rad/s              | ψ̇ = 0.052 rad/s                              |
| 𝑥₍c.g.₎ = 0.3            | 𝛼, 𝛽, ϕ, θ, δₜₕ, δₑ, δₐ, and δᵣ          |

- **δₜₕ**: Throttle angle (Minimum 0.0°, Maximum 1.0°) 

- **δₑ**: Elevator angle (Minimum -25.0°, Maximum 25.0°)

- **δₐ**: Aileron angle (Minimum -21.5°, Maximum 21.5°)

- **δᵣ**: Rudder angle (Minimum -30.0°, Maximum 30.0°)

| Constant     | Description                    | Value     | Unit          |
|--------------|--------------------------------|-----------|---------------|
| 𝑔           | Gravitational acceleration     | 9.81      | m/s²          |
| 𝜌           | Air density (at sea level)     | 1.225     | kg/m³         |


#### 1. Navigation Equation
<p align="center">
  <img src="image/Navigation_equation.png" alt="Image of Navigation Equation">
</p>

The transformation of the rotation matrix from the body frame of the F-16 aircraft to the Earth frame can be expressed using roll, pitch, and yaw (Rz Ry Rx).

Where:
- **ẋₑ**: The rate of change of position along the X-axis of the aircraft relative to the Earth frame.
- **ẏₑ**: The rate of change of position along the Y-axis of the aircraft relative to the Earth frame.
- **ż**: The rate of change of position along the Z-axis of the aircraft relative to the Earth frame.

#### 2. Kinematic Equation
<p align="center">
  <img src="image/f16_dynamics.png" alt="Image of f16_dynamics">
  <img src="image/Kinematic_equation.png" alt="Image of Kinematic Equation">
</p>

This equation describes the relationship between angular velocities in the body-fixed frame (P, Q, R) and the rates of change of the Euler angles (ϕ, θ, ψ).


Where:
- **(ϕ, θ, ψ)**: Roll, Pitch, Yaw (in radians)
- **(P, Q, R)**: Roll, Pitch, Yaw angular rates (in radians per second)


#### 3. Force Equation
<p align="center">
  <img src="image/Force_equation.png" alt="Image of Force equation">
</p>


This equation represents the translational dynamics of a rigid body in a body-fixed coordinate system.

#### 4. Moment Equation
<p align="center">
  <img src="image/Moment_equation.png" alt="Image of Moment Equation">
</p>


This set of equations represents the rotational dynamics of a rigid body in terms of its angular momentum and moments of inertia. These equations are derived from Euler's equations of motion for a rigid body.

where:
- **𝐼ₓₓ**: Moment of inertia around the X-axis (12874.847366 kg·m²)
- **𝐼ᵧᵧ**: Moment of inertia around the Y-axis (75673.623725 kg·m²)
- **𝐼𝓏𝓏**: Moment of inertia around the Z-axis (85552.113395 kg·m²)
- **𝑃, 𝑄, 𝑅**: Roll, Pitch, Yaw angular rates (rad/s)
- **𝐿, 𝑀, 𝑁**: External moments (Nm)
- **Γ**: Moments of inertia in the system (kg²·m⁴)

#### 5. Aerodynamics
<p align="center">
  <img src="image/Aerodynamics.png" alt="Image of Aerodynamics equation">
</p>

This set of equations describes the aerodynamic forces(𝐹ₓ,𝐹ᵧ,𝐹𝓏) and moments(L,M,N) acting on a body, such as an aircraft, in terms of aerodynamic coefficients. These forces and moments depend on several parameters like angle of attack(α), sideslip angle(β), control surface deflections(δₑ,δₐ,δᵣ), and the Mach number.

where:
- **q̄**: Dynamic pressure (Pa)
- **S**: Wing reference area (m²)
- **B**: Wingspan (m)
- **c̄**: Mean aerodynamic chord (m)

#### 6. Force equation new state
<p align="center">
  <img src="image/Force_equation_new_state.png" alt="Image of Force equation new state equation">
</p>

This set of equations appears to describe the translational and angular dynamics of a body in three dimensions.

where:
- **α**: Angle of attack (alpha) (องศา)

- **β**: Sideslip angle (beta) (องศา)

- **𝐹ₓ**: Force in the x-axis direction

- **𝐹ᵧ**: Force in the y-axis direction

- **𝐹𝓏**: Force in the z-axis direction


#### 7. Linear velocity  
$$
V_T = \sqrt{U^2 + V^2 + W^2}
$$

$$
\alpha = \tan^{-1}\left(\frac{W}{U}\right)
$$

$$
\beta = \sin^{-1}\left(\frac{V}{V_T}\right)
$$

$$
U = V_T \cos\alpha \cos\beta
$$

$$
V = V_T \sin\beta
$$

$$
W = V_T \sin\alpha \cos\beta
$$

This set of equations defines the relationship between the total velocity(𝑉ₜ) of a moving body and its velocity components in a 3D coordinate system. It also introduces two important aerodynamic angles: angle of attack(α), sideslip angle(β). These equations are widely used in flight mechanics and aerodynamics.

#### 8. Dynamic Pressure 
<p align="center">
𝑞̄ = ½ ρ 𝑉ₜ²
</p>


This equation represents dynamic pressure (q), a key parameter in fluid dynamics and aerodynamic.

where:
- ρ is air density,
- 𝑉ₜ is total velocity.

#### 9. Coefficient 
<p align="center">
  <img src="image/Coefficient.png" alt="Image of Coefficient">
</p>

This set of equations defines the aerodynamic coefficients that determine the aerodynamic forces and moments acting on an aircraft or similar body. These coefficients describe how the aerodynamic forces and moments depend on the angles of attack(α), sideslip angle(β), control surface deflections(δₑ,δₐ,δᵣ), and body rates(p,q,r)

where:
- **𝐶ₓ**: Non-dimensional \(x\)-body-axis force coefficient

- **𝐶ᵧ**: Non-dimensional \(y\)-body-axis force coefficient

- **𝐶𝓏**: Non-dimensional \(z\)-body-axis force coefficient

- **𝐶ₗ**: Lift coefficient

- **𝐶ₘ**: Non-dimensional pitching moment coefficient

- **𝐶ₙ**: Non-dimensional normal force coefficient, \(𝐶ₙ = -𝐶𝓏\)

##### **Coefficient graph from** `Stevens, B. L., Lewis, F. L., & Johnson, E. N. (2016). Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems (3rd ed.). John Wiley & Sons.`
![C_x](image/cx.png)
![C_z](image/cz.png)
![C_l](image/cl.png)
![C_m](image/cm.png)
![C_n](image/cn.png)

## Validation 
This section outlines the stability validation process for the F-16 under specific control surface configurations. The tests assess the aircraft's stability when all control surfaces are neutral and when each orientation is tested independently along each axis.

## 1. Stability with Neutral Control Surfaces
- **Objective**: Verify the aircraft's inherent stability when all control surfaces are set to zero.
- **Test Conditions**:
  - **δₜₕ (Throttle)**: 0.5 (50% thrust for steady flight)
  - **δₐ (Aileron)**: 0°
  - **δₑ (Elevator)**: 0°
  - **δᵣ (Rudder)**: 0°
- **Expected Outcome**:
  - The aircraft should maintain steady, level flight without external disturbances.
  - Minimal oscillations or divergence from the current orientation.

## 2. Stability Test: Orientation Along Each Axis
- **Objective**: Assess the stability response when control inputs are applied along each axis individually.

### 2.1 Roll Stability (Aileron Input)
- **Test Conditions**:
  - **δₜₕ (Throttle)**: 0.5
  - **δₐ (Aileron)**: ±21.5° (maximum deflection, alternating directions)
  - **δₑ (Elevator)**: 0°
  - **δᵣ (Rudder)**: 0°
- **Expected Outcome**:
  - The aircraft should exhibit roll response proportional to the input.
  - Upon returning δₐ to 0°, the aircraft should stabilize and return to level flight.

### 2.2 Pitch Stability (Elevator Input)
- **Test Conditions**:
  - **δₜₕ (Throttle)**: 0.5
  - **δₐ (Aileron)**: 0°
  - **δₑ (Elevator)**: ±30° (maximum deflection, alternating directions)
  - **δᵣ (Rudder)**: 0°
- **Expected Outcome**:
  - The aircraft should pitch up or down based on the elevator input.
  - Upon returning δₑ to 0°, the aircraft should stabilize to a steady state.

### 2.3 Yaw Stability (Rudder Input)
- **Test Conditions**:
  - **δₜₕ (Throttle)**: 0.5
  - **δₐ (Aileron)**: 0°
  - **δₑ (Elevator)**: 0°
  - **δᵣ (Rudder)**: ±25° (maximum deflection, alternating directions)
- **Expected Outcome**:
  - The aircraft should yaw to the right or left based on rudder input.
  - Upon returning δᵣ to 0°, the aircraft should stabilize without significant oscillations.

### 3. Tee lung ka Stability (Integrate 3 orientation axis)
- ** Test Conditions**:
  - **δₜₕ (Throttle)**: 0.5
  - **δₐ (Aileron)**: ±0°
  - **δₑ (Elevator)**: ±0°
  - **δᵣ (Rudder)**: ±0° 
- **Expected Outcome**:
  - The aircraft should "Tee lung ka" when apply Throttle, Aileron, Elevator, Rudder.
  - Upon returning δₐ, δₑ, δᵣ to 0°, the aircraft should stabilize without significant oscillations.

## Conclusion

Summarize why Out doesn't look like validate graph ==== Aerodynamics !!!

And validate web is dead!!!!!!!!!.


## Acknowledgements
This project is part of the coursework for FRA333 Robot Kinematics at the Institute of Field Robotics, King Mongkut’s University of Technology Thonburi. Special thanks to the course instructors for their guidance and support.

Feel free to explore, modify, and extend this project for educational and research purposes.

## References


## Collabulator
- **Napat Aiamwiratchai**
- **Tadtawan Chaloempornmongkol**
- **Asama Wankra**
- **Aittikit Kitcharoennon**