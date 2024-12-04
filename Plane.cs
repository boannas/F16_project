using UnityEngine;
using System.Collections.Generic;

public class Plane
{
    // Mass and inertia
    private float mass;  // kg
    private float[,] inertiaTensor;  // 3x3 matrix

    // Aerodynamic properties
    private float wingArea;  // m^2
    private float airDensity;  // kg/m^3

    // State variables
    private Vector3 position;  // m
    private Vector3 velocity;  // m/s in body frame
    private Quaternion orientation;  // Quaternion
    private Vector3 angularVelocity;  // rad/s in body frame

    // Control inputs
    private float thrust;  // N
    private Dictionary<string, float> controlSurfaces;  // {'aileron': 0.0, 'elevator': 0.0, 'rudder': 0.0}

    // Data arrays
    private float[,] CX_updated_data_full;
    private float[] CZ_updated_data_full;

    // Constructor
    public Plane(float mass, float[,] inertiaTensor, float wingArea, float airDensity, Dictionary<string, object> initialState)
    {
        // Mass and inertia
        this.mass = mass;
        this.inertiaTensor = inertiaTensor;

        // Aerodynamic properties
        this.wingArea = wingArea;
        this.airDensity = airDensity;

        // State variables
        List<float> positionList = (List<float>)initialState["position"];
        this.position = new Vector3(positionList[0], positionList[1], positionList[2]);

        List<float> velocityList = (List<float>)initialState["velocity"];
        this.velocity = new Vector3(velocityList[0], velocityList[1], velocityList[2]);

        List<float> orientationList = (List<float>)initialState["orientation"];
        this.orientation = new Quaternion(orientationList[0], orientationList[1], orientationList[2], orientationList[3]);

        List<float> angularVelocityList = (List<float>)initialState["angular_velocity"];
        this.angularVelocity = new Vector3(angularVelocityList[0], angularVelocityList[1], angularVelocityList[2]);

        // Thrust
        if (initialState.ContainsKey("thrust"))
            this.thrust = (float)initialState["thrust"];
        else
            this.thrust = 0.0f;

        // Control surfaces
        if (initialState.ContainsKey("control_surfaces"))
            this.controlSurfaces = (Dictionary<string, float>)initialState["control_surfaces"];
        else
            this.controlSurfaces = new Dictionary<string, float>() { { "aileron", 0.0f }, { "elevator", 0.0f }, { "rudder", 0.0f } };

        // Initialize data arrays
        this.CX_updated_data_full = new float[,]
        {
            {-0.099f, -0.081f, -0.081f, -0.063f, -0.025f, 0.044f, 0.097f, 0.113f, 0.145f, 0.167f, 0.174f, 0.166f},
            {-0.048f, -0.038f, -0.040f, -0.021f, 0.016f, 0.083f, 0.127f, 0.137f, 0.162f, 0.177f, 0.179f, 0.167f},
            {-0.022f, -0.020f, -0.021f, -0.004f, 0.032f, 0.094f, 0.128f, 0.130f, 0.154f, 0.161f, 0.155f, 0.138f},
            {-0.040f, -0.038f, -0.039f, -0.025f, 0.006f, 0.062f, 0.087f, 0.085f, 0.100f, 0.110f, 0.104f, 0.091f},
            {-0.083f, -0.073f, -0.076f, -0.072f, -0.046f, 0.012f, 0.024f, 0.025f, 0.043f, 0.053f, 0.047f, 0.040f}
        };

        this.CZ_updated_data_full = new float[]
        {
            0.770f, 0.241f, -0.100f, -0.416f, -0.731f, -1.053f,
            -1.366f, -1.646f, -1.917f, -2.120f, -2.248f, -2.229f
        };
    }

    public float ComputeAngleOfAttack()
    {
        // Transform velocity to inertial frame
        Vector3 v_inertial = orientation * velocity;
        float vx = v_inertial.x;
        float vy = v_inertial.y;
        float vz = v_inertial.z;

        // Compute angle of attack (alpha)
        float alpha = Mathf.Atan2(vz, vx);
        return alpha;
    }

    public float ComputeSideslipAngle()
    {
        // Transform velocity to inertial frame
        Vector3 v_inertial = orientation * velocity;
        float vx = v_inertial.x;
        float vy = v_inertial.y;
        float vz = v_inertial.z;

        // Compute sideslip angle (beta)
        float beta = Mathf.Atan2(vy, Mathf.Sqrt(vx * vx + vz * vz));
        return beta;
    }

    public float ComputeCxFixed(float el)
    {
        float alpha = ComputeAngleOfAttack();
        float beta = ComputeSideslipAngle();

        // Map Alpha and EL into indices and interpolate
        float[,] data = CX_updated_data_full;
        float scaled_alpha = 0.2f * alpha;
        int k = Mathf.FloorToInt(scaled_alpha);
        k = Mathf.Clamp(k, -2, 8);  // Constrain within bounds
        float da = scaled_alpha - k;
        int l = k + (int)Mathf.Sign(1.1f * da);
        l = Mathf.Clamp(l, -2, 8);  // Constrain within bounds

        int k_index = k + 2;  // Adjust index for array (0-based)
        int l_index = l + 2;  // Adjust index for array (0-based)

        float scaled_el = el / 12.5f;
        int m = Mathf.FloorToInt(scaled_el);
        m = Mathf.Clamp(m, -2, 2);  // Constrain within bounds
        float de = scaled_el - m;
        int n = m + (int)Mathf.Sign(1.1f * de);
        n = Mathf.Clamp(n, -2, 2);  // Constrain within bounds

        int m_index = m + 2;  // Adjust index for array (0-based)
        int n_index = n + 2;  // Adjust index for array (0-based)

        // Interpolate values
        float t = data[m_index, k_index];
        float u = data[n_index, k_index];
        float v = t + Mathf.Abs(da) * (data[m_index, l_index] - t);
        float w = u + Mathf.Abs(da) * (data[n_index, l_index] - u);

        return v + (w - v) * Mathf.Abs(de);
    }

    public float ComputeCy(float AIL, float RDR)
    {
        float beta = ComputeSideslipAngle();
        float c_y = (-0.02f) * beta + (0.021f) * (AIL / 20.0f) + (0.086f) * (RDR / 30.0f);
        return c_y;
    }

    public float ComputeCz(float el)
    {
        float alpha = ComputeAngleOfAttack();
        float beta = ComputeSideslipAngle();
        float[] data = CZ_updated_data_full;

        // Map Alpha into indices and interpolate
        float scaled_alpha = 0.2f * alpha;
        int k = Mathf.FloorToInt(scaled_alpha);
        k = Mathf.Clamp(k, -2, 8);  // Constrain within bounds
        float da = scaled_alpha - k;
        int l = k + (int)Mathf.Sign(1.1f * da);
        l = Mathf.Clamp(l, -2, 8);  // Constrain within bounds

        int k_index = k + 2;  // Adjust index for array (0-based)
        int l_index = l + 2;  // Adjust index for array (0-based)

        float s = data[k_index] + Mathf.Abs(da) * (data[l_index] - data[k_index]);
        return s * (1 - Mathf.Pow(beta / 57.3f, 2)) - 0.19f * (el / 25.0f);
    }

    public Vector3 ComputeAerodynamicForces()
    {
        float Cx = ComputeCxFixed(controlSurfaces["elevator"]);
        float Cz = ComputeCz(controlSurfaces["elevator"]);
        float Cy = ComputeCy(controlSurfaces["aileron"], controlSurfaces["rudder"]);
        float airspeed = velocity.magnitude;
        float q = 0.5f * airDensity * airspeed * airspeed;

        Vector3 aerodynamic_force = new Vector3(
            q * Cx * wingArea,
            q * Cy * wingArea,
            q * Cz * wingArea
        );
        Vector3 thrust_force = new Vector3(thrust, 0f, 0f);

        Vector3 total_force = aerodynamic_force + thrust_force;
        return total_force;
    }

    public Vector3 ComputeAerodynamicMoments()
    {
        float aileron = controlSurfaces["aileron"];
        float elevator = controlSurfaces["elevator"];
        float rudder = controlSurfaces["rudder"];

        Vector3 moments = new Vector3(aileron, elevator, rudder);

        Vector3 effectiveness = GetControlEffectiveness();

        moments = moments * Mathf.Deg2Rad;
        moments = new Vector3(moments.x * effectiveness.x, moments.y * effectiveness.y, moments.z * effectiveness.z);

        return moments;
    }

    public Vector3 GetControlEffectiveness()
    {
        // Placeholder for control effectiveness coefficients
        return new Vector3(4000.0f, 4000.0f, 4000.0f);  // Adjust as needed
    }

    public void UpdateState(float dt)
    {
        // Compute forces and moments
        Vector3 gravity_inertial = new Vector3(0.0f, 0.0f, 9.81f * mass);  // Gravity acts in the positive Z-direction in inertial frame

        // Transform gravity to body frame
        Vector3 gravity_body = Quaternion.Inverse(orientation) * gravity_inertial;

        // Compute aerodynamic forces
        Vector3 forces_body = ComputeAerodynamicForces();

        // Add gravity to forces in the body frame
        Vector3 total_forces_body = forces_body + gravity_body;
        Vector3 moments_body = ComputeAerodynamicMoments();
        float L = moments_body.x;  // Roll moment
        float M = moments_body.y;  // Pitch moment
        float N = moments_body.z;  // Yaw moment

        // Linear acceleration in body frame
        Vector3 acceleration_body = total_forces_body / mass;

        // Update velocities
        velocity += acceleration_body * dt;

        // Update positions (convert velocity to inertial frame)
        Vector3 velocity_inertial = orientation * velocity;
        position += velocity_inertial * dt;

        // Compute Γ (Gamma)
        float I_xx = inertiaTensor[0, 0];
        float I_yy = inertiaTensor[1, 1];
        float I_zz = inertiaTensor[2, 2];
        float I_xz = inertiaTensor[0, 2];
        float Gamma = I_xx * I_zz - I_xz * I_xz;

        // Angular accelerations (P_dot, Q_dot, R_dot)
        float P = angularVelocity.x;
        float Q = angularVelocity.y;
        float R = angularVelocity.z;

        float P_dot = (
            (I_zz * L + I_xz * N - ((I_zz * (I_zz - I_yy) + I_xz * I_xz) * Q * R) + I_xx * (I_yy - I_zz + I_xz) * P * Q) / Gamma
        );
        float Q_dot = (
            (M + (I_zz - I_xx) * P * R - I_xz * (P * P - R * R)) / I_yy
        );
        float R_dot = (
            (I_xx * N + I_xz * L + ((I_xx * (I_xx - I_yy) + I_xz * I_xz) * P * Q) - I_xz * (I_yy - I_xx + I_xz) * Q * R) / Gamma
        );

        // Combine angular accelerations
        Vector3 angular_acceleration = new Vector3(P_dot, Q_dot, R_dot);

        // Update angular velocities
        angularVelocity += angular_acceleration * dt;

        // Update orientation
        Vector3 rotationVector = angularVelocity * dt;
        float angle = rotationVector.magnitude;
        Quaternion delta_orientation;
        if (angle != 0f)
        {
            Vector3 axis = rotationVector / angle;  // Normalized
            delta_orientation = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
        }
        else
        {
            delta_orientation = Quaternion.identity;
        }
        orientation = delta_orientation * orientation;
    }

    public void SetControls(float thrust, float aileron, float elevator, float rudder)
    {
        this.thrust = thrust;
        this.controlSurfaces = new Dictionary<string, float>()
        {
            {"aileron", aileron},
            {"elevator", elevator},
            {"rudder", rudder}
        };
    }

    // Accessor methods to retrieve state variables
    public Vector3 GetPosition()
    {
        return position;
    }

    public Vector3 GetVelocity()
    {
        return velocity;
    }

    public Vector3 GetOrientationEuler()
    {
        // Convert orientation quaternion to Euler angles (degrees)
        return orientation.eulerAngles;
    }
    public Quaternion GetOrientationQuaternion()
    {
        return orientation;
    }
}
