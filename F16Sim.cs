using UnityEngine;
using System.Collections.Generic;

public class F16Sim : MonoBehaviour
{
    private Plane F_16;

    // Control variables (can be adjusted during simulation)
    public float thrust = 10.0f;
    public float aileron = 0.0f;
    public float elevator = 0.0f;
    public float rudder = 0.0f;
    public GameObject planeModel;

    private float dt;
    private float currentTime = 0.0f;

    private List<Vector3> positions = new List<Vector3>();
    private List<Vector3> velocities = new List<Vector3>();
    private List<Vector3> orientations = new List<Vector3>();
    private List<float> times = new List<float>();

    void Start()
    {
        Dictionary<string, object> initialState = new Dictionary<string, object>()
        {
            {"position", new List<float> { 0.0f, 0.0f, -10.0f } },
            {"velocity", new List<float> { 150.0f, 0.0f, 0.0f } },
            {"orientation", new List<float> { 0.0f, 0.0f, 0.0f, 1.0f } },
            {"angular_velocity", new List<float> { 0.0f, 0.0f, 0.0f } },
            {"thrust", thrust },
            {"control_surfaces", new Dictionary<string, float>() { {"aileron", aileron }, {"elevator", elevator }, {"rudder", rudder } } }
        };

        float mass = 9300.0f;  // kg
        float[,] inertiaTensor = new float[3, 3] { { 12875.0f, 0.0f, 0.0f }, { 0.0f, 75673.6f, 0.0f }, { 0.0f, 0.0f, 85552.1f } };
        float wingArea = 27.87f;  // m^2
        float airDensity = 1.225f;  // kg/m^3 at sea level

        F_16 = new Plane(mass, inertiaTensor, wingArea, airDensity, initialState);

        dt = Time.fixedDeltaTime;
    }

    void FixedUpdate()
    {
        currentTime += dt;

        // Update controls (you can adjust these variables in the Inspector or via UI)
        F_16.SetControls(
            thrust: thrust,
            aileron: aileron,
            elevator: elevator,
            rudder: rudder
        );

        // Update state
        F_16.UpdateState(dt);

        // Record data (optional)
        positions.Add(F_16.GetPosition());
        velocities.Add(F_16.GetVelocity());
        orientations.Add(F_16.GetOrientationEuler());
        times.Add(currentTime);

        // Update the visual representation in the scene
        UpdatePlaneVisual(F_16.GetPosition(), F_16.GetOrientationEuler()
);
    }

    private void UpdatePlaneVisual(Vector3 position, Vector3 eulerAngles)
    {
        Vector3 unityPosition = new Vector3(
          -position.y,          // Unity X = Simulation Y
          -position.z,         // Unity Y = -Simulation Z
          position.x           // Unity Z = Simulation X
      );
        Vector3 unityRotation = new Vector3(
            -eulerAngles.y,          // Unity X = Simulation Y
            -eulerAngles.z,         // Unity Y = -Simulation Z
            eulerAngles.x           // Unity Z = Simulation X
        );
        // Assuming you have a GameObject representing the plane
        // You can set this GameObject via the Inspector
        transform.position = unityPosition;
        transform.rotation = Quaternion.Euler(unityRotation);
    }
    private Quaternion TransformRotationToUnity(Quaternion simulationRotation)
    {
        // Define the rotation from simulation frame to Unity frame
        Quaternion rotationSimToUnity = Quaternion.Euler(0f, 90f, 180f);

        // Apply the coordinate transformation
        Quaternion unityRotation = rotationSimToUnity * simulationRotation;

        return unityRotation;
    }
}
