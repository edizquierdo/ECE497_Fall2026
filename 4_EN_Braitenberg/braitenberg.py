"""
Braitenberg vehicle simulation.

Defines the core building blocks for a light-seeking robot:
  - Point      : base class for any 2-D object with a position.
  - Vehicle    : a two-wheeled robot with two light sensors and a differential drive.
  - Light      : a stationary point light source at the origin (0, 0).

The vehicle uses a crossed (contralateral) wiring scheme: the left sensor
drives the right motor and the right sensor drives the left motor.  This
causes the vehicle to steer towards the brighter side, producing light-seeking
behavior from an extremely simple controller.
"""

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # Type hint fallback


def euclidean_distance(point1, point2):
    """Return the Euclidean distance between two Point objects."""
    return np.sqrt((point1.x_pos - point2.x_pos)**2 + (point1.y_pos - point2.y_pos)**2)


class Point:
    """A 2-D object with an (x, y) position."""

    def __init__(self, x_pos=0.0, y_pos=0.0):
        self.x_pos = x_pos
        self.y_pos = y_pos


class Vehicle(Point):
    """A Braitenberg vehicle with two light sensors and differential drive motors.

    The vehicle body is modelled as a circle of given radius.  Two sensors are
    mounted on the perimeter, symmetrically offset from the forward direction
    by ±angle_offset radians.

    Args:
        angle_offset: Angular separation between each sensor and the vehicle's
                      forward axis (radians).  π/2 places sensors at 90° on
                      each side; smaller values move them closer to the front.
        turn_gain:    How strongly the sensor difference steers the vehicle.
                      Larger values produce sharper turns.
        noise_stdev:  Standard deviation of Gaussian noise added to the
                      orientation at each step.  Zero means no noise.
        distance:     Initial distance to the light source; used to scale the
                      sensor gain so that raw sensor values stay near [0, 1].
    """

    def __init__(self, angle_offset=np.pi/2, turn_gain=0.1, noise_stdev=0.1, distance=10):
        super().__init__()
        self.orientation = 0.0          # heading angle (radians); starts pointing along +x
        self.velocity = 0.0             # forward speed (arbitrary units)
        self.radius = 1.0               # body radius; also the sensor arm length

        self.left_sensor  = 0.0        # current left sensor activation  [0, 1]
        self.right_sensor = 0.0        # current right sensor activation [0, 1]
        self.left_motor   = 0.0        # left motor command
        self.right_motor  = 0.0        # right motor command

        # Sensor gain: normalise raw distance so that a sensor at `distance`
        # from the light reads approximately 0.5 (mid-range).
        self.sensor_gain = 1 / distance

        self.turn_gain   = turn_gain
        self.vel_gain    = 1 / 50       # converts average motor output to forward speed
        self.noise_stdev = noise_stdev
        self.angle_offset = angle_offset

        # Sensor positions as Point objects, updated every step
        self.rs = Point()   # right sensor
        self.ls = Point()   # left sensor
        self.update_sensor_pos()

    def update_sensor_pos(self):
        """Recompute the world-frame positions of both sensors from the body pose."""
        self.rs.x_pos = self.x_pos + self.radius * np.cos(self.orientation + self.angle_offset)
        self.rs.y_pos = self.y_pos + self.radius * np.sin(self.orientation + self.angle_offset)
        self.ls.x_pos = self.x_pos + self.radius * np.cos(self.orientation - self.angle_offset)
        self.ls.y_pos = self.y_pos + self.radius * np.sin(self.orientation - self.angle_offset)

    def update_body_pos(self):
        """Advance the body one step along the current heading at the current velocity."""
        self.x_pos += self.velocity * np.cos(self.orientation)
        self.y_pos += self.velocity * np.sin(self.orientation)
        self.update_sensor_pos()

    def sense(self, light):
        """Read light intensity at each sensor position.

        Intensity follows an inverse-distance law: it is 1.0 when the sensor
        is at the light source and approaches 0 as distance grows.

        Args:
            light: A Light (or any Point) object representing the light source.
        """
        distance_right = self.sensor_gain * euclidean_distance(self.rs, light)
        distance_left  = self.sensor_gain * euclidean_distance(self.ls, light)
        self.right_sensor = 1.0 / (1.0 + distance_right)
        self.left_sensor  = 1.0 / (1.0 + distance_left)

    def think(self):
        """Map sensor activations to motor commands using crossed (contralateral) wiring.

        Contralateral connection: the left sensor drives the right motor and
        the right sensor drives the left motor.  When the light is to the
        right, the right sensor is stronger → the left motor speeds up →
        the vehicle turns right, towards the light.
        """
        self.right_motor = self.left_sensor
        self.left_motor  = self.right_sensor

    def move(self):
        """Update orientation and velocity, then advance the body position.

        Differential drive kinematics:
          - Turning rate  = turn_gain × (left_motor − right_motor) + noise
          - Forward speed = vel_gain  × average(left_motor, right_motor)

        Adding noise to the orientation simulates imperfect actuators and
        prevents the vehicle from following a perfectly straight trajectory.
        """
        noise = np.random.normal(0, self.noise_stdev)
        self.orientation += self.turn_gain * (self.left_motor - self.right_motor) + noise
        self.velocity = self.vel_gain * ((self.right_motor + self.left_motor) / 2)
        self.update_body_pos()

    def distance(self, light):
        """Return the Euclidean distance from the vehicle body centre to the light."""
        return euclidean_distance(self, light)


class Light(Point):
    """A stationary point light source at the origin (0, 0)."""

    def __init__(self):
        # Light is always at origin (0, 0)
        super().__init__(x_pos=0.0, y_pos=0.0)


class NeuralVehicle(Vehicle):
    """
    A Vehicle with a neural network controller instead of direct sensor-motor wiring.

    This class extends the base Vehicle with a neural controller that computes
    motor commands from sensor readings. The output layer uses Tanh to bound
    motors to [-1, 1].

    Args:
        controller: A NeuralController instance (loaded with evolved genome).
        angle_offset: Angular separation between sensors (radians).
        turn_gain: How strongly the sensor difference steers the vehicle.
        noise_stdev: Standard deviation of motion noise.
        distance: Distance to light source for sensor gain calculation.
    """

    def __init__(self, controller, angle_offset=np.pi / 2, turn_gain=0.1, noise_stdev=0.1, distance=10):
        super().__init__(angle_offset=angle_offset, turn_gain=turn_gain, noise_stdev=noise_stdev, distance=distance)
        self.controller = controller

    def think(self):
        """Use the neural network controller to compute motor commands."""
        sensors_tensor = torch.tensor(
            [self.left_sensor, self.right_sensor],
            dtype=torch.float32,
        )

        with torch.no_grad():
            motors = self.controller(sensors_tensor).numpy()

        # Output is already bounded to [-1, 1] by Tanh in the network
        self.left_motor = float(motors[0])
        self.right_motor = float(motors[1])