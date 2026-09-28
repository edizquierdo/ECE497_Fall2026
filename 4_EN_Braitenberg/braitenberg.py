"""
Braitenberg vehicle simulation.

Defines the core building blocks for a light-seeking robot:
  - Point         : base class for any 2-D object with a position.
  - Vehicle       : the Project 1 two-wheeled robot, with hand-wired sensor-to-motor
                    connections ("crossed" or "direct").
  - NeuralVehicle : the same robot, with its wiring replaced by a neural network.
  - Light         : a stationary point light source at the origin (0, 0).

Plus two helpers used by sim.py and evolve.py:
  - place_randomly()      : put a vehicle at a random spot on a circle around the light.
  - simulate_population() : a batched version of the sense -> think -> move loop, used
                            by evolve.py to evaluate a whole population at once.

Two things are different from Project 1:
  1. The light now sits at the origin, and each vehicle starts at a random point
     on a circle of radius `distance` around it, facing a random direction. (In
     Project 1 every vehicle started at the origin, facing a light placed at
     (distance, 0).) Evolution needs this: if every evaluation started from the
     same position and heading, evolution could "solve" phototaxis by memorizing
     one good path instead of learning to steer towards the light.
  2. Vehicle.think() is filled in (both wiring schemes from Project 1), so the
     hand-wired Vehicle is available as a baseline to compare against.
"""

import numpy as np
import torch


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
        wiring:       Sensor-to-motor wiring scheme, "crossed" (default) or "direct".
    """

    def __init__(self, angle_offset=np.pi/2, turn_gain=0.1, noise_stdev=0.1, distance=10,
                 wiring="crossed"):
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
        self.wiring = wiring

        # Sensor positions as Point objects, updated every step
        self.rs = Point()   # right sensor
        self.ls = Point()   # left sensor
        self.update_sensor_pos()

    def update_sensor_pos(self):
        """Recompute the world-frame positions of both sensors from the body pose.

        Angles are measured counterclockwise, so the left sensor sits at
        orientation + angle_offset and the right sensor at orientation - angle_offset.
        """
        self.ls.x_pos = self.x_pos + self.radius * np.cos(self.orientation + self.angle_offset)
        self.ls.y_pos = self.y_pos + self.radius * np.sin(self.orientation + self.angle_offset)
        self.rs.x_pos = self.x_pos + self.radius * np.cos(self.orientation - self.angle_offset)
        self.rs.y_pos = self.y_pos + self.radius * np.sin(self.orientation - self.angle_offset)

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
        """Map sensor activations to motor commands using the hand-designed wiring.

        "crossed" (contralateral): the left sensor drives the right motor and
        the right sensor drives the left motor -- this is the light-seeking
        wiring from Project 1.
        "direct" (ipsilateral): each sensor drives the motor on its own side.

        Note that sensor readings are always in (0, 1], so this vehicle's
        motors are too: it can never drive backwards.
        """
        if self.wiring == "crossed":
            self.right_motor = self.left_sensor
            self.left_motor  = self.right_sensor
        elif self.wiring == "direct":
            self.left_motor  = self.left_sensor
            self.right_motor = self.right_sensor
        else:
            raise ValueError(f"Unknown wiring '{self.wiring}'. Choose 'crossed' or 'direct'.")

    def move(self):
        """Update orientation and velocity, then advance the body position.

        Differential drive kinematics:
          - Turning rate  = turn_gain × (right_motor − left_motor) + noise
          - Forward speed = vel_gain  × average(left_motor, right_motor)

        A faster right wheel turns the vehicle left (counterclockwise, which
        increases orientation); a faster left wheel turns it right.

        Adding noise to the orientation simulates imperfect actuators and
        prevents the vehicle from following a perfectly straight trajectory.
        """
        noise = np.random.normal(0, self.noise_stdev)
        self.orientation += self.turn_gain * (self.right_motor - self.left_motor) + noise
        self.velocity = self.vel_gain * ((self.right_motor + self.left_motor) / 2)
        self.update_body_pos()

    def distance(self, light):
        """Return the Euclidean distance from the vehicle body centre to the light."""
        return euclidean_distance(self, light)


class NeuralVehicle(Vehicle):
    """
    A Vehicle with a neural network controller instead of hand-designed wiring.

    Everything else about the vehicle -- sensing, moving, noise -- is inherited
    unchanged from Vehicle; only think() is replaced. The network's output layer
    uses Tanh, so motor commands are in [-1, 1] (the vehicle *can* drive backwards).

    Args:
        controller: A NeuralController instance (loaded with an evolved genome).
        angle_offset, turn_gain, noise_stdev, distance: As in Vehicle.
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

        self.left_motor = float(motors[0])
        self.right_motor = float(motors[1])


class Light(Point):
    """A stationary point light source at the origin (0, 0)."""

    def __init__(self):
        super().__init__(x_pos=0.0, y_pos=0.0)


def place_randomly(vehicle, distance):
    """Start `vehicle` at a random point `distance` away from the light, facing a random direction."""
    start_angle = np.random.uniform(0, 2 * np.pi)
    vehicle.x_pos = distance * np.cos(start_angle)
    vehicle.y_pos = distance * np.sin(start_angle)
    vehicle.orientation = np.random.uniform(0, 2 * np.pi)
    vehicle.update_sensor_pos()


def simulate_population(motor_fn, pop, episodes, duration=2000, distance=10.0,
                        angle_offset=np.pi / 2, turn_gain=0.1, noise_stdev=0.1):
    """Simulate pop x episodes vehicles at once and return their distances to the light.

    This is the exact same sense -> think -> move loop as Vehicle (and the same
    random starting conditions as place_randomly()), written with tensors so
    that every vehicle in the whole population advances in a single step
    instead of one at a time. evolve.py uses it to evaluate an entire
    generation in one call. If you change the physics in Vehicle (e.g. add
    sensor noise, or a second light), make the same change here.

    Args:
        motor_fn: Callable mapping sensors (pop, episodes, 2) -> motors (pop, episodes, 2),
                  where the last dimension is (left, right). This is the "think" step.
        pop: Number of controllers (individuals).
        episodes: Number of independent vehicles (episodes) per controller.
        duration, distance, angle_offset, turn_gain, noise_stdev: As in Vehicle.

    Returns:
        Tensor of shape (pop, episodes, duration): distance from each vehicle's
        body to the light after every step.
    """
    shape = (pop, episodes)
    radius, vel_gain, sensor_gain = 1.0, 1 / 50, 1 / distance

    # Random starting conditions (same as place_randomly()); light at the origin.
    start_angle = torch.rand(shape) * 2 * np.pi
    x = distance * torch.cos(start_angle)
    y = distance * torch.sin(start_angle)
    orientation = torch.rand(shape) * 2 * np.pi

    dist = torch.empty(pop, episodes, duration)
    for t in range(duration):
        # -- sense (Vehicle.update_sensor_pos + Vehicle.sense) --
        ls_x = x + radius * torch.cos(orientation + angle_offset)
        ls_y = y + radius * torch.sin(orientation + angle_offset)
        rs_x = x + radius * torch.cos(orientation - angle_offset)
        rs_y = y + radius * torch.sin(orientation - angle_offset)
        right_sensor = 1.0 / (1.0 + sensor_gain * torch.sqrt(rs_x**2 + rs_y**2))
        left_sensor  = 1.0 / (1.0 + sensor_gain * torch.sqrt(ls_x**2 + ls_y**2))

        # -- think --
        motors = motor_fn(torch.stack([left_sensor, right_sensor], dim=-1))
        left_motor, right_motor = motors[..., 0], motors[..., 1]

        # -- move (Vehicle.move + Vehicle.update_body_pos) --
        noise = torch.randn(shape) * noise_stdev
        orientation = orientation + turn_gain * (right_motor - left_motor) + noise
        velocity = vel_gain * (left_motor + right_motor) / 2
        x = x + velocity * torch.cos(orientation)
        y = y + velocity * torch.sin(orientation)

        dist[:, :, t] = torch.sqrt(x**2 + y**2)

    return dist
