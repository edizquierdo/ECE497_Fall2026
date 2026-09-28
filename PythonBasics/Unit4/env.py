##################################################################################
# Braitenberg vehicle and light source for the embodied task.
# Same body as in Project 4, written in plain Python/NumPy:
#   - The light sits at the origin (0, 0).
#   - The vehicle starts at a random point on a circle of radius `distance`
#     around the light, facing a random direction.
#   - The vehicle's "brain" is a neural network (fnn.FNN) that maps the two
#     light sensors to the two motors.
##################################################################################
import numpy as np

class Point2D:

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def align(self, other):
        self.x = other.x
        self.y = other.y

    def move(self, magnitude, angle):
        self.x += magnitude * np.cos(angle)
        self.y += magnitude * np.sin(angle)

    def dist(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Light:

    def __init__(self):
        self.pos = Point2D(0.0, 0.0)                            # light always sits at the origin

class Braitenberg:

    def __init__(self, neuralnetwork, distance):
        self.radius = 1.0                                       # size/radius of the vehicle's body
        self.angleoffset = np.pi/2                              # sensors placed at +/- 90 degrees from the heading
        self.sensorgain = 1/distance                            # so that sensors read ~0.5 at the starting distance
        self.turngain = 0.5                                     # how strongly the motor difference turns the vehicle
        self.velgain = 0.1                                      # converts average motor output into forward speed
        self.noise = 0.1                                        # std of the noise added to the orientation each step
        self.sensors = [0.0, 0.0]                               # left and right sensor values
        self.motors = [0.0, 0.0]                                # left and right motor values
        self.controller = neuralnetwork

        # Start at a random point on a circle around the light, facing a random direction
        startangle = np.random.random()*2*np.pi
        self.pos = Point2D()
        self.pos.move(distance, startangle)
        self.orientation = np.random.random()*2*np.pi
        self.velocity = 0.0
        self.ls_pos = Point2D()
        self.rs_pos = Point2D()
        self.update_sensor_pos()

    def update_sensor_pos(self):
        # Angles are measured counterclockwise, so the left sensor is at +offset and the right one at -offset
        self.ls_pos.align(self.pos)
        self.ls_pos.move(self.radius, self.orientation + self.angleoffset)
        self.rs_pos.align(self.pos)
        self.rs_pos.move(self.radius, self.orientation - self.angleoffset)

    def sense(self, light):
        # Light intensity is 1 at the light source and decays towards 0 with distance
        self.sensors[0] = 1/(1 + self.sensorgain * self.ls_pos.dist(light.pos))     # Left sensor
        self.sensors[1] = 1/(1 + self.sensorgain * self.rs_pos.dist(light.pos))     # Right sensor

    def think(self):
        # The neural network maps [left sensor, right sensor] to [left motor, right motor]
        self.motors = self.controller.forward(self.sensors)[0]

    def move(self):
        # A faster right wheel turns the vehicle left (counterclockwise), and vice versa
        self.orientation += self.turngain * (self.motors[1] - self.motors[0]) + np.random.normal(0, self.noise)
        self.velocity = self.velgain * (self.motors[0] + self.motors[1])/2

        # Update position of the body and of the sensors
        self.pos.move(self.velocity, self.orientation)
        self.update_sensor_pos()

    def distance(self, light):
        return self.pos.dist(light.pos)
