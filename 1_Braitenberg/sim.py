"""
Braitenberg vehicle simulation runner.

Runs one or more independent simulations of the Braitenberg vehicle and
optionally visualises robot trajectories or distance-to-light over time.

Run as:
    python sim.py
    python sim.py --viztraces
    python sim.py --noise 0.01 --vizdist
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import braitenberg as bt
import argparse


def parse_args():
    """Parse command-line arguments for the Braitenberg simulator."""
    parser = argparse.ArgumentParser(description="Simulate a Braitenberg vehicle.")
    parser.add_argument("--duration", type=int, default=5000,
                        help="Number of simulation steps (default: 5000)")
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of independent simulation runs (default: 10)")
    parser.add_argument("--distance", type=float, default=10,
                        help="Distance from the robot to the light source (default: 10)")
    parser.add_argument("--angle_offset", type=float, default=np.pi/2,  # standardized option name
                        help="Angular separation between the two sensors, in radians (default: pi/2)")
    parser.add_argument("--turn_gain", type=float, default=0.1,  # standardized option name
                        help="Turning sensitivity (default: 0.1)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Amount of random motion noise (default: 0.1)")
    parser.add_argument("--wiring", type=str, default="crossed", choices=["crossed", "direct"],
                        help="Sensor-to-motor wiring scheme (default: crossed). Only has an "
                             "effect once you implement Vehicle.think()'s OPTIONAL runtime switch.")
    parser.add_argument("--viztraces", action='store_true', help="Enable visualization of individual traces")
    parser.add_argument("--vizdist", action='store_true', help="Enable visualization of average distance")
    parser.add_argument("--scores", action='store_true', help="Print avg distance per repetition")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save --viztraces/--vizdist figures to as PNGs, "
                             "instead of opening them in an interactive window. Useful for "
                             "batch-generating figures across many parameter values.")
    return parser.parse_args()


def run_simulation(duration=5000, reps=10, distance=10, angle_offset=np.pi/2,
                   turn_gain=0.1, noise=0.1, wiring="crossed",
                   viztraces=False, vizdist=False, seed=None, save=None):
    """Run the Braitenberg vehicle simulation.

    Args:
        duration: Number of simulation steps
        reps: Number of independent repetitions
        distance: Distance of light source from center
        angle_offset: Angle offset between sensors (radians)
        turn_gain: Gain factor for turning response
        noise: Standard deviation of motion noise
        wiring: Sensor-to-motor wiring scheme ("crossed" or "direct"). Only
                has an effect once you implement Vehicle.think()'s OPTIONAL
                runtime switch -- otherwise Vehicle.think() ignores it.
        viztraces: Enable trajectory visualization
        vizdist: Enable distance-to-light over time visualization
        seed: Random seed for reproducibility
        save: If given, a directory to save --viztraces/--vizdist figures
              to as PNGs instead of opening an interactive window. Handy
              when generating many figures across a parameter sweep.

    Returns:
        Currently returns 0 (placeholder). Part 3 of the assignment asks
        you to replace this with your own fitness value, computed from
        the recorded trajectories / distances below.
    """
    if seed is not None:
        np.random.seed(seed)

    # Variables to store data
    xpos = np.zeros((reps, duration)) if viztraces else None
    ypos = np.zeros((reps, duration)) if viztraces else None
    dist = np.zeros((reps, duration))

    # Initialize light source
    light = bt.Light(distance)

    for r in range(reps):
        # Create new agent for each repetition (pass standardized option names)
        agent = bt.Vehicle(angle_offset, turn_gain, noise, distance, wiring)

        # Run simulation
        for t in range(duration):
            agent.sense(light)
            agent.think()
            agent.move()

            if viztraces:
                xpos[r, t] = agent.x_pos
                ypos[r, t] = agent.y_pos

            dist[r, t] = agent.distance(light)

    # ------------------------------------------------------------------
    # TODO (Part 3 of the assignment): Define your own fitness function.
    #
    # At this point you have access to:
    #   dist     : array of shape (reps, duration) with the distance from
    #              the vehicle to the light at every time step, for every
    #              repetition.
    #   distance : the initial distance between the vehicle and the light.
    #
    # Replace the line below with your own computation (e.g. final
    # distance, closest approach, time-to-target, path length, etc.),
    # averaged appropriately across repetitions, into a single fitness
    # value.
    # ------------------------------------------------------------------
    fitness = 0

    if save is not None:
        os.makedirs(save, exist_ok=True)

    # Handle visualization
    if viztraces:
        plt.figure(figsize=(8, 8))
        for r in range(reps):
            plt.scatter(xpos[r], ypos[r], s=0.5, c=range(duration), cmap="plasma")
        plt.plot(0.0, 0.0, "y^", markersize=10, label="Start")
        plt.plot(light.x_pos, light.y_pos, "ko", markersize=10, label="Light")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.title("Vehicle Trajectories")
        plt.axis('equal')
        plt.tight_layout()
        if save is not None:
            plt.savefig(os.path.join(save, "traces.png"), dpi=150)
            plt.close()
        else:
            plt.show()

    if vizdist:
        avg_fitness_over_time = np.mean(dist, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(avg_fitness_over_time)
        plt.xlabel("Time step")
        plt.ylabel("Average distance from light  (lower = better)")
        plt.title("Fitness Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save is not None:
            plt.savefig(os.path.join(save, "dist.png"), dpi=150)
            plt.close()
        else:
            plt.show()

    return fitness


def main():
    args = parse_args()

    fitness = run_simulation(
        duration=args.duration,
        reps=args.reps,
        distance=args.distance,
        angle_offset=args.angle_offset,
        turn_gain=args.turn_gain,
        noise=args.noise,
        wiring=args.wiring,
        viztraces=args.viztraces,
        vizdist=args.vizdist,
        seed=args.seed,
        save=args.save
    )

    if args.scores:
        print(f"Total fitness: {fitness:.4f}")

    return fitness


if __name__ == "__main__":
    main()
