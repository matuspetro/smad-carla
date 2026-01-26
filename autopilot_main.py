#from carla_env import ActorSpawner
from carla_env import CarlaEnv

import traceback
import time


def main():
    """Main function to run the CARLA environment with a moving vehicle."""
    try:
        # Create the environment
        env = CarlaEnv()
        env.reset()
        
        logger = env.logger
        
        # Enable autopilot for the vehicle
        env.set_autopilot(True) #TODO ZATIAL AUTOPILOT NENASLEDUJE ROUTE
        
        # Get the spectator
        spectator = env.world.get_spectator()
        
        # Camera main loop for the car
        while env.vehicle is not None:
            spectator.set_transform(env.get_spectator_following_transform())
            # In your main loop or step method:
            # Inside your control loop or main function
            current_wp, distance, next_wp = env.track_waypoints(buffer_distance=2.0)
            alignment = env.calculate_waypoint_alignment(current_wp, next_wp)

            # For logging or display
            alignment_percent = (alignment + 1) / 2 * 100  # Convert [-1,1] to [0,100]
            print(f"Vehicle alignment with route: {alignment_percent:.1f}%")

            # For reward calculation in reinforcement learning
            reward = alignment * 1  # Higher reward when aligned well
            print(f"Reward: {reward:.2f}")
            # Log or use the information
            if current_wp:
                print(f"Distance to waypoint: {distance:.2f}m")
            time.sleep(0.05) # TODO AK VAM TOTO BLBNE KVOLI PERFORMANCE TAK SI DAJTE VACSI DELAY
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, exiting...")    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        logger.error(traceback.format_exc())
    finally:
        logger.info("Exiting environment...")


if __name__ == "__main__":
    main()