import random
import time
import numpy as np
import logging
import math 
import cv2
import gymnasium
from gymnasium import spaces
import carla
from tensorflow.keras.models import load_model
from agents.navigation.global_route_planner import GlobalRoutePlanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s --> %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


SECONDS_PER_EPISODE = 20

N_CHANNELS = 3
HEIGHT = 128
WIDTH = 256

SPIN = 3

SHOW_PREVIEW = True

SEED = 123

class CarEnv(gymnasium.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = WIDTH
	im_height = HEIGHT
	front_camera = None
	CAMERA_POS_Z = 1.3 
	CAMERA_POS_X = 1.4
	PREFERRED_SPEED = 30 # what it says
	SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the
	
	
	def __init__(self):
		super(CarEnv, self).__init__()
		#self.action_space = spaces.MultiDiscrete([5, 4])
		self.action_space = spaces.Box(
    		low=np.array([-0.7, 0.0]),  # [steer_min, throttle_min]
        	high=np.array([0.7, 1.0]),  # [steer_max, throttle_max]
        	dtype=np.float32
    		)
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_height, self.im_width, N_CHANNELS), dtype=np.uint8)

		# Connect to the CARLA server
		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(15.0)

		# Load the desired map (Town02)
		logging.info("Loading map: Town02")
		self.world = self.client.load_world("Town02")
		#self.world = self.client.get_world()
		self.curves = get_curves(self.world, self.world.get_map())
		# return 0

		# Apply world settings
		self.settings = self.world.get_settings()
		self.world.apply_settings(carla.WorldSettings(
			no_rendering_mode=False,
			synchronous_mode=True,
			fixed_delta_seconds=0.05
		))

		# Initialize other attributes
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
		self.spectator = self.world.get_spectator()
		self.start_waypoint = None
		self.end_waypoint = None
		self.current_route = None
		self.vehicle = None
		self.actor_list = []
		self.map = self.world.get_map()  # Initialize the map attribute
		self.start_spawn = 0

	def cleanup(self):
		"""Destroy all actors with improved cleanup."""
		logging.info("Destroying ego vehicle")
		if self.vehicle is not None:
			if self.vehicle.is_alive:
				self.vehicle.destroy()
			self._last_vehicle_id = None
		logging.info("Destroying all actors")
		for actor in self.actor_list:
			try:
				if actor is not None and actor.is_alive:
					actor.destroy()
					logging.info(f"Destroyed actor {actor.type_id} with ID {actor.id}")
			except Exception as e:
				logging.error(f"Error during cleanup: {e}")
		logging.info("Destroyed all actors")
		self.actor_list = []
		self.vehicle = None
		self.followed_vehicle = None
		time.sleep(0.1)  # Allow time for cleanup
		self.world.tick()  # Ensure the world is updated after cleanup
		logging.info("All actors destroyed successfully")
	
	def maintain_speed(self,s):
			''' 
			this is a very simple function to maintan desired speed
			s arg is actual current speed
			'''
			if s >= self.PREFERRED_SPEED:
				return 0
			elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
				return 0.7 # think of it as % of "full gas"
			else:
				return 0.3 # tweak this if the car is way over or under preferred speed 

	def step(self, action):
		self.world.tick()

		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

		trans = self.vehicle.get_transform()
		self.spectator.set_transform(
			carla.Transform(trans.location + carla.Location(z=30), carla.Rotation(yaw=-180, pitch=-90))
		)
		
		self.step_counter += 1

		#steer_idx = int(action[0])
		#throttle_idx = int(action[1])

		self.steer = float(action[0])  # Already in range [-1.0, 1.0]
		self.throttle = float(action[1])  # Already in range [0.0, 1.0]
    

		#steer_values = [-0.65, -0.15, 0.0, 0.15, 0.65]
		#throttle_values = [0.0, 0.1, 0.3, 0.6]
		# Extract steering and throttle from continuous action vector
		#self.steer = steer_values[steer_idx]  # Already in range [-1.0, 1.0]
		#self.throttle = throttle_values[throttle_idx]  # Already in range [0.0, 1.0]
		
		# Apply braking logic - brake when throttle is very low
		self.brakes = 1.0 #if self.throttle < 0.05 else 0.0

		# Apply control
		self.vehicle.apply_control(
			carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=self.brakes)
		)

		reward = 0
		done = False

		# Calculate distances and heading
		vehicle_location = self.vehicle.get_location()
		waypoint = self.world.get_map().get_waypoint(
			vehicle_location,
			project_to_road=True,
			lane_type=carla.LaneType.Driving
		)
		lane_center = waypoint.transform.location
		lateral_distance = vehicle_location.distance(lane_center)

		cos_yaw_diff, _, _ = get_reward_comp(self.vehicle, waypoint, None)

		# ====== REWARD SYSTEM ======


		# # ====== FÁZA 3: Jemné brúsenie správania ======

		# # Check if there is any actor in front of the agent
		# actors_in_front = self.world.get_actors().filter('vehicle.*')
		# agent_forward_vector = self.vehicle.get_transform().get_forward_vector()

		# # Reset vehicle_in_front state
		# self.vehicle_in_front = False

		# # # Check if we're in a curve for more lenient angle thresholds
		# in_curve = False
		# for curve in self.curves:
		# 	if vehicle_location.distance(curve.transform.location) < 15:  # Increased buffer for curves
		# 		in_curve = True
		# 		break

		# # Direction threshold - more lenient in curves
		# direction_threshold = 0.5 if in_curve else 0.7

		# for actor in actors_in_front:
		# 	if actor.id != self.vehicle.id:  # Exclude the agent itself
		# 		actor_location = actor.get_location()
		# 		actor_heading_vector = actor.get_transform().get_forward_vector()
		# 		direction_to_actor = actor_location - vehicle_location
		# 		distance_to_actor = math.sqrt(direction_to_actor.x**2 + direction_to_actor.y**2 + direction_to_actor.z**2)

		# 		# Skip if too far
		# 		if distance_to_actor > 20.0:  # Only process vehicles within reasonable distance
		# 			continue

		# 		# Normalize direction vector
		# 		direction_length = math.sqrt(direction_to_actor.x**2 + direction_to_actor.y**2 + direction_to_actor.z**2)
		# 		if direction_length > 0:
		# 			normalized_direction = carla.Vector3D(
		# 				direction_to_actor.x / direction_length,
		# 				direction_to_actor.y / direction_length,
		# 				direction_to_actor.z / direction_length
		# 			)
		# 		else:
		# 			continue  # Skip if vectors are too close

		# 		# 1. Check if the actor is in front (dot product with forward vector)
		# 		in_front_score = (agent_forward_vector.x * normalized_direction.x + 
		# 						agent_forward_vector.y * normalized_direction.y + 
		# 						agent_forward_vector.z * normalized_direction.z)
				
		# 		# 2. Check if heading in same direction (dot product of forward vectors)
		# 		same_direction_score = (agent_forward_vector.x * actor_heading_vector.x + 
		# 								agent_forward_vector.y * actor_heading_vector.y + 
		# 								agent_forward_vector.z * actor_heading_vector.z)
						
		# 		# Only consider vehicles in front AND going same direction
		# 		if in_front_score > 0.6 and same_direction_score > direction_threshold:
		# 			# Check if the actor is in the same lane as the agent
		# 			actor_waypoint = self.world.get_map().get_waypoint(
		# 				actor_location,
		# 				project_to_road=True,
		# 				lane_type=carla.LaneType.Driving
		# 			)
		# 			agent_waypoint = self.world.get_map().get_waypoint(
		# 				vehicle_location,
		# 				project_to_road=True,
		# 				lane_type=carla.LaneType.Driving
		# 			)
					
		# 			# In curves, be more lenient with lane checking
		# 			if in_curve or actor_waypoint.lane_id == agent_waypoint.lane_id:
		# 				if distance_to_actor < (15.0 if in_curve else 10.0):  # Greater distance threshold in curves
		# 					v_actor = actor.get_velocity()
		# 					kmh_actor = int(3.6 * math.sqrt(v_actor.x**2 + v_actor.y**2 + v_actor.z**2))
							
		# 					print(f"Vehicle ahead: distance={distance_to_actor:.1f}m, direction_match={same_direction_score:.2f}")
								
		# 					if kmh_actor < kmh:
		# 						print("\033[91mSlower vehicle detected in front!\033[0m")
		# 						reward -= 5.0
		# 					else:
		# 						print("\033[92mFaster vehicle detected in front!\033[0m")
		# 						reward += 8.0
		# 					self.vehicle_in_front = True
		# 					break  # Found a relevant vehicle, no need to check others
		# 				else:
		# 					self.vehicle_in_front = False  # Reset if no relevant vehicle found


		in_curve = False

		for curve in self.curves:
			if vehicle_location.distance(curve.transform.location) < 10:
				in_curve = True
				break

		if not in_curve:
			# Laterálna penalizácia
			if lateral_distance > 2.4:
				reward -= 50
				done = True
				print("\033[91mOff-road detected, episode ends!\033[0m")
			# elif lateral_distance > 1.5:
			# 	reward -= 2.0
			# elif lateral_distance > 1.2:
			# 	reward -= 1.5
			# elif lateral_distance > 0.8:
			# 	reward -= 1.0
			# elif lateral_distance > 0.5:
			# 	reward -= 0.5
			# else:
			# 	reward += 2

			# Heading zarovnanie (cos_yaw_diff)
			# reward += 2 * cos_yaw_diff

			# Penalizácia prudkého steeringu
			

			# # Odmena za rýchlosť
			# if 5 <= kmh <= 20:
			# 	reward += 2
			# elif kmh < 5:
			# 	reward -= 5
			# elif kmh > 17:
			# 	reward -= 2
			
		# else:
			# if lateral_distance > 2.2:
			# 	reward -= -4.0
			# elif lateral_distance > 2.0:
			# 	reward -= 3.0
			# elif lateral_distance > 1.7:
			# 	reward -= 2.0
			# elif lateral_distance > 1.3:
			# 	reward -= 1.5
			# elif lateral_distance > 1.0:
			# 	reward -= 1.0
			# else:
			# 	reward += 2.0
			
			# # Odmena za rýchlosť
			# if 5 <= kmh <= 20:
			# 	reward += 2
			# elif kmh < 5:
			# 	reward -= 5
			# elif kmh > 17:
			# 	reward -= 2

		if abs(self.steer) > 0.6:
			reward -= abs(self.steer) * 10
		else:
			reward += 100

		# Kolízia - veľký trest
		if len(self.collision_hist) != 0:
			if kmh > 0:
				reward -= 100
			done = True
			print("\033[91mCollision detected!\033[0m")

		# Distance traveled reward
		distance_travelled = self.initial_location.distance(vehicle_location)
		if distance_travelled > (self.distance_flag + 1) * 10:
			self.distance_flag += 1
			# reward += 20
			# print("\033[92mDistance milestone reached!\033[0m")

		if self.distance_flag > 10:
			done = True
			reward += 50
			print("\033[92mDistance limit reached!\033[0m")

		

		# Time limit
		truncated = False
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			print("\033[91mEpisode time limit reached!\033[0m")
			truncated = True

		# Logging every 50 steps
		if self.step_counter % 20 == 0:
			print(' ')
			print("\033[93m   - reward:", reward, "\033[0m")
			print('   - actionSpace:', action)
			# print('   - speed:', kmh)
			# print('   - steer:', self.steer)
			# print('   - traveled:', distance_travelled)
			# print('   - distance:', lateral_distance)
			# print('   - heading:', cos_yaw_diff)
			# print('   - in curve:', in_curve)

		terminated = done

		# reward = reward / 100.0

		return self.front_camera, reward, terminated, truncated, {}

	def reset(self, seed=SEED):
		logging.info("Reset method started.")
		self.cleanup()
		self.start_waypoint = None
		self.end_waypoint = None
		self.current_route = None
		self._prev_waypoint_index = 0
		self.current_waypoint_index = 0
		self.collision_hist = []
		self.lane_invade_hist = []
		self.actor_list = []
		self.steer = 0.0
		self.throttle = 0.0
		self.brakes = 0.0
		self.spawn_transform = self._generate_track()
		self.count_line_hits = 0
		self.distance_flag = 0
		self.vehicle_in_front = False

		self.vehicle = None
		while self.vehicle is None:
			try:
				logging.info("Spawning vehicle...")
				self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_transform)
				self._last_vehicle_id = self.vehicle.id
				self.world.tick()
			except Exception as e:
				logging.error(f"Failed to spawn vehicle: {e}")
		self.actor_list.append(self.vehicle)
		logging.info("Vehicle spawned successfully.")
		self.initial_location = self.vehicle.get_location()

		# Spawn traffic
		logging.info("Spawning traffic...")
		# self.actor_list.extend(self.spawn_car_in_front(distance=12, autopilot=True))
		#self.actor_list.extend(self.spawn_traffic(count=80, autopilot=True,retries=70))
		logging.info("Traffic spawned successfully.")

		# Camera and sensors setup
		logging.info("Initializing sensors...")
		self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.sem_cam.set_attribute("fov", f"90")

		camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))
		self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
		self.world.tick()
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))
		logging.info("Camera sensor spawned successfully.")

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		self.world.tick()
		angle_adj = random.randrange(-SPIN, SPIN, 1)
		trans = self.vehicle.get_transform()
		trans.rotation.yaw = trans.rotation.yaw + angle_adj
		self.vehicle.set_transform(trans)

		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
		self.world.tick()
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))
		logging.info("Collision sensor spawned successfully.")

		lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
		self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.lanesensor)
		self.lanesensor.listen(lambda event: self.lane_data(event))
		logging.info("Lane invasion sensor spawned successfully.")

		# Wait for front camera data with timeout
		logging.info("Waiting for front camera data...")
		start_time = time.time()
		while self.front_camera is None:
			self.world.tick()
			if time.time() - start_time > 10:  # Timeout after 10 seconds
				logging.error("Timeout waiting for front camera data.")
				break

		self.episode_start = time.time()
		self.steering_lock = False
		self.steering_lock_start = None
		self.step_counter = 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, brake=self.brakes))
		return self.front_camera, {}  # Return observation and an empty dictionary

	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)

		img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
		img_array = img_array.reshape((self.im_height, self.im_width, 4))[:, :, :3].copy()
		# img_array[(img_array[:, :, 0] == 128) & (img_array[:, :, 1] == 64) & (img_array[:, :, 2] == 128)] = [255, 255, 255]  # road

		# mask = np.logical_or(
		# 	np.all(img_array == [255, 255, 255], axis=-1),
		# 	np.all(img_array == [244, 35, 232], axis=-1),
		# )
		# img_array[~mask] = [0, 0, 0]

		# Convert to grayscale (1 channel)
		# gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

		# Save the image that the agent will see
		self.front_camera = img_array

		# Optional display
		# if self.step_counter % 10 == 0:
		# 	cv2.imshow("Car Camera View", self.front_camera)
		# 	cv2.waitKey(5)

	def collision_data(self, event):
		self.collision_hist.append(event)
		
	def lane_data(self, event):
		"""
		Handle lane invasion events and filter out false positives.
		"""
		# Get the vehicle's current location
		vehicle_location = self.vehicle.get_location()

		# Get the closest waypoint to the vehicle
		waypoint = self.world.get_map().get_waypoint(
			vehicle_location,
			project_to_road=True,
			lane_type=carla.LaneType.Driving  # Only consider driving lanes
		)

		# Calculate the lateral distance from the lane center
		lane_center = waypoint.transform.location
		lateral_distance = vehicle_location.distance(lane_center)

		# Define a threshold for valid lane invasions (e.g., 1.5 meters)
		lane_crossing_threshold = 0

		# Only log lane invasions if the lateral distance exceeds the threshold
		if lateral_distance > lane_crossing_threshold:
			self.lane_invade_hist.append(event)

	def calculate_waypoint_alignment(self, current_waypoint=None, next_waypoint=None):
		"""
		Calculate the cosine of the angle between vehicle's forward vector and the direction to the next waypoint.
		
		Returns:
			float: Cosine of the angle difference:
				1.0 = perfectly aligned
				0.0 = perpendicular 
				-1.0 = opposite direction
		"""
		if not self.vehicle:
			return 0.0
		
		# Use results from track_waypoints if waypoints not provided
		if current_waypoint is None or next_waypoint is None:
			current_waypoint, _, next_waypoint,_ = self.track_waypoints()
			if not current_waypoint or not next_waypoint:
				return 0.0
		
		# Get vehicle transform
		vehicle_transform = self.vehicle.get_transform()
		vehicle_location = vehicle_transform.location
		
		# Get vehicle's forward vector (normalized)
		vehicle_forward = vehicle_transform.get_forward_vector()
		
		# Get waypoint locations
		next_location = next_waypoint.transform.location
		
		# Calculate direction vector to next waypoint (target direction)
		target_direction = carla.Vector3D(
			x=next_location.x - vehicle_location.x,
			y=next_location.y - vehicle_location.y,
			z=0.0  # Ignore height difference
		)
		
		# Normalize the target direction vector
		target_length = np.sqrt(target_direction.x**2 + target_direction.y**2)
		if target_length < 0.001:  # Avoid division by zero
			return 1.0  # We're very close to target, assume alignment
			
		target_direction.x /= target_length
		target_direction.y /= target_length
		
		# Calculate cosine of the angle (dot product of normalized vectors)
		cos_yaw_diff = (vehicle_forward.x * target_direction.x + 
						vehicle_forward.y * target_direction.y)
		
		return cos_yaw_diff

	def track_waypoints(self, buffer_distance, checkpoint_rwd):
		"""
		Track vehicle progress along waypoints, automatically advancing to the next waypoint
		when the vehicle passes the current one. Dynamically updates the current waypoint
		to the closest one if the car is closer to another waypoint.
		"""
		reward = 0.0
		if not self.vehicle or not self.current_route:
			return None, float('inf'), None, 0.0

		 # Get vehicle location
		vehicle_location = self.vehicle.get_transform().location

		# Get current target waypoint
		current_waypoint = self.current_route[self.current_waypoint_index][0]
		current_location = current_waypoint.transform.location
		current_distance = vehicle_location.distance(current_location)

		if self.current_waypoint_index + 2 < len(self.current_route):
			next2_waypoint = self.current_route[self.current_waypoint_index+2][0]
			next2_location = next2_waypoint.transform.location
			next2_distance = vehicle_location.distance(next2_location)
		else:
			next2_waypoint = None

		if next2_waypoint is not None and  current_distance > next2_distance:
			self.current_waypoint_index += 2
			current_waypoint = self.current_route[self.current_waypoint_index][0]
			current_location = current_waypoint.transform.location
			current_distance = vehicle_location.distance(current_location)

		# Check if vehicle has reached the waypoint (within buffer)
		if current_distance < buffer_distance:
			# Advance to next waypoint if not at the end
			if self.current_waypoint_index < len(self.current_route) - 1:
				self.current_waypoint_index += 1

				if self.current_waypoint_index % 2 == 0:
					reward = checkpoint_rwd

		# Get next waypoint if available
		next_waypoint = None
		if self.current_waypoint_index < len(self.current_route) - 1:
			next_waypoint = self.current_route[self.current_waypoint_index + 1][0]

		# Display the next 3 waypoints in the simulator
		for i in range(1, 5):
			if self.current_waypoint_index + i < len(self.current_route):
				next_waypoint_location = self.current_route[self.current_waypoint_index + i][0].transform.location
				self.world.debug.draw_point(
					next_waypoint_location + carla.Location(z=0.5),  # Slightly above ground
					size=0.03,
					color=carla.Color(2, 0, 0),  # Green color for forward waypoints
					life_time=50  # Display for 50 seconds
				)

		return current_waypoint, current_distance, next_waypoint, reward
	
	def _generate_track(self):
		"""Generate start and end positions for the track"""
		logging.info("Generating track with start and end waypoints")
		
		# Get all available spawn points
		spawn_points = self.map.get_spawn_points()
		# start_numbers = [2, 4, 7, 9, 11, 14, 18, 20, 22, 24, 25, 32, 34, 37, 49, 52, 54, 55, 57, 58, 59, 60, 61, 65, 71, 72, 73, 75, 76, 77]

		# Select random start point and ensure it's valid
		# print(f"-- START POSITIONS: {self.start_spawn}")
		# self.start_spawn += 1
		start_transform = random.choice(spawn_points)
		self.start_waypoint = self.map.get_waypoint(
			start_transform.location,
			project_to_road=True,
			lane_type=carla.LaneType.Driving
		)
		
		target_location = carla.Location(x=-7.403693199157715, y=273.4846496582031, z=0.5)
		self.end_waypoint =self.map.get_waypoint(target_location,
                  project_to_road=True,
                  lane_type=carla.LaneType.Driving)

		# Generate route between start and end points
		self.current_route = self._generate_route(
			self.start_waypoint.transform.location,
			self.end_waypoint.transform.location
		)
		
		logging.info("Track generated from %s to %s", self.start_waypoint, self.end_waypoint)
		return start_transform

	def _generate_route(self, start_location, end_location):
		"""Generate a route between start and end locations"""
		logging.info("Generating route from %s to %s", start_location, end_location)
		
		# Get waypoints
		start_waypoint = self.map.get_waypoint(start_location)
		end_waypoint = self.map.get_waypoint(end_location)
		
		# Generate route using CARLA's built-in path planner
		grp = GlobalRoutePlanner(self.map, sampling_resolution=4.0)
		
		# Calculate route
		route = grp.trace_route(
			start_waypoint.transform.location,
			end_waypoint.transform.location
		)
		
		if not route:
			logging.warning("No route found, regenerating track")
			return self._generate_track()
		else:
			logging.info("Route generated successfully with %d waypoints", len(route))
			
		return route
	
	def spawn_traffic(self, count: int, autopilot=True, retries=10):
		"""Spawn traffic vehicles at random locations"""
		actors: "list[carla.Vehicle]" = []
		spawn_points = self.map.get_spawn_points()
		while len(actors) < count:
			spawn_point = random.choice(spawn_points)
			vehicle_bp = random.choice(self.blueprint_library.filter("vehicle.*"))
			vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
			if vehicle:
				actors.append(vehicle)
				if autopilot:
					vehicle.set_autopilot(True)
			else:
				if retries == 0:
					break
				retries -= 1
		return actors
	
	def spawn_car_in_front(self, distance=12.0, autopilot=True):
		"""Spawn one car directly in front of the agent's car, and save it as followed_vehicle."""
		
		start_transform = self.start_waypoint.transform
		forward_vector = start_transform.get_forward_vector()

		spawn_location = start_transform.location + forward_vector * distance
		spawn_location.z += 0.5  # Slightly lifted up

		spawn_transform = carla.Transform(
			location=spawn_location,
			rotation=start_transform.rotation
		)

		vehicle_bp = random.choice(self.blueprint_library.filter("vehicle.*"))

		# Try multiple times
		for _ in range(2):
			vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
			if vehicle:
				if autopilot:
					vehicle.set_autopilot(True)
				
				# Save the vehicle into self.followed_vehicle
				self.followed_vehicle = vehicle

				print(f"Successfully spawned followed vehicle with id {vehicle.id}")
				return [vehicle]  # Return it if needed
			else:
				spawn_location += forward_vector * 2.0
				spawn_transform.location = spawn_location

		print("Failed to spawn a followed vehicle.")
		self.followed_vehicle = None
		return []

def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

    collision = 0 if collision is None else 1
    
    return cos_yaw_diff, dist, collision

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward

def correct_yaw(x):
    return(((x%360) + 360) % 360)

def draw_circle(world, center, radius=12.0, segments=20, z_offset=0.5):
	for i in range(segments):
		angle = 2 * math.pi * i / segments
		x = center.x + radius * math.cos(angle)
		y = center.y + radius * math.sin(angle)
		z = center.z + z_offset + 15
		world.debug.draw_point(
			carla.Location(x=x, y=y, z=z),
			size=0.1,
			color=carla.Color(255, 255, 0),
			life_time=0.0
		)

def get_curves(world, map):
	custom_waypoints = []
	custom_locations = [
		carla.Location(x=-4, y=303, z=15),
		carla.Location(x=189, y=303, z=15),
		carla.Location(x=189, y=109, z=15),
		carla.Location(x=44, y=193.5, z=15),
		carla.Location(x=-4, y=108, z=15),
		carla.Location(x=44, y=302, z=15),
		carla.Location(x=46, y=239, z=15),
		carla.Location(x=-3, y=189, z=15),
		carla.Location(x=188, y=238, z=15),
		carla.Location(x=133, y=238, z=15),
		carla.Location(x=133, y=193.5, z=15),
		carla.Location(x=188, y=189, z=15),
	]

	for loc in custom_locations:
		waypoint = map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
		if waypoint:
			custom_waypoints.append(waypoint)
			world.debug.draw_point(
				waypoint.transform.location,
				size=0.4,
				color=carla.Color(255, 0, 0),
				life_time=10.0
			)
			# draw_circle(world, waypoint.transform.location)
			# print(f"Custom WP: x={loc.x}, y={loc.y}, z={loc.z}")
	return custom_waypoints