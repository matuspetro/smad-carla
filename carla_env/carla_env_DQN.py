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


SECONDS_PER_EPISODE = 120

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
		self.action_space = spaces.Discrete(20)
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.im_height, self.im_width, N_CHANNELS), dtype=np.uint8)

		# Connect to the CARLA server
		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(15.0)

		# Load the desired map (Town02)
		logging.info("Loading map: Town02")
		self.world = self.client.load_world("Town02")

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
					# logging.info(f"Destroyed actor {actor.type_id} with ID {actor.id}")
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

		# Define mappings
		steer_values = [-0.65, -0.15, 0.0, 0.15, 0.65]
		throttle_values = [0.0, 0.1, 0.3, 0.6]

		# We now have 5 * 4 = 20 possible actions!
		action = int(action)
		if not (0 <= action < 20):
			print(f"WARNING: Invalid action {action}, clipping.")
			action = np.clip(action, 0, 19)

		# Map action to throttle and steer values
		steer_index = action // 4 
		throttle_index = action % 4

		self.steer = steer_values[steer_index]
		self.throttle = throttle_values[throttle_index]
		self.brakes = 1.0 if self.throttle == 0.0 else 0.0

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

		# ====== FÁZA 1: Základná reward funkcia ======

		# # Small alive reward to encourage staying alive
		# reward += 0.2

		# # Lane center reward
		# if lateral_distance > 2.4:
		# 	reward -= 300
		# 	done = True
		# 	print("\033[91mOff-road detected, episode ends!\033[0m")
		# elif lateral_distance > 1.8:
		# 	reward -= 4
		# elif lateral_distance > 1.4:
		# 	reward -= 2.5
		# elif lateral_distance > 1.0:
		# 	reward -= 1.5
		# elif lateral_distance > 0.5:
		# 	reward -= 0.8
		# else:
		# 	reward += 2.0

		# # Heading alignment reward
		# reward += 2 * cos_yaw_diff

		# # Speed control
		# if 5 <= kmh <= 20:
		# 	reward += 2
		# elif kmh < 5 or kmh > 25:
		# 	reward -= 1

		# # Distance traveled reward
		# distance_travelled = self.initial_location.distance(vehicle_location)
		# if distance_travelled > (self.distance_flag + 1) * 10:
		# 	self.distance_flag += 1
		# 	# reward += 50
		# 	# print("\033[92mDistance milestone reached!\033[0m")

		# if self.distance_flag > 20:
		# 	done = True
		# 	reward += 100
		# 	print("\033[92mDistance limit reached!\033[0m")

		# # Collision penalty
		# if len(self.collision_hist) != 0:
		# 	reward -= 200
		# 	done = True
		# 	print("\033[91mCollision detected!\033[0m")
		# 	with open("collisions.txt", "r+") as file:
		# 		collisions = int(file.read().strip())
		# 		collisions += 1
		# 		file.seek(0)
		# 		file.write(str(collisions))
		# 		file.truncate()


		# ====== FÁZA 2: Jemnejšia reward funkcia ======

		# # Alive bonus
		# reward += 0.1

		# # Laterálna penalizácia
		# if lateral_distance > 2.5:
		# 	reward -= 400
		# 	done = True
		# 	print("\033[91mOff-road detected, episode ends!\033[0m")
		# elif lateral_distance > 2.0:
		# 	reward -= 8.0
		# elif lateral_distance > 1.5:
		# 	reward -= 6.0
		# elif lateral_distance > 1.0:
		# 	reward -= 4.0
		# elif lateral_distance > 0.5:
		# 	reward -= 2.0
		# else:
		# 	reward += 2

		# # Heading zarovnanie (cos_yaw_diff)
		# reward += 4 * cos_yaw_diff

		# # Penalizácia prudkého steeringu
		# if abs(self.steer) > 0.5:
		# 	reward -= 0.5
		# if abs(self.steer) > 0.7:
		# 	reward -= 0.8

		# # Odmena za rýchlosť
		# if 10 <= kmh <= 30:
		# 	reward += 2
		# elif kmh < 5 or kmh > 40:
		# 	reward -= 1

		# # Kolízia - veľký trest
		# if len(self.collision_hist) != 0:
		# 	reward -= 400
		# 	done = True
		# 	print("\033[91mCollision detected!\033[0m")

		# # Distance traveled reward
		# distance_travelled = self.initial_location.distance(vehicle_location)
		# if distance_travelled > (self.distance_flag + 1) * 10:
		# 	self.distance_flag += 1
		# 	# reward += 20
		# 	# print("\033[92mDistance milestone reached!\033[0m")

		# if self.distance_flag > 20:
		# 	done = True
		# 	reward += 100
		# 	print("\033[92mDistance limit reached!\033[0m")




		# ====== FÁZA 3: Jemné brúsenie správania ======

		# Alive bonus
		reward += 0.1

		in_curve = False

		# for curve in self.curves:
		# 	if vehicle_location.distance(curve.transform.location) < 8:
		# 		in_curve = True
		# 		break

		# if not in_curve:
		# 	# Laterálna penalizácia
		# 	if lateral_distance > 2.0:
		# 		reward -= 600
		# 		done = True
		# 		print("\033[91mOff-road detected, episode ends!\033[0m")
		# 	elif lateral_distance > 1.5:
		# 		reward -= 10.0
		# 	elif lateral_distance > 1.2:
		# 		reward -= 7.5
		# 	elif lateral_distance > 0.8:
		# 		reward -= 5.0
		# 	elif lateral_distance > 0.3:
		# 		reward -= 2.0
		# 	else:
		# 		reward += 2.0

		# 	# Heading zarovnanie (cos_yaw_diff)
		# 	reward += 2 * cos_yaw_diff

		# 	# Penalizácia prudkého steeringu
		# 	if abs(self.steer) > 0.1:
		# 		reward -= 1
		# 	if abs(self.steer) > 0.5:
		# 		reward -= 2

		# 	# Odmena za rýchlosť
		# 	if 5 <= kmh <= 25:
		# 		reward += 3
		# 	elif kmh < 5 or kmh > 27:
		# 		reward -= 5

		# 	# Check if there is any actor in front of the agent
		# 	actors_in_front = self.world.get_actors().filter('vehicle.*')
		# 	agent_forward_vector = self.vehicle.get_transform().get_forward_vector()

		# 	for actor in actors_in_front:
		# 		if actor.id != self.vehicle.id:  # Exclude the agent itself
		# 			actor_location = actor.get_location()
		# 			direction_to_actor = actor_location - vehicle_location
		# 			distance_to_actor = math.sqrt(direction_to_actor.x**2 + direction_to_actor.y**2 + direction_to_actor.z**2)

		# 			# Check if the actor is within a certain distance and in front of the agent
		# 			if ( agent_forward_vector.x * direction_to_actor.x +
		# 				agent_forward_vector.y * direction_to_actor.y +
		# 				agent_forward_vector.z * direction_to_actor.z
		# 			) / math.sqrt(direction_to_actor.x**2 + direction_to_actor.y**2 + direction_to_actor.z**2) > 0.8:
		# 				# Check if the actor is in the same lane as the agent
		# 				actor_waypoint = self.world.get_map().get_waypoint(
		# 					actor_location,
		# 					project_to_road=True,
		# 					lane_type=carla.LaneType.Driving
		# 				)
		# 				agent_waypoint = self.world.get_map().get_waypoint(
		# 					vehicle_location,
		# 					project_to_road=True,
		# 					lane_type=carla.LaneType.Driving
		# 				)
		# 				if actor_waypoint.lane_id == agent_waypoint.lane_id:
		# 					if distance_to_actor < 10.0:
		# 						v_actor = actor.get_velocity()
		# 						kmh_actor = int(3.6 * math.sqrt(v_actor.x**2 + v_actor.y**2 + v_actor.z**2))
		# 						if kmh_actor < kmh:
		# 							print("\033[91mSlower vehicle detected in front!\033[0m")
		# 							reward -= 15
		# 						else:
		# 							print("\033[92mFaster vehicle detected in front!\033[0m")
		# 							reward += 8

		# else:
		# 	# Odmena za rýchlosť
		# 	if 2 <= kmh <= 20:	
		# 		reward += 4
		# 	elif kmh > 22:
		# 		reward -= 2
		# 	elif kmh < 2:
		# 		reward = -4
		
		# if self.closest_traffic_light is not None:
		# 	traffic_light_state = self.closest_traffic_light.get_state()
		# 	if traffic_light_state == carla.TrafficLightState.Red:
		# 		if self.traffic_light_distance > 7:
		# 			if kmh > 0:
		# 				reward -= 5
		# 			if kmh == 0:
		# 				reward += 7
		# 				print("\033[92mRed traffic light detected!\033[0m")
		# 		else:
		# 			if kmh > 0:
		# 				reward -= 15
		# 				print("\033[91mRed traffic light detected, but vehicle is not stopped!\033[0m")
		# 				self.traffic_light_hist += 1
		# 				if self.traffic_light_hist > 10:
		# 					reward -= 600
		# 					done = True
		# 					print("\033[91mRed traffic light detected, episode ends!\033[0m")
		# 			else:
		# 				reward += 7
		# 				print("\033[92mRed traffic light detected!\033[0m")
		# else:
		# 	self.traffic_light_hist = 0


		# Kolízia - veľký trest
		if len(self.collision_hist) != 0:
			if kmh > 0:
				reward -= 600
			done = True
			print("\033[91mCollision detected!\033[0m")
			with open("collisions.txt", "r+") as file:
				collisions = int(file.read().strip())
				collisions += 1
				file.seek(0)
				file.write(str(collisions))
				file.truncate()

		# Distance traveled reward
		# distance_travelled = self.initial_location.distance(vehicle_location)
		# if distance_travelled > (self.distance_flag + 1) * 10:
		# 	self.distance_flag += 1
		# # 	reward += 5
		# # 	print("\033[92mDistance milestone reached!\033[0m")

		# if self.distance_flag > 20:
		# 	done = True
		# 	reward += 200
		# 	print("\033[92mDistance limit reached!\033[0m")

		# # Lane invasion penalty
		# if len(self.lane_invade_hist) != 0:
		# 	if lateral_distance > 1:
		# 		reward -= 100
		# 		done = True
		# 		print("\033[91mLane invasion detected!\033[0m")
		# 	else:
		# 		self.count_line_hits = 0

		# Time limit
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			print("\033[91mEpisode time limit reached!\033[0m")
			done = True

		# Logging every 50 steps
		if self.step_counter % 30 == 0:
			print(' ')
			print("\033[93m   - reward:", reward, "\033[0m")
			print('   - speed:', kmh)
			print('   - steer:', self.steer)
			# print('   - traveled:', distance_travelled)
			# print('   - distance:', lateral_distance)
			# print('   - heading:', cos_yaw_diff)
			# print('   - in curve:', in_curve)

		terminated = done
		truncated = done

		reward = reward / 100.0

		return self.front_camera, reward, terminated, truncated, {}


	# def step(self, action):
	# 	self.world.tick()
	# 	trans = self.vehicle.get_transform()
	# 	self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=30),carla.Rotation(yaw =-180, pitch=-90)))
		
	# 	self.step_counter +=1
		
		
	# 	# map steering actions
	# 	if action ==0:
	# 		self.steer = - 0.9
	# 	elif action ==1:
	# 		self.steer = -0.25
	# 	elif action ==2:
	# 		self.steer = -0.1
	# 	elif action ==3:
	# 		self.steer = -0.05
	# 	elif action ==4:
	# 		self.steer = 0.0 
	# 	elif action ==5:
	# 		self.steer = 0.05
	# 	elif action ==6:
	# 		self.steer = 0.1
	# 	elif action ==7:
	# 		self.steer = 0.25
	# 	elif action ==8:
	# 		self.steer = 0.9
		
	# 	v = self.vehicle.get_velocity()
	# 	kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
	# 	estimated_throttle = self.maintain_speed(kmh)
	# 	# map throttle and apply steer and throttle	
	# 	if action == 9:
	# 		self.throttle = 0.0
	# 		self.brakes = 1.0
	# 	elif action == 10:
	# 		self.throttle = 0.3
	# 		self.brakes = 0.0
	# 	elif action == 11:
	# 		self.throttle = 0.5
	# 		self.brakes = 0.0
	# 	elif action == 12:
	# 		self.throttle = 0.7
	# 		self.brakes = 0.0
		
	# 	self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake = self.brakes))
		
	# 	distance_travelled = self.initial_location.distance(self.vehicle.get_location())

	# 	# track steering lock duration to prevent "chasing its tail"
	# 	#lock_duration = 0
	# 	#if self.steering_lock == False:
	# 	#	if steer<-0.6 or steer>0.6:
	# 	#		self.steering_lock = True
	# 	#		self.steering_lock_start = time.time()
	# 	#else:
	# 	#	if steer<-0.6 or steer>0.6:
	# 	#		lock_duration = time.time() - self.steering_lock_start

	# 	# start defining reward from each step
	# 	reward = 0
	# 	done = False

	# 	vehicle_location = self.vehicle.get_location()
	# 	end_location = self.end_waypoint.transform.location

	# 	# if self.followed_vehicle is not None:		
	# 	# 	followed_location = self.followed_vehicle.get_location()

	# 	# 	followed_distance = vehicle_location.distance(followed_location)
	# 	# 	if followed_distance < 10.0:
	# 	# 		reward -= 50
	# 	# 	elif followed_distance < 20.0:
	# 	# 		reward += 200
	# 	# 	elif followed_distance < 30.0:
	# 	# 		reward += 50
	# 	# 	else:
	# 	# 		reward -= 50

	# 	# # Compute distance to end waypoint
	# 	# dist = math.sqrt(
	# 	# 	(vehicle_location.x - end_location.x) ** 2 +
	# 	# 	(vehicle_location.y - end_location.y) ** 2 +
	# 	# 	(vehicle_location.z - end_location.z) ** 2
	# 	# )

	# 	# if dist < 4.0:
	# 	# 	reward += 2000
	# 	# 	done = True
	# 	# 	print("\033[92mGoal reached!\033[0m")


	# 	# Penalize distance from lane center
	# 	waypoint = self.world.get_map().get_waypoint(
	# 		self.vehicle.get_location(),
	# 		project_to_road=True,
	# 		lane_type=carla.LaneType.Driving
	# 	)
	# 	lane_center = waypoint.transform.location
	# 	lateral_distance = self.vehicle.get_location().distance(lane_center)

	# 	if lateral_distance < 0.5:
	# 		reward += 0.5
	# 	elif lateral_distance > 1.5:
	# 		reward -= 10

	# 	# Reward alignment with lane direction
	# 	cos_yaw_diff, _, _ = get_reward_comp(self.vehicle, waypoint, None)
	# 	reward += cos_yaw_diff * 5

	# 	# Reward progress along the route
	# 	# _, _, _, checkpoint_reward = self.track_waypoints(buffer_distance=1, checkpoint_rwd=500)
	# 	# reward += checkpoint_reward
	# 	# if checkpoint_reward > 0:
	# 	# 	print("\033[92mCheckpoint reached!\033[0m")

	# 	# Penalize collisions
	# 	if len(self.collision_hist) != 0:
	# 		reward -= 200
	# 		done = True
	# 		print("\033[91mCollision detected!\033[0m")
	# 		with open("collisions.txt", "r+") as file:
	# 			collisions = int(file.read().strip())
	# 			collisions += 1
	# 			file.seek(0)
	# 			file.write(str(collisions))
	# 			file.truncate()

	# 	# Penalize lane invasions
	# 	if len(self.lane_invade_hist) != 0:
	# 		# reward -= 300
	# 		# done = True

	# 		# Define a threshold for lane crossing
	# 		lane_crossing_threshold = 1.2  # Adjust this value as needed

	# 		# Check if the car is outside the lane
	# 		if lateral_distance > lane_crossing_threshold:
	# 			reward -= 50
	# 			done = True
	# 			print("\033[91mLane invasion detected!\033[0m")
	# 			# self.count_line_hits+=1
	# 			# if self.count_line_hits > 5:
	# 			# 	done = True
	# 		else:
	# 			self.count_line_hits = 0

	# 	# in_slow_zone = any(
	# 	# 	wp.transform.location.distance(self.vehicle.get_location()) < 10
	# 	# 	for wp in self.curves
	# 	# )

	# 	# if in_slow_zone:
	# 	# 	if kmh > 1 and kmh < 15:
	# 	# 		reward += kmh / 3
	# 	# 	else:
	# 	# 		reward -= 10
	# 	# else:
	# 	# 	if kmh > 5:
	# 	# 		reward += kmh / 5
	# 	# 	else:
	# 	# 		reward -= 10

	# 	# reward for making distance
	# 	# vehicle_location = self.vehicle.get_location()
	# 	# waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, 
    #     #             lane_type=carla.LaneType.Driving)
	# 	# cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, waypoint, None)
	# 	# reward += reward_value(cos_yaw_diff, dist, collision) * 2

	# 	#reward += self.calculate_waypoint_alignment() *
		
	# 	if kmh < 5:
	# 		reward -= 5
	# 	elif 5 <= kmh <= 30:
	# 		reward += 1
	# 	elif kmh > 40:
	# 		reward -= 5

	# 	if kmh > 40:
	# 		reward -= 5
	# 		if kmh > 50:
	# 			done = True
	# 			print("\033[91mOver-speeding! Episode ends.\033[0m")


	# 	if distance_travelled > (self.distance_flag + 1) * 5:
	# 		self.distance_flag += 1
	# 		reward += 50
	# 		print("\033[92mDistance milestone reached!\033[0m")

	# 	if self.distance_flag > 40:
	# 		done = True
	# 		reward += 100
	# 		print("\033[91mDistance limit reached!\033[0m")

	# 	# if reward < -100:
	# 	# 	reward -= 50
	# 	# 	done = True
	# 	# 	print("\033[91mReward too low!\033[0m")

	# 	reward += 1


	# 	# check for episode duration
	# 	if self.episode_start + SECONDS_PER_EPISODE < time.time():
	# 		print("\033[91mEpisode time limit reached!\033[0m")
	# 		done = True
	# 	terminated = done  # Use `done` for both terminated and truncated
	# 	truncated = done
	# 	if self.step_counter % 50 == 0:
	# 		print(' ')
	# 		print("\033[93m   - reward:", reward, "\033[0m")
	# 		print('   - speed:',kmh)
	# 		print('   - steer input from model:',self.steer)
	# 		# print('   - lateral distance: -',lateral_distance * 18)
	# 		# print('   - cos yaw diff:',cos_yaw_diff * 18)
	# 	return self.front_camera, reward, terminated, truncated, {}

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
		self.closest_traffic_light = None
		self.traffic_light_hist = 0
		self.traffic_light_distance = 12

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

		# trans = self.vehicle.get_transform()
		# self.spectator.set_transform(
		# 	carla.Transform(trans.location + carla.Location(z=30), carla.Rotation(yaw=0, pitch=0))
		# )

		# Spawn traffic
		logging.info("Spawning traffic...")
		# self.actor_list.extend(self.spawn_car_in_front(distance=random.randint(8, 12), autopilot=True))
		self.actor_list.extend(self.spawn_traffic(count=30, autopilot=True))
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

		# mask = (img_array[:, :, 0] == 60) & (img_array[:, :, 1] == 20) & (img_array[:, :, 2] == 220)
		# img_array[mask] = [142, 0, 0]

		# Get the vehicle's current location and forward vector
		vehicle_location = self.vehicle.get_location()
		vehicle_forward_vector = self.vehicle.get_transform().get_forward_vector()

		# Find the closest traffic light in the same lane
		traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
		closest_traffic_light = None
		min_distance = float('inf')

		for traffic_light in traffic_lights:
			# Get the traffic light's location
			traffic_light_location = traffic_light.get_location()
			direction_to_light = traffic_light_location - vehicle_location
			distance_to_light = math.sqrt(direction_to_light.x**2 + direction_to_light.y**2 + direction_to_light.z**2)

			# Check if the traffic light is in front of the agent
			if (vehicle_forward_vector.x * direction_to_light.x +
				vehicle_forward_vector.y * direction_to_light.y +
				vehicle_forward_vector.z * direction_to_light.z) / distance_to_light > 0.8:
				
				# Check if the traffic light is in the same lane
				traffic_light_waypoint = self.world.get_map().get_waypoint(
					traffic_light_location,
					project_to_road=True,
					lane_type=carla.LaneType.Driving
				)
				agent_waypoint = self.world.get_map().get_waypoint(
					vehicle_location,
					project_to_road=True,
					lane_type=carla.LaneType.Driving
				)
				if traffic_light_waypoint and agent_waypoint and traffic_light_waypoint.lane_id == agent_waypoint.lane_id:
					if distance_to_light < min_distance and distance_to_light < 10.0:
						min_distance = distance_to_light
						closest_traffic_light = traffic_light
						self.traffic_light_distance = min_distance

		# Determine the color to draw
		if closest_traffic_light:
			state = closest_traffic_light.state
			self.closest_traffic_light = closest_traffic_light
			if state == carla.TrafficLightState.Red:
				if min_distance > 7.0:
					cv2.circle(img_array, (128, 64), 30, (142, 0, 0), -1)
				else:
					cv2.circle(img_array, (128, 64), 50, (142, 0, 0), -1)
			else:
				self.closest_traffic_light = None
		
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
		
		# Select random start point and ensure it's valid
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
		"""Spawn only 4-wheel vehicles, exclude bikes, motorbikes, scooters."""
		actors = []
		spawn_points = self.map.get_spawn_points()
		
		# Filter only 4-wheel blueprints
		vehicle_blueprints = [
			bp for bp in self.blueprint_library.filter("vehicle.*")
			if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4
		]

		while len(actors) < count:
			spawn_point = random.choice(spawn_points)
			vehicle_bp = random.choice(vehicle_blueprints)
			vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
			if vehicle:
				actors.append(vehicle)
				if autopilot:
					vehicle.set_autopilot(True)
			else:
				retries -= 1
				if retries == 0:
					break
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

		# Filter only 4-wheel blueprints
		vehicle_blueprints = [
			bp for bp in self.blueprint_library.filter("vehicle.*")
			if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) == 4
		]

		vehicle_bp = random.choice(vehicle_blueprints)

		# Try multiple times
		for _ in range(2):
			vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
			if vehicle:
				if autopilot:
					vehicle.set_autopilot(True)
				
				# ✅ Save the vehicle into self.followed_vehicle
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
				life_time=3.0
			)
			# draw_circle(world, waypoint.transform.location)
			# print(f"Custom WP: x={loc.x}, y={loc.y}, z={loc.z}")
	return custom_waypoints