import matplotlib
import matplotlib.pyplot as plt  # for later visualization
import numpy as np  # for easier vector computations
import time
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

beamng = BeamNGpy('localhost', 64256)
beamng.open()

# Create a vehicle with a camera sensor attached to it
vehicle = Vehicle('ego_vehicle', model='etki', licence='PYTHON', color='Green')
cam_pos = np.array([0, 0, 0])  # placeholder values that will be recomputed later
cam_dir = np.array([0, 0, 0])  # placeholder values that will be recomputed later
cam_fov = 70
cam_res = (512, 512)
camera = Camera(cam_pos, cam_dir, cam_fov, cam_res, colour=True)
vehicle.attach_sensor('camera', camera)

# Simple scenario with the vehicle we just created standing at the origin
car_pos = np.array([0, 0, 0])
scenario = Scenario('west_coast_usa', 'tech_test',
                    description='Random driving for research')
scenario.add_vehicle(vehicle, pos=car_pos, rot=(0, 0, 0))
scenario.make(beamng)

beamng.load_scenario(scenario)
beamng.start_scenario()




