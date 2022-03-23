import beamngpy
import random
import numpy as np
import time
from matplotlib import pyplot as plt
from beamngpy import BeamNGpy, Scenario, Vehicle, setup_logging, beamngcommon
from beamngpy.sensors import Camera,Lidar


def simulatorStart():
    global vehicle
    beamngcommon.set_up_simple_logging()
    beamng = BeamNGpy('localhost', 64256)
    bng = beamng.open(launch=True)
    scenario = Scenario("west_coast_usa","Master-Thesis")
    vehicle = Vehicle("ego_vehicle", model="etk800",license="Master_Patch")
    cam_pos = (0, 0, 0)
    cam_dir = (0, 0, 0)
    cam_fov = 70
    cam_res = (1920, 1080)
    camera = Camera(cam_pos, cam_dir, cam_fov, cam_res, colour=True)
    vehicle.attach_sensor('front_cam', camera)

    pos=(-717.121, 101, 118.675)
    rot_q = (0, 0, 0.3826834, 0.9238795)
    scenario.add_vehicle(vehicle,pos=pos, rot=None, rot_quat=rot_q)
    scenario.make(bng)

    bng.hide_hud()
    bng.set_deterministic()  # Set simulator to be deterministic
    bng.set_steps_per_second(60)
    bng.load_scenario(scenario)
    bng.start_scenario()
    bng.pause()
    vehicle.ai_set_mode('span')

    fig = plt.figure(1, figsize=(10, 5))
    axarr = fig.subplots(1, 3)

    a_colour = axarr[0]
    a_depth = axarr[1]
    a_annot = axarr[2]
    while True:
        # Retrieve sensor data and show the camera data.
        sensors = vehicle.poll_sensors()
        a_colour.imshow(sensors['front_cam']['colour'].convert('RGB'))
        a_depth.imshow(sensors['front_cam']['depth'].convert('L'))
        a_annot.imshow(sensors['front_cam']['annotation'].convert('RGB'))
        plt.pause(0.00001)
        image = sensors['front_cam']['colour'].convert('RGB')
        yield image

def send_instructions_to_car():
    throttle = 0.0
    brake=1.0
    vehicle.control(throttle=throttle, brake=brake)

if __name__ == "__main__":
    simulatorStart()

