from glob import glob
import json 
import uuid
from itertools import groupby
from operator import itemgetter
import os

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pyquaternion import Quaternion


def extract_info(filename):
    path = filename.split("/")
    filename = path[-1]
    team = path[-2]
    log = path[-3]
    scene = filename.split(".")[0]
    return team, log, scene


def detection_json_to_map(detections, filename):
    team, log, scene = extract_info(filename)

    with open(file=filename, mode="r") as fp:
        datumaro = json.load(fp)

    detections[log][team][scene] = {}
    for frame in datumaro["items"]:
        detections[log][team][scene][frame["id"]] = [{"id": annotation["id"], "position": annotation["position"], "size": annotation["scale"], "yaw": annotation["rotation"][2]} for annotation in frame["annotations"]]

    return detections


def generate_unique_token():
    return uuid.uuid4().hex

def generate_log(log, team):
    """ Generate one log per session
    """
    return {
        'token': generate_unique_token(),
        "logfile": "", #@TODO
        "vehicle": team,
        "date_captured": log, # @TODO use this as specification when it was?
        "location": f"Yas Marina Circuit",
    }

def generate_scene(scene, log, team, nbr_samples):
    return {
        'token': generate_unique_token(),
        'log_token': "",
        'nbr_samples': nbr_samples,
        'first_sample_token': "",
        'last_sample_token': "",
        'name': scene,
        'description': f"{scene} of {log} for team {team}",
    }

def generate_sample(scene_token, timestamp):
    return {
        "token": generate_unique_token(),
        "scene_token": scene_token,
        "timestamp": timestamp,
        "next": "",
        "prev": ""
    }

def generate_sample_annotation(sample_token, annotation):
    return {
        "token": generate_unique_token(),
        "sample_token": sample_token,
        "instance_token": get_instance_token(INSTANCE_MAP, annotation["id"]),
        "annotation_id": annotation["id"], # this is a custom field and is just there to match instances to sample_annotations -> its badly implemented tbh
        "prev": "",
        "next": "",
        "attribute_tokens": [],
        "visibility_token": "0", # We do not annotate this
        "translation": annotation["position"],
        "size": [annotation["size"][1], annotation["size"][0], annotation["size"][2]], # flip annotation sizes to match NUS format
        "rotation": Quaternion(axis=[0, 0, 1], angle=annotation["yaw"]).q.tolist(),
        "num_lidar_pts": 0,
        "num_radar_pts": 0
    }


def generate_sample_data(sample_token, team_name, ego_pose_token, calibrated_sensor_token, timestamp, type="LIDAR"):
    return {
        "token": generate_unique_token(),
        "sample_token": sample_token,
        "ego_pose_token": ego_pose_token,
        "calibrated_sensor_token": calibrated_sensor_token,
        "timestamp": timestamp,
        "fileformat": "pcd",
        "is_key_frame": True,
        "height": 0,
        "width": 0,
        "filename": f"samples/LIDAR/LIDAR_{team_name}_{timestamp}.pcd.bin",
        "prev": "",
        "next": ""
    }

def generate_ego_pose(timestamp, translation, rotation):
    return {
        "token": generate_unique_token(),
        "timestamp": timestamp,
        "rotation": rotation,
        "translation": translation
    }


def generate_calibrated_sensor(sensor_token, translation, rotation, camera_intrinsic):
    return {
        "token": generate_unique_token(),
        "sensor_token": sensor_token,
        "translation": translation,
        "rotation": rotation,
        "camera_intrinsic": camera_intrinsic
    }

INSTANCE_MAP = {}
def get_instance_token(map, id):
    if id not in map:
        map[id] = generate_unique_token()

    return map[id]


def find_last_sample_annotation(samples_annotations, token):
    for sample_annotation in reversed(samples_annotations):
        if sample_annotation['instance_token'] == token:
            return sample_annotation
    
    return None


def generate_instances(samples_annotations, racecar_category_token):
    samples_annotations.sort(key=itemgetter('instance_token'))
    grouped_objects = {k: list(v) for k, v in groupby(samples_annotations, key=itemgetter('instance_token'))}
    instances = []
    for _, group in grouped_objects.items():
        instances.append({
            "token": get_instance_token(INSTANCE_MAP, group[0]["annotation_id"]),
            "category_token": racecar_category_token,
            "nbr_annotations": len(group),
            "first_annotation_token": group[0]["token"],
            "last_annotation_token": group[-1]["token"]
        })

    return instances


def load_transform_save_lidar_data(path_data, path_output, log, scene, team, filename):
    # not the most beautiful way but filename is already in sample/...
    original_filename = filename.split(os.path.sep)[-1].split("_")[-1][:-4]
    pcd = o3d.io.read_point_cloud(os.path.join(path_data, log, team, scene, original_filename))
    
    points = np.asarray(pcd.points)
    points_nus = np.zeros([points.shape[0], 5], dtype=np.float32)
    points_nus[:, :points.shape[1]] = points
    
    
    file_path = os.path.join(path_output, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    with open(file_path, "wb") as fp:
       points_nus.tofile(fp)


def detections_to_nuscenes(detections, path_json, path_data, path_output):
    attributes = []
    calibrated_sensors = []
    categories = [{
        "token": generate_unique_token(),
        "name": "racecar",
        "description": "A racecar participating in the A2RL competition."
    }]
    ego_poses = []
    instances = []
    logs = []
    maps = [{
        "category": "semantic_prior",
        "token": "53992ee3023e5494b90c316c183be829",
        "filename": "maps/53992ee3023e5494b90c316c183be829.png",
        "log_tokens": []
    },]
    samples = []
    samples_annotations = []
    samples_data = []
    scenes = []
    vehicle_translation = [0,0,0]
    vehicle_rotation = [1,0,0,0]
    sensor_lidar_vehicle_token = generate_unique_token()
    sensor_radar_vehicle_token = generate_unique_token()
    sensors = [
        {
            "token": sensor_lidar_vehicle_token,
            "channel": "LIDAR_VEHICLE",
            "modality": "lidar"
        },
        {
            "token": sensor_radar_vehicle_token,
            "channel": "RADAR_VEHICLE",
            "modality": "radar"
        },
    ]
    visibility = [{
        "token": "0",
        "description": "not implemented",
        "level": "0"
    }]


    for log, teams in detections.items():
        print(f"Extracting {log}")
        for team, log_scenes in tqdm(teams.items()):
            log = generate_log(log, team)
            logs.append(log)
            for scene, frames in log_scenes.items():
                scene = generate_scene(scene, log["date_captured"], team, len(frames))

                scene_samples = []
                scene_sample_annotations = []
                scene_sample_data = []
                for sample_idx, (timestamp, frame) in enumerate(frames.items()):
                    #
                    # Generate Samples
                    #
                    sample = generate_sample(scene["token"], timestamp)
                    if len(scene_samples) > 0:
                        sample["prev"] = scene_samples[sample_idx - 1]["token"]
                        scene_samples[-1]["next"] = sample["token"]
                    scene_samples.append(sample)

                    #
                    # Generate Annotations
                    #
                    for annotation in frame:
                        sample_annotation = generate_sample_annotation(sample["token"], annotation)
                        last_sample_annotation = find_last_sample_annotation(scene_sample_annotations, sample_annotation["instance_token"])
                        if last_sample_annotation:
                            sample_annotation["prev"] = last_sample_annotation["token"]
                            last_sample_annotation["next"] = sample_annotation["token"]
                        
                        scene_sample_annotations.append(sample_annotation)

                    #
                    # Generate Sample Data
                    #
                    # We define all LiDAR frames as key frames. Radar / Camera are associated to closest
                    #

                    ego_pose = generate_ego_pose(timestamp, translation=vehicle_translation, rotation=vehicle_rotation) # @TODO: gnss values?
                    ego_poses.append(ego_pose)

                    # LiDAR pointclouds and detections are already in vehicle frame
                    calibrated_sensor = generate_calibrated_sensor(sensor_lidar_vehicle_token, translation=vehicle_translation, rotation=vehicle_rotation, camera_intrinsic=[])
                    calibrated_sensors.append(calibrated_sensor)

                    sample_data = generate_sample_data(
                        sample["token"],
                        team,
                        ego_pose_token=ego_pose["token"],
                        calibrated_sensor_token=calibrated_sensor["token"],
                        timestamp=timestamp,
                        type="LIDAR"
                    )
                    if len(scene_sample_data) > 0:
                        sample_data["prev"] = scene_sample_data[-1]["token"]
                        scene_sample_data[-1]["next"] = sample_data["token"]
                    scene_sample_data.append(sample_data)

                    # Move LiDAR data
                    load_transform_save_lidar_data(path_data, path_output, log["date_captured"], scene["name"], team, sample_data["filename"])
                    

                # @TODO: Add Sample Data + other related json for Radar and Camera 
                #    -> go through files in a loop and add them later on per scene
                instances.extend(generate_instances(scene_sample_annotations, categories[0]["token"]))
                scenes.append(scene)
                samples.extend(scene_samples)
                samples_annotations.extend(scene_sample_annotations)
                samples_data.extend(scene_sample_data)
    
    maps[0]["log_tokens"] = [log["token"] for log in logs]

    #
    # Dump NuScenes format
    #
    with open(os.path.join(path_json, "attribute.json"), 'w') as json_file:
        json.dump(attributes, json_file, indent=2)    
    with open(os.path.join(path_json, "calibrated_sensor.json"), 'w') as json_file:
        json.dump(calibrated_sensors, json_file, indent=2)
    with open(os.path.join(path_json, "category.json"), 'w') as json_file:
        json.dump(categories, json_file, indent=2)
    with open(os.path.join(path_json, "ego_pose.json"), 'w') as json_file:
        json.dump(ego_poses, json_file, indent=2)
    with open(os.path.join(path_json, "instance.json"), 'w') as json_file:
        json.dump(instances, json_file, indent=2)
    with open(os.path.join(path_json, "log.json"), 'w') as json_file:
        json.dump(logs, json_file, indent=2)
    with open(os.path.join(path_json, "map.json"), 'w') as json_file:
        json.dump(maps, json_file, indent=2)
    with open(os.path.join(path_json, "sample.json"), 'w') as json_file:
        json.dump(samples, json_file, indent=2)
    with open(os.path.join(path_json, "sample_annotation.json"), 'w') as json_file:
        json.dump(samples_annotations, json_file, indent=2)
    with open(os.path.join(path_json, "sample_data.json"), 'w') as json_file:
        json.dump(samples_data, json_file, indent=2)
    with open(os.path.join(path_json, "scene.json"), 'w') as json_file:
        json.dump(scenes, json_file, indent=2)
    with open(os.path.join(path_json, "sensor.json"), 'w') as json_file:
        json.dump(sensors, json_file, indent=2)
    with open(os.path.join(path_json, "visibility.json"), 'w') as json_file:
        json.dump(visibility, json_file, indent=2)


if __name__ == "__main__":
    import random
    random.seed(42)

    annotations_path = "/home/marvin/dev/aipex/dataset/dataset/data/scene"
    path_data = "/home/marvin/dev/aipex/dataset/dataset/data/sensors"
    path_output = "/home/marvin/dev/aipex/dataset/dataset/data/v1.0-mini"

    detections = {
        "race_1": {
            "tum": {},
            "c19": {},
        }
    }
    for datumaro_filepath in sorted(glob(f"{annotations_path}/*/*/*.json")):
        detection_json_to_map(detections, datumaro_filepath)

    os.makedirs(path_output, exist_ok=True)
    os.makedirs(os.path.join(path_output, "maps"), exist_ok=True)
    path_json = os.path.join(path_output, "v1.0-mini")
    os.makedirs(path_json, exist_ok=True)
    os.makedirs(os.path.join(path_output, "samples"), exist_ok=True)
    os.makedirs(os.path.join(path_output, "sweeps"), exist_ok=True)
    detections_to_nuscenes(detections, path_json, path_data, path_output)