from nuscenes.nuscenes import NuScenes
import numpy as np
np.bool = np.bool_
from pyquaternion import Quaternion

from tracking.model import AB3DMOT


def get_gt_detections(sample):
    annotation_tokens = sample['anns']
    boxes = []

    for ann_token in annotation_tokens:
        box = nusc.get_box(ann_token)
        # [h,w,l,x,y,z,theta]
        boxes.append([box.wlh[2], box.wlh[0], box.wlh[1], box.center[0], box.center[1], box.center[2], box.orientation.radians])

    return np.array(boxes)


def get_sample_result(sample_token, track):
    q = Quaternion(axis=[0, 0, 1], angle=track[6])

    return {
        "sample_token": sample_token,
        "translation": track[3:6].tolist(), # center_x, center_y, center_z.
        "size": [track[1], track[2], track[0]], # width, length, height.
        "rotation": q.elements.tolist(), # w, x, y, z.
        "velocity": [0.0, 0.0], # Not supported
        "tracking_id": track[7],
        "tracking_name": "car",
        "tracking_score": track[-1]
    }

def process_scene(scene, tracker, total_tracks):
    print(f"Processing scene: {scene['name']}")
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)

    while sample:
        total_tracks["results"][sample["token"]] = []
        detections = get_gt_detections(sample)

        detections = {"dets": detections, "info": np.zeros_like(detections)}
        # h,w,l,x,y,z,theta, ID, ..., confidence
        tracks, affi = tracker.track(detections)
        tracks = tracks[0]# ???

        for track in tracks:
            total_tracks["results"][sample["token"]].append(get_sample_result(sample["token"], track))
            

        # Move to the next sample
        if sample['next']:
            sample = nusc.get('sample', sample['next'])
        else:
            sample = None
    
    

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='./data/v1.0-mini', verbose=True)

    split = "mini_train"
    from nuscenes.eval.common.loaders import create_splits_scenes
    splits = create_splits_scenes()
    splits = splits[split]

    total_tracks = {
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False
            },
            "results": {
            }
        }
    for scene in nusc.scene:
        if not scene["name"] in splits:
            continue

        tracker = AB3DMOT()
        process_scene(scene, tracker, total_tracks)

    import json
    with open(f"prediction_tracks_{split}.json", 'w') as json_file:
        json.dump(total_tracks, json_file, indent=4)
