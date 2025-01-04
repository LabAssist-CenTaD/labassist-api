import os
import json
import jsonpatch
from copy import deepcopy

from app.schemas.annotation import Annotation

class VideoJSONManager:
    def __init__(self, json_path: str, video_json: dict = None):
        self.video_json = video_json
        self.json_path = json_path
        self.video_template = {
            "file_name": None,
            "file_path": None,
            "status_list": [],
            "annotations": [],
            "status_counts": {
                "info": 0,
                "warning": 0,
                "error": 0
            }
        }
        self.annotation_template = {
            "type": None,
            "message": None,
            "timestamp": "00:00:00"
        }
        if video_json is None:
            self.video_json = self.load_json()
            if 'videos' not in self.video_json:
                self.video_json['videos'] = {}
            if 'active_tasks' not in self.video_json:
                self.video_json['active_tasks'] = {}
            
    def sync_videos(self, video_dir: str) -> None:
        for device_id in self.video_json['videos']:
            for video in self.video_json['videos'][device_id]:
                if not os.path.exists(os.path.join(video_dir, device_id, video["file_name"])):
                    self.video_json['videos'][device_id].remove(video)
        self.save_json()
        
    def add_device(self, device_id: str) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            self.video_json['videos'][device_id] = []
        self.save_json()
        return self.create_patch([], self.video_json['videos'][device_id])
    
    def remove_device(self, device_id: str) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"Device ID {device_id} not found"}
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        del self.video_json['videos'][device_id]
        self.save_json()
        return self.create_patch(old_device_videos, [])
        
    def add_video(self, device_id: str, video_name: str, video_path: str, status: list[str] = ["uploaded"], annotations: list[dict[str]] = None) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            self.video_json['videos'][device_id] = []
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        video_entry = deepcopy(self.video_template)
        video_entry["file_name"] = video_name
        video_entry["file_path"] = video_path
        self.video_json['videos'][device_id].append(video_entry)
        if status is not None:
            for s in status:
                self.add_status(device_id, video_name, s)
        if annotations is not None:
            for annotation in annotations:
                self.add_annotation(device_id, video_name, annotation)
        self.save_json()
        return self.create_patch(old_device_videos, self.video_json['videos'][device_id])
    
    def remove_video(self, device_id: str, video_name: str) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"device ID {device_id} not found"}
        elif not any(video["file_name"] == video_name for video in self.video_json['videos'][device_id]):
            return {"message": f"Video {video_name} not found for device {device_id}"}
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        self.video_json['videos'][device_id] = [video for video in self.video_json['videos'][device_id] if video["file_name"] != video_name]
        self.save_json()
        return self.create_patch(old_device_videos, self.video_json['videos'][device_id])
    
    def clear_status(self, device_id: str, video_name: str) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"device ID {device_id} not found"}
        elif not any(video["file_name"] == video_name for video in self.video_json['videos'][device_id]):
            return {"message": f"Video {video_name} not found for device {device_id}"}
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        video = next(video for video in self.video_json['videos'][device_id] if video["file_name"] == video_name)
        video["status_list"] = []
        self.save_json()
        return self.create_patch(old_device_videos, self.video_json['videos'][device_id])
    
    def add_status(self, device_id: str, video_name: str, status: str) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"device ID {device_id} not found"}
        elif not any(video["file_name"] == video_name for video in self.video_json['videos'][device_id]):
            return {"message": f"Video {video_name} not found for device {device_id}"}
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        video = next(video for video in self.video_json['videos'][device_id] if video["file_name"] == video_name)
        video["status_list"].append(status)
        self.save_json()
        return self.create_patch(old_device_videos, self.video_json['videos'][device_id])
        
    def add_annotation(self, device_id: str, video_name: str, annotation: dict | Annotation) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"device ID {device_id} not found"}
        elif not any(video["file_name"] == video_name for video in self.video_json['videos'][device_id]):
            return {"message": f"Video {video_name} not found for device {device_id}"}
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        video = next(video for video in self.video_json['videos'][device_id] if video["file_name"] == video_name)
        if isinstance(annotation, Annotation):
            annotation = annotation.to_dict()
        video["annotations"].append(annotation)
        video["status_counts"][annotation['type']] += 1
        self.save_json()
        return self.create_patch(old_device_videos, self.video_json['videos'][device_id])
        
    def clear_annotations(self, device_id: str, video_name: str) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"Device ID {device_id} not found"}
        elif not any(video["file_name"] == video_name for video in self.video_json['videos'][device_id]):
            return {"message": f"Video {video_name} not found for device {device_id}"}
        old_device_videos = deepcopy(self.get_device_videos(device_id))
        video = next(video for video in self.video_json['videos'][device_id] if video["file_name"] == video_name)
        video["annotations"] = []
        video["status_counts"] = {
            "info": 0,
            "warning": 0,
            "error": 0
        }
        self.save_json()
        return self.create_patch(old_device_videos, self.video_json['videos'][device_id])
    
    def add_task(self, device_id: str, video_name: str, task_id: str) -> None:
        if device_id not in self.video_json['active_tasks']:
            self.video_json['active_tasks'][device_id] = {}
        self.video_json['active_tasks'][device_id][video_name] = task_id
        
    def get_task(self, device_id: str, video_name: str) -> str:
        if device_id in self.video_json['active_tasks'] and video_name in self.video_json['active_tasks'][device_id]:
            return self.video_json['active_tasks'][device_id][video_name]
        return None
        
    def remove_task(self, device_id: str, video_name: str) -> None:
        if device_id in self.video_json['active_tasks'] and video_name in self.video_json['active_tasks'][device_id]:
            del self.video_json['active_tasks'][device_id][video_name]
        
    def get_video(self, device_id: str, video_name: str) -> dict:
        if device_id not in self.video_json['videos']:
            return {"message": f"device ID {device_id} not found"}
        elif not any(video["file_name"] == video_name for video in self.video_json['videos'][device_id]):
            return {"message": f"Video {video_name} not found for device {device_id}"}
        return next(video for video in self.video_json['videos'][device_id] if video["file_name"] == video_name)
    
    def get_device_videos(self, device_id: str) -> list[dict]:
        if device_id not in self.video_json['videos']:
            return []
        return self.video_json['videos'][device_id]
    
    def get_all_videos(self) -> dict:
        return self.video_json['videos']
    
    def create_patch(self, old_json: dict, new_json: dict) -> jsonpatch.JsonPatch:
        return jsonpatch.JsonPatch.from_diff(old_json, new_json)
    
    def apply_patch(self, device_id: str, patch: jsonpatch.JsonPatch) -> jsonpatch.JsonPatch:
        if device_id not in self.video_json['videos']:
            return {"message": f"device ID {device_id} not found"}
        old_device_videos = self.get_device_videos(device_id)
        new_device_videos = patch.apply(old_device_videos)
        self.video_json['videos'][device_id] = new_device_videos
        self.save_json()
        return new_device_videos
        
    def load_json(self, json_path: str = None) -> dict:
        if json_path is None:
            json_path = self.json_path
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            return {}
        
    def save_json(self, json_path: str = None) -> None:
        if json_path is None:
            json_path = self.json_path
        with open(json_path, 'w') as f:
            json.dump(self.video_json, f)
            
    def reset_json(self) -> None:
        self.video_json = {}
        self.save_json()
        
    def __str__(self) -> str:
        return json.dumps(self.video_json, indent=4)
    
    def __repr__(self) -> str:
        return f"VideoJsonManager({self.video_json}, {self.json_path})"
    
if __name__ == '__main__':
    json_path = 'video_json.json'
    vjm = VideoJSONManager(json_path)
    print(vjm.sync_videos('uploads'))
    # print(vjm.add_video("device1", "video2.mp4", "/path/to/video2.mp4"))
    # print(vjm.add_annotation("device1", "video2.mp4", "warning", "This is a warning", "00:00:05"))
    # print(vjm.clear_annotations("device1", "video2.mp4"))
    # print(vjm.remove_video("device1", "video2.mp4"))
    # print(vjm.remove_device("device1"))
    