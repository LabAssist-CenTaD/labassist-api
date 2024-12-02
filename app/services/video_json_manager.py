import json
import jsonpatch
from copy import deepcopy

class VideoJSONManager:
    def __init__(self, json_path: str, video_json: dict = None):
        self.video_json = video_json
        self.json_path = json_path
        self.video_template = {
            "fileName": None,
            "filePath": None,
            "status_list": [],
            "annotations": [],
            "status_counts": {
                "correct": 0,
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
        
    def add_client(self, client_id: str) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            self.video_json[client_id] = []
        self.save_json()
        return self.create_patch({}, self.video_json[client_id])
    
    def remove_client(self, client_id: str) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        old_client_videos = deepcopy(self.get_client_videos(client_id))
        del self.video_json[client_id]
        self.save_json()
        return self.create_patch(old_client_videos, {})
        
    def add_video(self, client_id: str, video_name: str, video_path: str, annotations: list[dict[str]] = None) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            self.video_json[client_id] = []
        old_client_videos = deepcopy(self.get_client_videos(client_id))
        video_entry = self.video_template.copy()
        video_entry["fileName"] = video_name
        video_entry["filePath"] = video_path
        self.video_json[client_id].append(video_entry)
        if annotations is not None:
            for annotation in annotations:
                self.add_annotation(client_id, video_name, annotation["type"], annotation["message"], annotation["timestamp"])
        self.save_json()
        return self.create_patch(old_client_videos, self.video_json[client_id])
    
    def remove_video(self, client_id: str, video_name: str) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        elif not any(video["fileName"] == video_name for video in self.video_json[client_id]):
            return {"message": f"Video {video_name} not found for client {client_id}"}
        old_client_videos = deepcopy(self.get_client_videos(client_id))
        self.video_json[client_id] = [video for video in self.video_json[client_id] if video["fileName"] != video_name]
        self.save_json()
        return self.create_patch(old_client_videos, self.video_json[client_id])
        
    def add_annotation(self, client_id: str, video_name: str, status: str, message: str, timestamp: str) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        elif not any(video["fileName"] == video_name for video in self.video_json[client_id]):
            return {"message": f"Video {video_name} not found for client {client_id}"}
        old_client_videos = deepcopy(self.get_client_videos(client_id))
        video = next(video for video in self.video_json[client_id] if video["fileName"] == video_name)
        annotation = self.annotation_template.copy()
        annotation["type"] = status
        annotation["message"] = message
        annotation["timestamp"] = timestamp
        video["annotations"].append(annotation)
        video["status_counts"][status] += 1
        self.save_json()
        return self.create_patch(old_client_videos, self.video_json[client_id])
        
    def clear_annotations(self, client_id: str, video_name: str) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        elif not any(video["fileName"] == video_name for video in self.video_json[client_id]):
            return {"message": f"Video {video_name} not found for client {client_id}"}
        old_client_videos = deepcopy(self.get_client_videos(client_id))
        video = next(video for video in self.video_json[client_id] if video["fileName"] == video_name)
        video["annotations"] = []
        video["status_counts"] = {
            "correct": 0,
            "warning": 0,
            "error": 0
        }
        self.save_json()
        return self.create_patch(old_client_videos, self.video_json[client_id])
        
    def get_video(self, client_id: str, video_name: str) -> dict:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        elif not any(video["fileName"] == video_name for video in self.video_json[client_id]):
            return {"message": f"Video {video_name} not found for client {client_id}"}
        return next(video for video in self.video_json[client_id] if video["fileName"] == video_name)
    
    def get_client_videos(self, client_id: str) -> list[dict]:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        return self.video_json[client_id]
    
    def get_all_videos(self) -> dict:
        return self.video_json
    
    def create_patch(self, old_json: dict, new_json: dict) -> jsonpatch.JsonPatch:
        return jsonpatch.JsonPatch.from_diff(old_json, new_json)
    
    def apply_patch(self, client_id: str, patch: jsonpatch.JsonPatch) -> jsonpatch.JsonPatch:
        if client_id not in self.video_json:
            return {"message": f"Client ID {client_id} not found"}
        old_client_videos = self.get_client_videos(client_id)
        new_client_videos = patch.apply(old_client_videos)
        self.video_json[client_id] = new_client_videos
        self.save_json()
        return new_client_videos
        
    def load_json(self, json_path: str = None) -> dict:
        if json_path is None:
            json_path = self.json_path
        with open(json_path, 'r') as f:
            return json.load(f)
        
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
    print(vjm)
    # print(vjm.add_video("client1", "video2.mp4", "/path/to/video2.mp4"))
    # print(vjm.add_annotation("client1", "video2.mp4", "warning", "This is a warning", "00:00:05"))
    # print(vjm.clear_annotations("client1", "video2.mp4"))
    # print(vjm.remove_video("client1", "video2.mp4"))
    # print(vjm.remove_client("client1"))
    