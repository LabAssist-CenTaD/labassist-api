import json
from app.schemas.annotation import Annotation
from app.utils.object_detection_utils import get_objects, get_biggest_boxes, calculate_iou, get_valid_tile

def compile_annotations(results: list) -> list[Annotation]:
    annotations = []
    annotations.extend(process_swirling(results))
    annotations.extend(process_goggles(results))
    annotations.extend(process_tile(results))
    return annotations

def process_goggles(results: list) -> list[Annotation]:
    annotations = []
    group_flag = False
    group_start = 0 
    for result in results:
        object_pred = json.loads(result["object_pred"])
        faces = get_objects(object_pred, 'Face')
        goggles = get_objects(object_pred, 'Lab-goggles')
        
        if faces and goggles:
            faces = get_biggest_boxes(faces)
            goggles = get_biggest_boxes(goggles)
            iou = calculate_iou(
                [  faces['x1'],   faces['y1'],   faces['x2'],   faces['y2']], 
                [goggles['x1'], goggles['y1'], goggles['x2'], goggles['y2']]
            )
            if iou < 0.1:
                ann_type = "error"
                message = "Goggles should be worn properly"
                group_start = result["start_seconds"]
                group_flag = True
            elif group_flag:
                group_flag = False
                if result['start_seconds'] - group_start > 10:
                    annotations.append(Annotation(type=ann_type, message=message, start_seconds=group_start, end_seconds=result["start_seconds"]))
        elif group_flag:
            group_flag = False
            if result['start_seconds'] - group_start > 10:
                annotations.append(Annotation(type=ann_type, message=message, start_seconds=group_start, end_seconds=result["start_seconds"]))
    if group_start - result['start_seconds'] > 10:
            annotations.append(Annotation(type=ann_type, message=message, start_seconds=group_start, end_seconds=result["start_seconds"]))
            
    return annotations

def process_swirling(results: list) -> list[Annotation]:
    annotations = []
    message = None
    group_type = None
    group_start = None
    for result in results:
        # print(f'{result["action_pred"]} {group_type} {group_start}')
        if result["action_pred"] != group_type:
            if group_start and result["start_seconds"] - group_start> 6:
                annotations.append(Annotation(type=ann_type, message=message, start_seconds=group_start, end_seconds=result["start_seconds"]))
            group_start = None
        if result["action_pred"]:
            if result["action_pred"] == "Correct":
                ann_type = "info"
                message = "Correct swirling detected"
            elif result["action_pred"] == "Incorrect":
                ann_type = "error"
                message = "Conical flask should not be grinded on the white tile"
            elif result["action_pred"] == "Stationary":
                ann_type = "warning"
                message = "Conical flask should be swirled to ensure proper mixing"
            if not group_start:
                group_start = result["start_seconds"]
        else:
            message = None
        group_type = result["action_pred"]
    if result["action_pred"] != group_type and group_start and result["start_seconds"] - group_start> 6:
        annotations.append(Annotation(type=ann_type, message=message, start_seconds=group_start, end_seconds=result["start_seconds"]))
    return annotations

def process_tile(results: list) -> list[Annotation]:
    annotations = []
    group_flag = False
    group_start = 0 
    for result in results:
        if result['action_pred'] in ['Correct', 'Incorrect'] and not get_valid_tile(result["object_pred"]):
            group_flag = True
            group_start = result['start_seconds']
        elif group_flag and result['start_seconds'] - group_start > 6:
            annotations.append(Annotation(type='error', message='Conical flask should be placed on the white tile', start_seconds=group_start, end_seconds=result['start_seconds']))
            group_flag = False
    if group_flag and result['start_seconds'] - group_start > 6:
        annotations.append(Annotation(type='error', message='Conical flask should be placed on the white tile', start_seconds=group_start, end_seconds=result['start_seconds']))
    return annotations