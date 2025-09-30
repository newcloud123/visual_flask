import os 
import cv2
from utils import convert_to_browser_compatible_video

import argparse
from typing import Any

import cv2.dnn
import numpy as np
from config import model_config
from app import app 
CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def perform_inference(input_path, algorithm, prompt, additional_info):

    """示例推理函数 - 用户需要实现自己的逻辑"""
    # 这里仅做示例：将图片转为灰度图作为结果
    model_path = model_config[str(algorithm)]
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        result,anno = yolo_det_onnx_infer_img(model_path,input_path)
        # 保存结果
        output_filename = f"result_{os.path.basename(input_path)}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        cv2.imwrite(output_path, result)
        
        return output_path

    # 视频处理示例 - 仅做占位
    elif input_path.lower().endswith(('.mp4', '.avi')):
        posttype = input_path.rsplit('.')[-1]
        new_path = input_path.replace('.'+ posttype,'lsy.'+posttype).replace('uploads','results')
        new_path1 = input_path.replace('.'+ posttype,'lsy1.'+posttype).replace('uploads','results')
        yolo_det_onnx_infer_video(model_path,input_path,new_path)
         # 转换为浏览器兼容的H.264格式
        print(f"开始转换视频 {input_path} 为H.264格式...")
        if convert_to_browser_compatible_video(new_path, new_path1):
            # 删除临时文件
            # os.remove(temp_path)
            print(f"视频 {input_path} 处理完成")
     
        return new_path1
        # return "Video processing would happen here"
    
    return input_path  # 返回原始文件作为占位



def draw_bounding_box(
    img: np.ndarray, class_id: int, confidence: float, x: int, y: int, x_plus_w: int, y_plus_h: int
) -> None:
    """
    Draw bounding boxes on the input image based on the provided arguments.

    Args:
        img (np.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def yolo_det_onnx_infer_img(onnx_model: str, input_image: str) -> list[dict[str, Any]]:
    """
    Load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        (list[dict[str, Any]]): List of dictionaries containing detection information such as class_id, class_name,
            confidence, box coordinates, and scale factor.
    """
    # Load the ONNX model
    if isinstance(onnx_model,str):
        onnx_model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image = None
    if isinstance(input_image,np.ndarray):
        original_image = input_image
    elif isinstance(input_image,str):
        original_image: np.ndarray = cv2.imread(input_image)
    else:
        print("lsy-info | input_image can not recognize!")
    [height, width, _] = original_image.shape


    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    onnx_model.setInput(blob)

    # Perform inference
    outputs = onnx_model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x center - width/2 = left x
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y center - height/2 = top y
                outputs[0][i][2],  # width
                outputs[0][i][3],  # height
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    return original_image,detections
    # cv2.imwrite("image.jpg", original_image)
    # return detections
def yolo_det_onnx_infer_video(model_path,input_path,new_path):
    onnx_model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件")
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    
    # 创建最终结果路径 (H.264编码)
    result_path = new_path
    fourcc =  cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
    res_anno = []
    # 处理每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每10帧打印一次进度
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理视频 {input_path}: 已处理 {frame_count} 帧")
        
        result,anno = yolo_det_onnx_infer_img(onnx_model,frame)
        res_anno.append(anno)
        # 获取渲染后的帧
        annotated_frame = result
        out.write(annotated_frame)
    
    # 释放资源
    cap.release()
    out.release()
    return res_anno