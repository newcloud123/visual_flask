from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import uuid
from datetime import datetime
import cv2
import numpy as np
from utils import convert_to_browser_compatible_video
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB限制

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 支持的算法列表
ALGORITHMS = [
    "yolo11_m_std",
    "目标检测",
    "语义分割",
    "实例分割",
    "姿态估计",
    "自定义算法"
]



@app.route('/')
def index():
    return render_template('index.html', algorithms=ALGORITHMS)

@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传接口"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # 生成唯一文件名
        ext = os.path.splitext(file.filename)[1]
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            "filename": filename,
            "url": url_for('uploaded_file', filename=filename)
        })

@app.route('/uploads/')
def uploaded_file(filename):
    """提供上传文件访问"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/infer', methods=['POST'])
def infer():
    """推理接口 - 用户需要在此实现自己的推理逻辑"""
    data = request.json
    
    # 获取参数
    filename = data.get('filename')
    algorithm = data.get('algorithm')
    prompt = data.get('prompt', '')
    additional_info = data.get('additional_info', '')
    
    # 文件路径
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if algorithm == "yolo11_m_std":
        from yolo_det_onnx import perform_inference
        result_path = perform_inference(input_path, algorithm, prompt, additional_info)
    else:
        # 示例处理 - 这里需要替换为实际推理逻辑
        result_path = perform_inference(input_path, algorithm, prompt, additional_info)
    print("result_path = ",result_path)
    print("filename  = ",os.path.basename(result_path))
    return jsonify({
        "result_url": url_for('result_file', filename=os.path.basename(result_path))
    })

def perform_inference(input_path, algorithm, prompt, additional_info):
    """示例推理函数 - 用户需要实现自己的逻辑"""
    # 这里仅做示例：将图片转为灰度图作为结果
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(input_path)
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 保存结果
        output_filename = f"result_{os.path.basename(input_path)}"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        cv2.imwrite(output_path, result)
        
        return output_path
    
    # 视频处理示例 - 仅做占位
    elif input_path.lower().endswith(('.mp4', '.avi')):
        posttype = input_path.rsplit('.')[-1]
        new_path = input_path.replace('.'+ posttype,'lsy.'+posttype).replace('uploads','results')
         # 转换为浏览器兼容的H.264格式
        print(f"开始转换视频 {input_path} 为H.264格式...")
        if convert_to_browser_compatible_video(input_path, new_path):
            # 删除临时文件
            # os.remove(temp_path)
            print(f"视频 {input_path} 处理完成")
     
        return new_path
        # return "Video processing would happen here"
    
    return input_path  # 返回原始文件作为占位



@app.route('/results/<filename>')          # ① 原来缺少 <>
def result_file(filename):
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(path):
        return "File not found", 404
    # ② 明确告诉浏览器这是一张图片，让它直接渲染
    return send_file(path, mimetype='image/jpeg')

@app.route('/save-result', methods=['POST'])
def save_result():
    """保存结果接口 - 用户可扩展保存逻辑"""
    data = request.json
    filename = data.get('filename')
    
    # 这里可以添加保存到数据库或其他存储的逻辑
    # 示例中仅返回成功    
    return jsonify({
        "status": "success",
        "message": f"Result {filename} saved successfully"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001, debug=True)