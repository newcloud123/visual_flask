import subprocess

def convert_to_browser_compatible_video(input_path, output_path):
    """
    将视频转换为浏览器兼容的H.264格式
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :return: 转换是否成功
    """
    try:
        # 使用FFmpeg转换为浏览器兼容的H.264格式
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-i', input_path,  # 输入文件
            '-c:v', 'libx264',  # H.264编码
            '-preset', 'fast',  # 编码速度
            '-crf', '23',  # 质量参数
            '-movflags', '+faststart',  # 优化网络播放
            '-pix_fmt', 'yuv420p',  # 浏览器兼容的像素格式
            '-c:a', 'copy',  # 复制音频流（如果有）
            output_path  # 输出文件
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg转换失败: {e.output.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"视频转换错误: {str(e)}")
        return False