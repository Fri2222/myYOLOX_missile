##功能为将视频每隔I帧切片为图片

##现在你的脚本彻底进化成了命令行工具，以后不管跑哪个视频，一行代码搞定。

##在你打开的终端（或 Anaconda Prompt）中，直接输入以下命令并回车：


##python extract_frames.py -p "H:\Missile_Video_Dataset\Dataset_FixedView_20260414_114331\flight_video.avi"

##如果你觉得间隔 3 帧太密了，想每 5 帧抽一次，只需加上 -i 参数：


##python extract_frames.py -p "H:\Missile_Video_Dataset\Dataset_FixedView_20260414_114331\flight_video.avi" -i 5





import cv2
import os
import argparse
import re

def make_parser():
    """定义命令行参数"""
    parser = argparse.ArgumentParser("视频抽帧自动化脚本")
    
    # 只需要输入这一个核心参数：视频文件的绝对路径
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="要抽帧的视频文件绝对路径"
    )
    
    # 可选参数：抽帧间隔，默认是 3
    parser.add_argument(
        "-i", "--interval", type=int, default=3, help="抽帧间隔 (默认: 3)"
    )
    return parser

def extract_time_string(video_path):
    """
    智能提取：从类似 Dataset_FixedView_20260414_114331 的路径中
    精确提取出 20260414_114331 这串时间戳。
    """
    # 查找连续的 8位数字_6位数字 格式
    match = re.search(r"(\d{8}_\d{6})", video_path)
    if match:
        return match.group(1)
    else:
        # 如果没找到标准时间戳，退而求其次，提取视频所在的父文件夹名
        print("⚠️ 未找到标准时间戳格式，将使用父文件夹名称")
        return os.path.basename(os.path.dirname(video_path))

def extract_frames(video_path, frame_interval):
    # 1. 动态生成输出路径
    time_str = extract_time_string(video_path)
    
    # 拼接出你要求的专属文件夹路径
    output_dir = os.path.join(r"H:\Missile_Picture_Dataset", f"Images_{time_str}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 2. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频文件，请检查路径是否正确: {video_path}")
        return

    frame_count = 0  
    saved_count = 0  

    print(f"🎬 开始处理视频: {video_path}")
    print(f"📁 截图将自动保存至: {output_dir}")
    print(f"⏱️ 抽取间隔: 每 {frame_interval} 帧抽取 1 张")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 3. 执行按间隔抽帧保存
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            save_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"  ...已保存 {saved_count} 张截图")
            
        frame_count += 1

    cap.release()
    print(f"\n✅ 处理完成！")
    print(f"📊 视频总长: {frame_count} 帧")
    print(f"🖼️ 成功保存: {saved_count} 张截图在 {output_dir}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    extract_frames(args.path, args.interval)