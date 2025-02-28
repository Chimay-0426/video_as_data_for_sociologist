
from moviepy.editor import VideoFileClip
import cv2
import os
import pandas as pd

# パス設定
BASE_DIR = 'your_directory'  # Googleドライブのベースディレクトリ
output_dir = os.path.join(BASE_DIR, 'video_as_data_PTS')  # 出力用ディレクトリ
os.makedirs(output_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)

# ファイルパス設定
video_path = os.path.join(output_dir, 'video.mp4')  # 動画ファイル

# 映像処理: フレーム抽出 (並列処理対応)
def extract_frames():
    video = VideoFileClip(video_path)  
    frame_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    frame_list = []
    for t in range(0, int(video.duration)):
        frame = video.get_frame(t)
        frame_path = os.path.join(frame_dir, f"frame_{t:04d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_list.append([t, frame_path])
    df_frames = pd.DataFrame(frame_list, columns=["Timestamp", "Path"])
    df_frames.to_csv(os.path.join(output_dir, 'frames.csv'), index=False, encoding='utf-8-sig')
    print(f"フレーム抽出完了: {len(frame_list)} 枚")
    return df_frames


# メイン実行
if __name__ == "__main__":

    # 映像処理
    print("フレーム抽出を実行中...")
    df_frames = extract_frames()

    print("出力ファイル:")
    print(" - frames.csv")
    