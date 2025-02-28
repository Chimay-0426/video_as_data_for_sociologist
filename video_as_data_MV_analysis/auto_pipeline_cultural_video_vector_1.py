import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
# pyscenedetect (scenedetect) 関連
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
# Transformersでビジョンモデルを利用
from transformers import ViTImageProcessor, ViTModel


# -----------------------
# GPU設定＆モデル準備
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_name)
vit_model = ViTModel.from_pretrained(model_name).to(device)
vit_model.eval()

# -----------------------
# シーン検出
# -----------------------
def detect_scenes(video_path, threshold=8.0):
    """
    threshold=30.0 は ContentDetector のカット検出しきい値(デフォルト)。
    return: [(start_sec, end_sec), ...] のリスト
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor(1)  # 必要に応じて縮小
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    scene_ranges = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scene_ranges.append((start_time, end_time))

    video_manager.release()
    return scene_ranges

# -----------------------
# フレームサンプリング
# -----------------------
def sample_frames_from_scene(video_path, start_sec, end_sec, num_frames=3):
    """
    シーン区間 [start_sec, end_sec] の間で均等に num_frames 枚フレームをサンプリング。
    ffmpeg サブプロセスでフレームを一時ファイル出力 → OpenCV で読み込み。
    return: PIL Image のリスト
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return []

    # 抽出時刻を等間隔に計算
    timestamps = np.linspace(start_sec, end_sec, num_frames+2)[1:-1]

    images = []
    for ts in timestamps:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            tmp_image_path = tmpfile.name

        cmd = [
            'ffmpeg',
            '-ss', str(ts),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',  # 画質設定
            '-y', tmp_image_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        frame_bgr = cv2.imread(tmp_image_path)
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            images.append(pil_img)

        if os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)

    return images

# -----------------------
# ViTによるベクトル化
# -----------------------
def extract_vit_features(image_list, image_processor, vit_model, device="cuda"):
    """
    image_list: PIL Image のリスト
    return: 画像ごとのCLSトークンベクトル (shape=(len(image_list), hidden_dim))
    """
    if len(image_list) == 0:
        return []

    inputs = image_processor(images=image_list, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vit_model(**inputs)  # [batch_size, seq_len, hidden_size]

    last_hidden = outputs.last_hidden_state  # (batch, seq, hidden_size)
    cls_embeddings = last_hidden[:, 0, :]    # CLSトークンのみ抽出

    return cls_embeddings.cpu().numpy()  # numpy array に変換

# -----------------------
# シーンの平均 → PV全体ベクトル
# -----------------------
def get_scene_feature(video_path, start_sec, end_sec,
                      image_processor, vit_model, device="cuda", num_frames=3):
    """
    シーン [start_sec, end_sec] でフレーム num_frames枚サンプリング → ViT特徴を平均しシーンの代表ベクトルを返す
    """
    images = sample_frames_from_scene(video_path, start_sec, end_sec, num_frames=num_frames)
    if len(images) == 0:
        return None

    feats = extract_vit_features(images, image_processor, vit_model, device=device)
    return feats.mean(axis=0)  # (hidden_size,)

def get_pv_vector(video_path, scene_threshold=1.0, num_frames=3):
    """
    1本のPVを読み込み:
      - シーン検出 (scene_threshold)
      - 各シーンで複数フレームをサンプリング(num_frames)→ ViTベクトル化 → シーン平均
      - 全シーンのベクトルをさらに平均してPV全体の1ベクトルを得る
    """
    scenes = detect_scenes(video_path, threshold=scene_threshold)
    scene_vectors = []
    for (start_sec, end_sec) in scenes:
        feat = get_scene_feature(video_path, start_sec, end_sec,
                                 image_processor, vit_model,
                                 device=device, num_frames=num_frames)
        if feat is not None:
            scene_vectors.append(feat)

    if len(scene_vectors) == 0:
        return None

    # 全シーンのベクトルを平均
    pv_vector = np.mean(scene_vectors, axis=0)
    return pv_vector

# ---------------------------------------------------
# 2. df_mvs のPVを処理
# ---------------------------------------------------

VIDEO_DIR = '（映像mp4があるdirectory）' #例えば'/content/drive/MyDrive/video_as_data_MV_analysis'


df_mvs_vector = df_mvs.copy()

pv_features = []
for idx, row in df_mvs_vector.iterrows():
    #以下のidの構成などは対象によって要操作。
    video_id = row['id']
    video_filename = f"{video_id}_video.mp4"  # 例: "LbtQM793jn8_video.mp4"のような名前を想定
    video_path = os.path.join(VIDEO_DIR, video_filename)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        pv_features.append((video_id, None))
        continue

    print(f"Processing video: {video_filename}")
    pv_vec = get_pv_vector(video_path, scene_threshold=30.0, num_frames=3)
    pv_features.append((video_id, pv_vec))

# 結果をDataFrameに格納
df_features = pd.DataFrame(pv_features, columns=["id", "pv_vector"])
df_features.head()

