import cv2
import numpy as np
import torch
from PIL import Image
import av

model_name = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_name)
VIDEO_DIR = '（映像mp4があるdirectory）' #例えば'/content/drive/MyDrive/video_as_data_MV_analysis'


def get_video_vector_by_frames(
    video_path,
    step_sec=1.0,
    out_width=1280,
    image_processor=None,
    vit_model=None,
    device="cuda"
):
    """
    動画を開き、step_sec(秒)ごとにフレームを抽出してViTベクトル化し、
    全フレームの平均ベクトルを返す。

    Args:
        video_path (str): 動画ファイルのパス
        step_sec (float): 何秒おきにフレームを取得するか
        out_width (int): フレームをリサイズするときの横幅 (縦はアスペクト比に応じて計算)
        image_processor, vit_model: HuggingFaceのビジョンモデル関連
        device (str): 'cuda' か 'cpu'

    Returns:
        np.ndarray or None: shape=(hidden_dim,) のベクトル (全フレーム平均)
                            フレーム取得に失敗した場合などは None
    """
    if image_processor is None or vit_model is None:
        print("Error: image_processor or vit_model is not provided.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    # 総フレーム数とFPSを取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fpsが0なら暫定的に30FPSとみなす

    duration_sec = total_frames / fps

    # 取得するタイムスタンプを計算
    # 例: step_sec=1なら [0, 1, 2, ..., floor(duration_sec)]
    timestamps = np.arange(0, duration_sec, step_sec)

    all_features = []
    for t in timestamps:
        # ミリ秒に変換して指定
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # 読み込めないケース（可変フレームレート等）で失敗する場合がある
            # 必要に応じてwarnを表示する
            # print(f"Warning: Failed to read frame at {t:.2f}s from {video_path}")
            continue

        # OpenCVはBGRなのでRGBへ
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # リサイズ (幅out_width, 高さはアスペクト比に合わせ自動計算)
        if out_width is not None:
            h, w, _ = frame_rgb.shape
            if w != 0:
                out_height = int(h * (out_width / w))
                frame_rgb = cv2.resize(frame_rgb, (out_width, out_height), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(frame_rgb)

        # ViTでベクトル化
        inputs = image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = vit_model(**inputs)
        # CLSトークンを取得
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # shape=(1, hidden_dim)
        all_features.append(cls_emb[0])

    cap.release()

    if len(all_features) == 0:
        return None

    # 全フレームベクトルを平均
    all_features = np.array(all_features)  # shape=(N, hidden_dim)
    video_vector = all_features.mean(axis=0)  # shape=(hidden_dim,)
    return video_vector


def sample_frames_pyav(
    video_path,
    skip_frames=30,
    downscale=2
):
    """
    PyAVを使って動画を連続読み込みしつつ、
    指定したフレーム間引き (skip_frames) でフレームをピックアップし、
    PIL Image のリストにして返す。

    Args:
        video_path (str): 動画ファイルのパス
        skip_frames (int): フレームを何枚飛ばしで読み込むか
            (例: 30なら1秒前後おきに取得、24fpsの場合)
        downscale (int): 画像を縮小する倍率(2なら縦横1/2)

    Returns:
        list of PIL.Image: フレーム画像(PIL)のリスト
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # フレームレートがわからない場合は仮に30FPSとする。youtubeなどでメタデータを取得するfpsも取得できる。
    fps = video_stream.average_rate
    if fps is None or fps == 0:
        fps = 30

    images = []
    frame_count = 0

    for packet in container.demux(video_stream):
        for frame in packet.decode():
            if frame.pts is None:
                continue
            # skip_frames間隔でのみ採用
            if frame_count % skip_frames == 0:
                # frameをndarray (RGBにするため "rgb24") に変換
                arr = frame.to_ndarray(format="rgb24")  # shape=(H, W, 3)
                if downscale > 1:
                    h, w, c = arr.shape
                    arr = cv2.resize(
                        arr,
                        (w // downscale, h // downscale),
                        interpolation=cv2.INTER_AREA
                    )
                # PILに変換
                pil_img = Image.fromarray(arr)
                images.append(pil_img)

            frame_count += 1

    container.close()
    return images


def get_pv_vector_frames(
    video_path,
    skip_frames=30,
    downscale=2,
    image_processor=None,
    vit_model=None,
    device="cuda"
):
    """
    フレームを定期サンプリングしてViTでベクトル化→全フレームの平均ベクトルを返す。
    """
    # 1) フレーム抽出 (PIL Images)
    images = sample_frames_pyav(
        video_path,
        skip_frames=skip_frames,
        downscale=downscale
    )

    if len(images) == 0:
        print(f"No frames extracted from {video_path}")
        return None

    # 2) ViT埋め込み
    #    下記 extract_vit_features はすでに定義済みと想定
    feats = extract_vit_features(images, image_processor, vit_model, device=device)
    if len(feats) == 0:
        return None

    # 3) 全フレームのベクトルを平均
    return np.mean(feats, axis=0)  # shape=(hidden_dim,)

df_mvs["pv_vector"] = None
# もし 'pv_vector' カラムが存在しないなら、None で初期化
# 1) pv_vectorカラムをobject型で用意
if 'pv_vector' not in df_mvs.columns:
    df_mvs['pv_vector'] = None
df_mvs['pv_vector'] = df_mvs['pv_vector'].astype(object)

for idx, row in df_mvs.iterrows():
    video_id = row['id']
    video_filename = f"{video_id}_video.mp4"
    video_path = os.path.join(VIDEO_DIR, video_filename)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        continue

    print(f"Re-processing video (index={idx}): {video_filename}")

    pv_vec = get_pv_vector_frames(
        video_path,
        skip_frames=30,
        downscale=2,
        image_processor=image_processor,
        vit_model=vit_model,
        device=device
    )

    df_mvs.at[idx, 'pv_vector'] = pv_vec


# 確認
df_mvs[["id", "pv_vector"]].head(10)


