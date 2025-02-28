import os
import cv2
import ast
import base64
import numpy as np
import pandas as pd
from deepface import DeepFace
import deepface.commons as functions
import hdbscan

def _crop_face(frame_path, bbox_list):
    """
    frame_path と [x, y, w, h] から顔画像を切り出して返す (OpenCV BGR)
    """
    x, y, w, h = bbox_list
    img_bgr = cv2.imread(frame_path)
    if img_bgr is None:
        return None
    height, width = img_bgr.shape[:2]

    x2 = x + w
    y2 = y + h
    if x < 0: x = 0
    if y < 0: y = 0
    if x2 > width: x2 = width
    if y2 > height: y2 = height

    if x2 <= x or y2 <= y:
        return None

    face_bgr = img_bgr[y:y2, x:x2]
    return face_bgr

def analyze_frames_dir(
    frames_dir,
    output_csv,
    actions=["emotion","age","gender"],  # DeepFace.analyzeで解析する属性
    model_name="Facenet512",
    detector_backend="retinaface",       # "retinaface", "mtcnn", "mediapipe", "opencv" など
    enforce_detection=True,              
    min_cluster_size=2
):
    """
    指定ディレクトリ内のフレーム画像を1枚ずつ DeepFace.analyze にかけ、
    - 1つのフレームに複数の顔がある場合は全て解析
    - 各顔について BoundingBox, 感情内訳, 年齢, 性別内訳, Embedding, Base64画像を取得
    - HDBSCAN で埋め込みをクラスタリング (Cluster 列)
    - CSV に書き出す
    """
    rows = []
    face_counter = 0

    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in frame_files:
        if not frame_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        frame_path = os.path.join(frames_dir, frame_file)

        try:
            # DeepFace.analyzeでフル画像を解析 (複数顔があれば list で返る)
            results = DeepFace.analyze(
                img_path=frame_path,
                actions=actions,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection
            )
        except Exception as e:
            print(f"[WARN] Could not analyze {frame_file}: {e}")
            continue

        # 複数顔か単数かで返り値が違うため、単数なら list に包む
        if isinstance(results, dict):
            results = [results]

        # フレーム内の全ての顔を処理
        for face_data in results:
            region = face_data.get("region", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)
            bbox_list = [x, y, w, h]

            # 感情, 年齢, 性別の内訳を取得
            dominant_emotion = face_data.get("dominant_emotion", "unknown")
            emotions_dict     = face_data.get("emotion", {})
            age_val           = face_data.get("age", None)
            gender_dict       = face_data.get("gender", {})

            # FaceID
            face_id = f"{os.path.splitext(frame_file)[0]}_face_{face_counter}"
            face_counter += 1

            # 埋め込みのために顔をクロップ
            cropped_bgr = _crop_face(frame_path, bbox_list)
            if cropped_bgr is not None and cropped_bgr.size > 0:
                # Embedding
                try:
                    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                    rep = DeepFace.represent(
                        img_path=cropped_rgb,
                        model_name=model_name,
                        enforce_detection=False,
                        detector_backend="opencv"
                    )
                    if rep and isinstance(rep, list) and len(rep) > 0:
                        embedding = rep[0].get("embedding", None)
                    else:
                        embedding = None
                except:
                    embedding = None

                # Base64画像
                retval, buffer = cv2.imencode(".jpg", cropped_bgr)
                if retval:
                    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
                    base64_img = f"data:image/jpeg;base64,{jpg_as_text}"
                else:
                    base64_img = ""
            else:
                embedding = None
                base64_img = ""

            rows.append({
                "Frame": frame_file,
                "FaceID": face_id,
                "BoundingBox": str(bbox_list),
                "DominantEmotion": dominant_emotion,
                "Emotions": emotions_dict,
                "Gender": gender_dict,
                "Age": age_val,
                "Embedding": embedding,
                "Image": base64_img
            })

    # DataFrame化
    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No faces detected or no valid data to output.")
        df.to_csv(output_csv, index=False)
        return df

    # HDBSCAN クラスタリング
    valid_indices = []
    valid_embeddings = []
    for idx, emb in df["Embedding"].items():
        if emb is not None:
            valid_indices.append(idx)
            valid_embeddings.append(emb)

    if len(valid_embeddings) > 0:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(np.array(valid_embeddings))
        df["Cluster"] = -1
        for i, c_label in zip(valid_indices, cluster_labels):
            df.at[i, "Cluster"] = c_label
    else:
        df["Cluster"] = -1

    # カラム順を整理
    df = df[[
        "Frame", "FaceID", "BoundingBox",
        "DominantEmotion", "Emotions", "Gender", "Age",
        "Embedding", "Image", "Cluster"
    ]]

    # CSV出力
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"CSV saved to {output_csv}, total {len(df)} faces.")

    return df

def save_cluster_images(df, base_output_dir):
    """
    DataFrame に含まれる行を Cluster 別にフォルダに分けて、"Image" (Base64) をjpgファイルとして保存。
    - base_output_dir/cluster_0/FaceID.jpg
    - base_output_dir/cluster_1/FaceID.jpg
      ...
    - base_output_dir/cluster_-1/...  (embedding取れなかった等)
    """
    os.makedirs(base_output_dir, exist_ok=True)
    unique_clusters = df["Cluster"].unique()

    # クラスタごとにフォルダを作る
    for c in unique_clusters:
        cluster_dir = os.path.join(base_output_dir, f"cluster_{c}")
        os.makedirs(cluster_dir, exist_ok=True)

    # 各行の Image (Base64) をデコードして jpg で保存。以下の部分はイマイチdecodeがうまくいって誘うだが、ただFace_IDをキーに識別できる。
    for idx, row in df.iterrows():
        cluster_label = row["Cluster"]
        face_id = row["FaceID"]
        image_b64 = row["Image"]  # data:image/jpeg;base64,XXXX

        if image_b64 and image_b64.startswith("data:image"):
            # "data:image/jpeg;base64," の後ろが実際のbase64
            base64_string = image_b64.split(",", 1)[1]
            image_data = base64.b64decode(base64_string)

            out_dir = os.path.join(base_output_dir, f"cluster_{cluster_label}")
            out_path = os.path.join(out_dir, f"{face_id}.jpg")
            with open(out_path, "wb") as f:
                f.write(image_data)

    print(f"Images saved by cluster under: {base_output_dir}")

# ============================================================================
# 実行
# ============================================================================
if __name__ == "__main__":
    frames_dir = "（framesが存在するdirectory）"
    output_csv = "directoru for output"
    cluster_img_dir = "(クラスタごとの顔を収録するdirectoru)"

    df_result = analyze_frames_dir(
        frames_dir=frames_dir,
        output_csv=output_csv,
        actions=["emotion","age","gender"],
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=True,
        min_cluster_size=2
    )
    # クラスタごとの画像出力
    #save_cluster_images(df_result, cluster_img_dir)

    display(df_result.head(10))  

