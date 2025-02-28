import os
import math
import torch
import whisper
import torchaudio
import pandas as pd
import json
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from pyannote.core import Segment
from transformers import pipeline  # Hugging Face の pipeline を使用

# --- 各種パスの設定 ---
BASE_DIR = 'your_directory'
output_dir = os.path.join(BASE_DIR, 'video_as_data_PTS')
audio_path = os.path.join(output_dir, 'audio.wav')

# --- 話者ダイアリゼーションのパイプライン ---
pipeline_diary = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="your_token"
)

# --- Whisper による文字起こし --- 
whisper_model = whisper.load_model("medium")
asr_result = whisper_model.transcribe(audio_path)

# --- ダイアリゼーション結果の取得 ---
diarization_result = pipeline_diary(audio_path)
final_result = diarize_text(asr_result, diarization_result)
# final_result は各要素 (Segment, speaker, sentence) のタプル

# ==============================
# ここから10分（600秒）ごとのダイアリゼーション結果CSV出力処理
# ==============================

chunk_duration = 600  # seconds
max_time = max(seg.end for seg, spk, sent in final_result)
num_chunks = math.ceil(max_time / chunk_duration)

for i in range(num_chunks):
    start_interval = i * chunk_duration
    end_interval = (i + 1) * chunk_duration
    # セグメントの開始時刻が該当区間に入るものを抽出
    chunk_results = [
        (seg, spk, sent)
        for seg, spk, sent in final_result
        if seg.start >= start_interval and seg.start < end_interval
    ]
    if chunk_results:
        chunk_df = pd.DataFrame([
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": spk,
                "sentence": sent
            }
            for seg, spk, sent in chunk_results
        ])
        chunk_csv_path = os.path.join(output_dir, f"diarize_text_result_chunk_{i+1}.csv")
        chunk_df.to_csv(chunk_csv_path, index=False, encoding="utf-8-sig")
        print(f"Chunk {i+1} diarization CSV saved: {chunk_csv_path}")

# 全体のダイアリゼーション結果をCSV出力
df_diarization = pd.DataFrame([
    {
        "start": seg.start,
        "end": seg.end,
        "speaker": spk,
        "sentence": sent
    }
    for seg, spk, sent in final_result
])
full_csv_path = os.path.join(output_dir, "diarize_text_result_full.csv")
df_diarization.to_csv(full_csv_path, index=False, encoding="utf-8-sig")
print(f"Full diarization CSV saved: {full_csv_path}")

# ==============================
# ここから感情認識処理へ移行
# ==============================

# --- 音声ファイル全体の読み込み (感情認識のため) ---
waveform, sr = torchaudio.load(audio_path)

# --- Hugging Face pipeline を用いた音声感情認識 ---
# 必要に応じて use_auth_token パラメータを追加してください
emotion_classifier = pipeline(
    "audio-classification",
    model="Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition"
)

results_with_emotion = []  # 各要素 (Segment, speaker, sentence, emotion_detail) のタプルリスト
for seg, spk, sent in final_result:
    # セグメントに対応するサンプルインデックスの計算
    start_sample = int(seg.start * sr)
    end_sample = int(seg.end * sr)
    segment_audio = waveform[:, start_sample:end_sample]

    # モノラル入力に変換（複数チャネルの場合は平均を取る）
    if segment_audio.shape[0] > 1:
        segment_audio = torch.mean(segment_audio, dim=0, keepdim=True)
    # Tensor を numpy array に変換
    segment_audio_np = segment_audio.squeeze(0).cpu().numpy()

    # pipeline への入力は、辞書形式で "array" と "sampling_rate" を指定
    audio_input = {"array": segment_audio_np, "sampling_rate": sr}

    try:
        prediction = emotion_classifier(audio_input)
        # prediction はリスト形式（各感情のスコア・ラベルの内訳）
        predicted_emotion_detail = json.dumps(prediction, ensure_ascii=False)
    except Exception as e:
        print(f"Error during classification for segment starting at {seg.start:.2f}: {e}")
        predicted_emotion_detail = "error"

    results_with_emotion.append((seg, spk, sent, predicted_emotion_detail))
    print(f'{seg.start:.2f} {seg.end:.2f} {spk} {sent} [Emotion: {predicted_emotion_detail}]')

# ==============================
# ここから感情認識結果の10分毎のCSV追記保存処理
# ==============================

# 感情認識結果を追記するファイルのパスを定義
cumulative_emotion_csv = os.path.join(output_dir, "diarize_text_emotion_result_cumulative.csv")
# 既存のファイルがあれば削除（初回は新規作成）
if os.path.exists(cumulative_emotion_csv):
    os.remove(cumulative_emotion_csv)

num_chunks_emotion = math.ceil(max(seg.end for seg, _, _, _ in results_with_emotion) / chunk_duration)
for i in range(num_chunks_emotion):
    start_interval = i * chunk_duration
    end_interval = (i + 1) * chunk_duration
    # チャンクに含まれる結果を抽出（セグメントの開始時刻でグループ化）
    chunk_emotion_results = [
        (seg, spk, sent, emotion)
        for seg, spk, sent, emotion in results_with_emotion
        if seg.start >= start_interval and seg.start < end_interval
    ]
    if chunk_emotion_results:
        df_emotion_chunk = pd.DataFrame([
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": spk,
                "sentence": sent,
                "emotion": emotion  # 内訳の JSON 文字列
            }
            for seg, spk, sent, emotion in chunk_emotion_results
        ])
        # 既存の cumulative CSV に追記（ファイルがなければ新規作成）
        if not os.path.exists(cumulative_emotion_csv):
            df_emotion_chunk.to_csv(cumulative_emotion_csv, index=False, encoding="utf-8-sig")
        else:
            df_emotion_chunk.to_csv(cumulative_emotion_csv, index=False, encoding="utf-8-sig", mode='a', header=False)
        print(f"Appended emotion CSV chunk for {i+1} chunk(s): {cumulative_emotion_csv}")

# --- 全体の感情認識結果をまとめた CSV の出力 ---
df_emotion_full = pd.DataFrame([
    {
        "start": seg.start,
        "end": seg.end,
        "speaker": spk,
        "sentence": sent,
        "emotion": emotion  # 内訳の JSON 文字列
    }
    for seg, spk, sent, emotion in results_with_emotion
])
emotion_full_csv_path = os.path.join(output_dir, "diarize_text_emotion_result_full.csv")
df_emotion_full.to_csv(emotion_full_csv_path, index=False, encoding="utf-8-sig")
print(f"Full emotion CSV saved: {emotion_full_csv_path}")
