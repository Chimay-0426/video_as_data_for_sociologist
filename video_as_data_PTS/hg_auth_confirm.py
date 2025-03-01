from transformers import pipeline
from pyannote.audio import Model
import os

# Hugging Faceトークン設定
HF_TOKEN = os.getenv('HF_TOKEN')

# トークンチェック
if HF_TOKEN is None or not HF_TOKEN.startswith('hf_'):
    raise ValueError("Hugging Faceのトークンが設定されていません。")

# トークン表示
print(f"設定されたトークン: {HF_TOKEN[:5]}...")

# 1. Hugging Face接続テスト
print("\n1. Hugging Face接続テスト...")
try:
    pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        token=HF_TOKEN
    )
    print("Hugging Face接続テスト成功！")
except Exception as e:
    print(f"Hugging Face接続テスト失敗: {e}")

# 2. pyannote/segmentationモデルのテスト
print("\n2. pyannote/segmentationモデルのテスト...")
try:
    model = Model.from_pretrained(
        "pyannote/segmentation", use_auth_token=HF_TOKEN
    )
    print("pyannote/segmentationモデルのロード成功！")
except Exception as e:
    if 'Access denied' in str(e):
        print("pyannote/segmentationへのアクセス権限がありません。利用条件を承諾してください。")
    elif 'token' in str(e):
        print("トークンが無効です。設定を確認してください。")
    else:
        print(f"pyannote/segmentationモデルのロード失敗: {e}")

# 3. pyannote/speaker-diarizationモデルのテスト
print("\n3. pyannote/speaker-diarizationモデルのテスト...")
try:
    from pyannote.audio import Pipeline
    test_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=HF_TOKEN
    )
    print("pyannote/speaker-diarizationモデルのロード成功！")
except Exception as e:
    if 'Access denied' in str(e):
        print("pyannote/speaker-diarizationへのアクセス権限がありません。利用条件を承諾してください。")
    elif 'token' in str(e):
        print("トークンが無効です。設定を確認してください。")
    else:
        print(f"pyannote/speaker-diarizationモデルのロード失敗: {e}")