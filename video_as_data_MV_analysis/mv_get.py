import yt_dlp
import json
import os
from google.colab import drive

# Googleドライブ上の保存先設定
BASE_DIR = 'your_directory'  #例えば /content/drive/MyDrive/
OUTPUT_DIR = os.path.join(BASE_DIR, 'video_as_data_MV_analysis')  # データ保存ディレクトリ
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(url):
    # -----------------------------
    # 1. メタデータの取得と保存
    # -----------------------------
    ydl_opts_meta = {
        'quiet': True,
        'simulate': True,   # 実際のダウンロードは行わず、メタデータのみ取得
        'dumpjson': True
    }
    with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
        info_dict = ydl.extract_info(url, download=False)

    # 動画のIDを取得（存在しなければ'unknown'）
    video_id = info_dict.get('id', 'unknown')

    # 取得したメタデータをファイルに保存（例：metadata_<video_id>.json）
    metadata_file = os.path.join(OUTPUT_DIR, f'metadata_{video_id}.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)
    print(f"メタデータを保存しました: {metadata_file}")

    # -----------------------------
    # 2. 音声データ（WAV）のダウンロード
    # -----------------------------
    # 出力テンプレートに動画IDを付与して重複を防止
    audio_outtmpl = os.path.join(OUTPUT_DIR, f'{video_id}_audio.%(ext)s')
    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'wav',  # WAV形式で保存
        'outtmpl': audio_outtmpl,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',  # 高音質
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])
    print(f"音声データをダウンロードしました: {os.path.join(OUTPUT_DIR, f'{video_id}_audio.wav')}")

    # -----------------------------
    # 3. 映像データ（MP4）のダウンロード
    # -----------------------------
    # 出力テンプレートに動画IDを付与して重複を防止
    video_outtmpl = os.path.join(OUTPUT_DIR, f'{video_id}_video.%(ext)s')
    ydl_opts_video = {
        'format': 'bestvideo+bestaudio',  # 映像と音声を統合
        'merge_output_format': 'mp4',     # MP4形式で保存
        'outtmpl': video_outtmpl
    }
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([url])
    print(f"映像データをダウンロードしました: {os.path.join(OUTPUT_DIR, f'{video_id}_video.mp4')}")

if __name__ == "__main__":
    # 対象動画のURLリスト
    #official髭男dismのインディーズから2022年前の公式で公開されているMV。一部のvideoは中身の点から取り込まない
    video_urls = [
       # "https://www.youtube.com/watch?app=desktop&v=qrtKLNTB71c",
       # "https://www.youtube.com/watch?app=desktop&v=n0e0y01w92A",
       # "https://www.youtube.com/watch?app=desktop&v=i68HdrqOPcE",
       # "https://www.youtube.com/watch?app=desktop&v=0M-6ImZ9FSk",
        "https://www.youtube.com/watch?app=desktop&v=Vho5jBUfR28",
        "https://www.youtube.com/watch?app=desktop&v=PB8dUVLPnuo",
        "https://m.youtube.com/watch?v=IzyrINr2Xj4",
        "https://www.youtube.com/watch?app=desktop&v=3IDvi4buNdk",
        "https://m.youtube.com/watch?v=1fO0mY5vLA8",
        "https://m.youtube.com/watch?v=0nzgi6dz8VY",
        "https://m.youtube.com/watch?v=EHw005ZqCXk",
        "https://m.youtube.com/watch?v=22mOCjkwQjM",
        "https://m.youtube.com/watch?v=sem3UU-EQJs",
        "https://m.youtube.com/watch?v=TQ8WlA2GXbk",
        "https://m.youtube.com/watch?v=-kgOFJG881I",
        "https://m.youtube.com/watch?v=DuMqFknYHBs",
        "https://m.youtube.com/watch?v=cMLTX2FClxw",
        "https://m.youtube.com/watch?v=bt8wNQJaKAk",
        "https://m.youtube.com/watch?v=pkoxFpmiCWo",
        "https://m.youtube.com/watch?v=kff_DXor7jc",
        "https://m.youtube.com/watch?v=p1qM75a9FeE",
        "https://m.youtube.com/watch?v=6lnS-8FVod4",
        "https://m.youtube.com/watch?v=O1bhZgkC4Gw",
        "https://m.youtube.com/watch?v=PZX2npwj6jY",
        "https://m.youtube.com/watch?v=l_HTf8JRTts",
        "https://m.youtube.com/watch?v=4BVd6TK8_EQ",
        "https://m.youtube.com/watch?v=H_2jH6XVYU0",
        "https://m.youtube.com/watch?v=CbH2F0kXgTY",
        "https://m.youtube.com/watch?v=I8fxhn_l08g",
        "https://m.youtube.com/watch?v=hN5MBlGv2Ac",
        "https://m.youtube.com/watch?v=_ciQX22n9NE",
        "https://m.youtube.com/watch?v=oLrp9uTa9gw",
        "https://m.youtube.com/watch?v=qo55wGLXcOQ",
        "https://www.youtube.com/watch?app=desktop&v=LbtQM793jn8",
        "https://m.youtube.com/watch?v=-H_k7iwrWVY"
    ]

    for url in video_urls:
        print(f"Processing video: {url}")
        process_video(url)

    print("すべてのデータ取得とGoogleドライブへの保存が完了しました。")