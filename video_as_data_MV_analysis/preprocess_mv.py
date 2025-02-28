import os
import json
import pandas as pd
from datetime import datetime

# ------------------------------------------------------
# 1. メタデータJSONを読み込み、DataFrameを作成する
# ------------------------------------------------------

# 例：Googleドライブ上のメタデータ保存先ディレクトリ
METADATA_DIR = 'your_directory' #例えば'/content/drive/MyDrive/video_as_data_MV_analysis'
def load_metadata_and_create_df(metadata_dir=METADATA_DIR):
    rows = []

    # Pretenderの公開日をヒット曲の境界と定義（例: 2019/4/16）
    pretender_date = datetime(2019, 4, 16)

    for file in os.listdir(metadata_dir):
        if file.startswith("metadata_") and file.endswith(".json"):
            json_path = os.path.join(metadata_dir, file)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 動画ID
            video_id = data.get('id', None)

            # タイトル
            title = data.get('title', "")

            # 再生回数
            view_count = data.get('view_count', 0)

            # コメント数
            comment_count = data.get('comment_count', 0)

            # 動画の長さ（秒）
            duration = data.get('duration', 0)

            # いいね数（取得できる場合）
            like_count = data.get('like_count', 0)

            # アップロード日（YouTubeでは upload_date が "YYYYMMDD" 形式で入る）
            upload_date_str = data.get('upload_date', "")
            release_date = None
            if upload_date_str and len(upload_date_str) == 8:
                try:
                    # 例: "20200115" → datetime(2020,1,15)
                    release_date = datetime.strptime(upload_date_str, "%Y%m%d")
                except:
                    release_date = None

            # indie_major カラム: 2018年以降なら1, それ以前なら0
            # （厳密には Official髭男dism のメジャーデビュー時期と合わせることを想定）
            if release_date and release_date.year >= 2018:
                indie_major = 1
            else:
                indie_major = 0

            # hitsong_after カラム:
            # Pretender公開日(2019/4/16)以降にリリースされた曲は1, それ以前は0
            # ただし、「Pretender」タイトルを含む場合も1にする
            hitsong_after = 0
            if release_date and release_date >= pretender_date:
                hitsong_after = 1
            if "pretender" in title.lower():  # タイトルに"pretender"が含まれる場合
                hitsong_after = 1

            # 本タスクで作りたいレコード
            row = {
                "id": video_id,
                "title": title,
                "view_count": view_count,
                "comment_count": comment_count,
                "duration": duration,
                "like_count": like_count,
                "release_date": release_date,
                "indie_major": indie_major,
                "hitsong_after": hitsong_after
            }

            rows.append(row)

    df = pd.DataFrame(rows)


    return df

# 実行例
df_mvs = load_metadata_and_create_df()
df_mvs.head(10)
