
# Easy Crash Course of Analyzing Video as Data  
**– Preliminary Application for Sociological Questions –**

この GitHub リポジトリは、2025年2月16～18日に筑波大学で開催された計算社会学会（CSSJ2025）でのポスター発表で用いたコードのチュートリアルです。

## 概要

- **開発環境**: Google Colab (T4 GPU / TPU v5e-1)  
- **可視化について**: 本リポジトリには可視化のコードは含めていません。必要に応じて matplotlib 等で行ってください。  
- **動画データについて**: YouTube Data API や `yt_dlp` ライブラリ等を用いて取得してください。音声は wav、映像は mp4 形式などで想定しています。

---

## 1. ネット討論番組の分析

### 対象動画
[YouTube: https://www.youtube.com/watch?v=Oikl21gsnLc](https://www.youtube.com/watch?v=Oikl21gsnLc)

### 分析の目的
1. **音声（auto_pipeline_SER.py）**  
   - 自動文字起こし & 話者識別 (pyannote)  
   - 音声感情認識 (SER) を適用して発言者ごとの感情分布を可視化  

2. **映像（auto_pipeline_face_emo.py）**  
   - 一定間隔でフレームを抽出  
   - 各フレームの顔を検出し、感情・年齢・性別を推定  
   - 顔埋め込みベクトルを取得 → HDBSCAN で出演者をクラスタリング → 出演者ごとの感情分布を可視化  

---

### 1-1) 音声解析: auto_pipeline_SER.py

#### 目的
- **話者ダイアリゼーション**: pyannote モデルで話者区間を分割  
- **自動文字起こし**: Whisper を用いて音声ファイルから書き起こし  
- **音声感情認識 (SER)**: 区間ごとに音声を切り出し、日本語対応モデルで感情スコアを取得  

#### スクリプト
- **auto_pipeline_SER.py**

#### 処理手順

1. **話者分割 & 文字起こし**  
   - pyannote を用いて話者ごとに区間分割  
   - Whisper を用いて各区間の音声を文字起こし  

2. **感情認識**  
   - torchaudio + Hugging Face の SER モデルを用いて区間ごとの感情スコアを取得  
   - 各セグメントごとに感情スコアの詳細 (JSON形式) を含む結果を CSV にまとめる  

3. **使い方**  
   1. `requirements_PTS.txt` を参照し、必要ライブラリをインストール  
   2. [pyannote-whisper](https://github.com/yinruiqing/pyannote-whisper.git) などの依存関係をクローンし、import 可能にする  
   3. `your_token` やモデル名などのパラメータを設定 (Gated モデルを利用する場合は認証要)  
   4. `audio.wav` を用意 → スクリプト実行 → 処理結果が CSV で出力される  

#### 成果物
- **diarize_text_emotion_result_full.csv** (今回の発表で用いた実例)

#### 問題点
- 精度向上のために利用モデルの検討や fine tuning が必要  
- `wav2vec2-xlsr-japanese-speech-emotion-recognition` を現状採用しているが、他モデルの可能性も要調査  

---

### 1-2) 映像解析: auto_pipeline_face_emo.py

#### 目的
- **フレーム単位で出演者の顔を検出し、感情・年齢・性別を推定**  
- 顔埋め込みを HDBSCAN でクラスタリングし、出演者ごとに感情分布を可視化  

#### スクリプト
- **frame_extract.py**, **auto_pipeline_face_emo.py**

#### 処理手順

1. **フレーム画像の読み込み**  
   - 指定ディレクトリにある画像ファイル (フレーム) を順次取得  

2. **DeepFace 分析**  
   - `DeepFace.analyze` で複数顔を検出・感情/年齢/性別の推定  
   - `DeepFace.represent` で顔埋め込みを取得  

3. **顔画像の切り抜き & Base64エンコード**  
   - 顔領域をクロップ → JPEG化 → Base64文字列として保持  

4. **HDBSCAN クラスタリング**  
   - 取得した顔埋め込みを基にクラスタを推定 (出演者識別)  

5. **CSV 出力**  
   - フレーム名、顔ID、感情/年齢/性別、埋め込み、Base64画像、クラスタ番号などをまとめて出力  

#### 使い方

1. `requirements_PTS.txt` を参照し、必要ライブラリをインストール  
2. `your_directory` やモデルのパラメータを設定 (Gatedモデルは認証要)  
3. 動画からフレームを抽出 (`frame_extract.py`) → 抽出したフレームに対して `auto_pipeline_face_emo.py` を実行  

#### 成果物

- **face_attributes_complete.csv**  
- `out` ディレクトリ配下にフレーム画像やクラスタリング結果が保存される  

#### 問題点

- モデル精度向上のための検討や fine tuning が必要  
- 低解像度の場合でも、HDBSCAN によりある程度分類は可能だが画質次第で精度が変動  

---

## 2. ポピュラー音楽のアーティストPV分析

### 対象動画
- **Official髭男dism** インディーズ期～2022年までの公式PV (YouTube で公開されているもの)

### 目的
- **映像表現を定量化**: PV をフレーム単位でベクトル化し、曲ごとの動画ベクトルとして可視化・分析したい  

### スクリプト
- **auto_pipeline_cultural_video_vector_1.py**  
  - PySceneDetect でシーンを検出してフレームを抽出  
- **auto_pipeline_cultural_video_vector_2.py**  
  - PyAV でフレームを抽出 (4K 動画等で PySceneDetect が動かない場合の代替)  

#### 補助スクリプト

- **mv_get.py** (YouTube ダウンロード: `yt_dlp` 利用)  
- **preprocess_mv.py** (YouTube動画のメタデータを取得・整形)

---

### 2-1) auto_pipeline_cultural_video_vector_1.py (PySceneDetect 版)

#### 処理手順

1. **シーン検出**  
   - `detect_scenes` + `ContentDetector` で切り替わり区間を取得  
   - `scene_threshold` で検出感度を調整  

2. **シーンごとのフレームサンプリング**  
   - 各シーンで `num_frames` 枚のフレームを等間隔に抽出 (ffmpeg で一時ファイル出力 → OpenCV+PIL 変換)  

3. **ViT モデルで特徴抽出**  
   - フレームを `vit_model` に入力 → CLS トークンを埋め込みベクトルとして取得  
   - シーン内フレームを平均して「シーン代表ベクトル」を作成  

4. **動画全体のベクトル化**  
   - 全シーン代表ベクトルをさらに平均 → 動画(PV)を表す 1 ベクトルに集約  
   - `df_mvs` に格納  

---

### 2-2) auto_pipeline_cultural_video_vector_2.py (PyAV 版)

#### 処理手順

1. **PyAV を使ったフレーム抽出**  
   - `sample_frames_pyav` で動画を開き、`skip_frames` ごとにフレームを取り出す  
   - `downscale` で縮小し、PIL イメージとして取得  

2. **フレーム数ベースでのサンプリング**  
   - 前述の PySceneDetect 版は「シーンごとの等間隔」または「秒数ベース」  
   - 今回は「N フレームごと」という基準でスキップし、可変フレームレートにも対応しやすい  

3. **ViT 埋め込み → 全フレーム平均**  
   - 取得フレームを ViT で埋め込み → 平均ベクトルを算出  
   - 結果を `df_mvs` の `pv_vector` 列に格納  

---

### 使い方

1. `requirements_MV_vector.txt` を参考に必要ライブラリをインストール  
2. `your_directory` / `your_token` / モデル名などのパラメータを設定  
3. 動画ファイルごとにスクリプトを走らせる → データフレーム (`df_features` や `df_mvs`) にベクトルが追加される  

#### 成果物
- **df_pv.csv**  
  - 今回の発表で用いた PV 一覧とベクトルなどを記録した CSV

---

## 最後に

- **本リポジトリ**は、上記のような **音声＋映像** の分析処理例を示すコード集です。  
- すべてのコードは Google Colab などで動作確認済みですが、環境やバージョン差異により挙動が変わる場合があります。  
- より高い精度・効率を求める場合は、モデルの変更・fine tuning・前処理の改善などを検討してください。  

作成者：https://www.linkedin.com/in/shimmei-yamauchi-0a37ba30b/

何か質問がある場合は、以下のアドレスにご連絡ください。saepo12100426(at)gmail.com


