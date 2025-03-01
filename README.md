<img width="1027" alt="image" src="https://github.com/user-attachments/assets/455f241f-d16b-4f9d-ae71-62918215f924" /># Easy Crash Course of Analyzing Video as Data ----Preliminary Application for sociological questions

このgithubは2月16-18日の日程行われた計算社会学会@筑波大学（CSSJ2025）で私が行ったポスター発表で用いた分析のコードのチュートリアルです。

事務局に提出した論文では大きく以下の通り分析を行いました。
・環境：GoogleColabで実装（ランタイム-T4GPUおよびv5e-1TPU）。
・可視化のコードは挙げていませんので適宜matplotlibなどで行なってください。
・動画についてはYoutubeDataAPIやyt_dlpライブラリなどを用いて適宜取得してください（音声→wav形式、映像→mp4形式）。

①　Youtubeのネット討論番組について（対象動画：https://www.youtube.com/watch?v=Oikl21gsnLc）
1）音声（audio）

目的：

自動文字起こしと話者識別を兼ねたパイプラインに音声感情認識（SER）をかけ発言（者）のチャンクごとに識別しその分布を可視化する。

スクリプト：

auto_pipeline_SER.py

概要 :

このスクリプトは、音声ファイルから以下の処理を自動実施します。

処理の手順：

（文字起こし＆話者ダイアリゼーション）pyannote のモデルを用いて話者の切り分け（ダイアリゼーション）を実施しその区間ごとに音声を切り出し、Whisper を用いて音声ファイルから文字起こしを実施。

（音声感情認識）上の文字起こし・話者識別後のデータを受け取り、torchaudioで対象区間ごとに音声全体を読み込み、感情認識を実施（Hugging Face の日本語対応モデルを利用）し、各セグメントの感情スコアの詳細（JSON形式）を含む結果を取得する。

感情認識結果の CSV 出力：

感情認識結果も10分ごとにチャンクとして追記保存し、全体の結果もまとめて CSV ファイルに出力（処理がセッションの関係で途中で止まってしまったときのよう）

使い方：

1)requirements_PTS.txtをもとにpipで、必要なライブラリをインストールする。また（!git clone ）https://github.com/yinruiqing/pyannote-whisper.gitこのgithubのpyannote_whisper.utils関数を用いているので適当なdirectoryに配置し呼び出せるようにする。

2)your_directory や your_token、使用するモデル名などのパラメータの設定する。いくつかのモデルについてはhuggingfaceでのGatedモデルなので認証を通しておく。
→hg_auth_confirm.pyはGatedモデルの認証が通ってるか確認するスクリプト

3)対象の音声ファイル（audio.wav）を所定のディレクトリに配置し、スクリプトを実行すると、処理結果が指定ディレクトリ内に CSV ファイルとして保存される。

成果物：
参考までに今回の発表で用いたoutputについても置いておく（対象動画：https://www.youtube.com/watch?v=Oikl21gsnLc）。
→diarize_text_emotion_result_full.csv

問題点：

・精度がイマイチでないので用いているモデルの検討やfine tuningを考えたい。wav2vec2-xlsr-japanese-speech-emotion-recognitionを現状書いてますが、他によりモデルがあったら良い。

2） 映像（visual）

目的：

ある程度決まった感覚でframe（フレーム）を抽出し、そこにいる、出演者の顔から感情を推定し（今回はgenderとageも推定）、その後顔ごとにクラスタリングして出演者を識別し、出演者ごとの感情の分布を可視化。

スクリプト：

frame_extract.py、auto_pipeline_face_emo.py

概要 :

このスクリプトは、音声ファイルから以下の処理を自動実施します。

処理の手順：

①フレーム画像の読み込み
指定ディレクトリ内の画像ファイル（*.jpg, *.pngなど）を順次取得します。

②DeepFaceで複数顔を解析
DeepFace.analyze を利用し、フレーム内のすべての顔を検出。
顔ごとに感情、年齢、性別などを推定し、さらに DeepFace.represent で顔の埋め込み（数値ベクトル）を取得します。
顔領域の切り抜き・Base64エンコード

③フレーム画像から顔領域をクロップ
切り抜いた画像をJPEGにエンコードし、Base64文字列として保持します。

④HDBSCANでクラスタリング
取得した顔埋め込みをHDBSCANに入力し、類似度に応じてクラスタを割り振ります。

⑤結果のCSV出力
フレーム名、顔ID、感情・年齢・性別、埋め込み、Base64画像、クラスタ番号などをDataFrameにまとめてCSVに書き出します。

使い方：

1)requirements_PTS.txtをもとにpipで、必要なライブラリをインストールする

2)your_directory や your_token、使用するモデル名などのパラメータの設定する。いくつかのモデルについてはhuggingfaceでのGatedモデルなので認証を通しておく。
→hg_auth_confirm.pyはGatedモデルの認証が通ってるか確認するスクリプト

3)まず対象動画について、frame_extract.pyにかけフレームを作成し、そのフレームに対して、auto_pipeline_face_emo.pyを実行する。

成果物：
参考までに今回の発表で用いたoutputについても置いておく（対象動画：https://www.youtube.com/watch?v=Oikl21gsnLc）。
→face_attributes_complete.csv　また、outディレクトリ傘下に抽出したframeの一部とクラスタリングした顔の画像を収録させていたので参考にしていただければ。

問題点：

・こちらも精度がイマイチでないので用いているモデルの検討やfine tuningを考えたい。ただ顔の識別はHDBSCANを用いだたが画像サイズが小さい場合は識別が難しいがそれでもかなりの精度で分類できていた（clustered_images傘下の画像を参考にいただければ）


①　ポピュラー音楽のアーティストのPVについて（対象動画：Official髭男dismのインディーズ時代から2022年までの公式で公開されているPV）

目的：

映像表現を定量化して扱いたく、PVからframeを抽出し、各フレームに対してvisiob transformerにかけベクトル化した後、全てのフレームの重心を作成し動画のベクトルとした。

スクリプト：

auto_pipeline_cultural_video_vector_1.py, auto_pipeline_cultural_video_vector_2.py

*_1はPysceneDetectでframeを作成しています。_2はPyAVで作成しています。これは一部の動画がPysceneDetectだとシーン切り出しができない事象があったためです。特におそらく4K仕様の動画だと切り出しが私の環境だとできませんでした。

cf：
mv_get.py（参考までにMVを取得したyt_dlpライブラリを用いたコード）
preprocess_mv.py（youtube動画のメタデータ取得して結合処理を行う前処理コード）

概要 :

このスクリプトは、音声ファイルから以下の処理を自動実施します。

処理の手順（auto_pipeline_cultural_video_vector_1.py）：

①シーン検出（Scene Detect）
detect_scenes 関数を用いて、pyscenedetect の ContentDetector で動画内のシーン切り替わり（開始秒・終了秒）を検出
scene_threshold パラメータを変更することで検出感度を調整

②シーンごとのフレームサンプリング
検出した各シーン区間 [start_sec, end_sec] の間を、num_frames 枚だけ等間隔にフレームを抽出
ffmpeg をサブプロセスとして呼び出し、一時ファイルにフレームを書き出してから OpenCV + PIL で画像オブジェクトを作成

③ViT モデルによる特徴抽出
サンプリングした複数フレームを Vision Transformer (vit_model) に入力し、CLS トークンの埋め込みベクトルを取得
同じシーン内で得られたフレーム埋め込みを平均し「シーンの代表ベクトル」を計算

④動画(PV)全体のベクトル化
すべてのシーンで求めた「シーン代表ベクトル」をさらに平均し、PV 全体を表す 1 本のベクトルとして出力
DataFrameへの格納

⑤動画ファイルごとに抽出したベクトルをまとめ、df_mvs に対応する形で列 (pv_vector) として格納
最終的に、各 id ごとの埋め込みを含む df_features を得る

処理の手順（auto_pipeline_cultural_video_vector_2.py）：

①PyAV を使ったフレーム抽出 (sample_frames_pyav)
PyAV (FFmpeg バインディング) で動画を読み込み
skip_frames=30 なら「30フレーム飛ばし」で 1 枚フレームを取り出す
取得したフレームを downscale 倍率で縮小 → PIL 画像のリストとして返す

②フレーム数ベースでのサンプリング
前の例は「step_sec 秒ごと」という 時間ベースでフレームを取得していた
今回は フレーム数ベース (frame_count % skip_frames == 0) で抽出している
可変フレームレートの動画でも比較的単純に“数”で間引けるメリットがある

③その他の流れは基本的に_1.pyと同じ
取得したフレームを ViT で埋め込みベクトル化
全フレームの CLS トークンを平均して、1 つの動画を表すベクトル (pv_vector) を生成
結果を df_mvs の pv_vector 列に格納

使い方：

1)requirements_MV_vector.txtをもとにpipで、必要なライブラリをインストールする

2)your_directory や your_token、使用するモデル名などのパラメータの設定し、対象directoryに対してコードを走らせる。

成果物：
参考までに今回の発表で用いたoutputについても置いておく（対象動画：Official髭男dismのインディーズ時代から2022年までの公式で公開されているPV）
→df_pv.csv



