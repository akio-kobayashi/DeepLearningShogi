# DeepLearningShogi(dlshogi)
[![pypi](https://img.shields.io/pypi/v/dlshogi.svg)](https://pypi.python.org/pypi/dlshogi)

将棋でディープラーニングの実験をするためのプロジェクトです。

基本的にAlphaGo/AlphaZeroの手法を参考に実装していく方針です。

検討経緯、実験結果などは、随時こちらのブログに掲載していきます。

http://tadaoyamaoka.hatenablog.com/

## ダウンロード
[Releases](https://github.com/TadaoYamaoka/DeepLearningShogi/releases)からダウンロードできます。

最新のモデルファイルは、[棋神アナリティクス](https://kishin-analytics.heroz.jp/lp/)でご利用いただけます。

## ソース構成
|フォルダ|説明|
|:---|:---|
|cppshogi|Aperyを流用した将棋ライブラリ（盤面管理、指し手生成）、入力特徴量作成|
|dlshogi|ニューラルネットワークの学習（Python）|
|dlshogi/utils|ツール類|
|selfplay|MCTSによる自己対局|
|test|テストコード|
|usi|対局用USIエンジン|
|usi_onnxruntime|OnnxRuntime版ビルド用プロジェクト|

## ビルド環境
### USIエンジン、自己対局プログラム
#### Windowsの場合
* Windows 11 64bit
* Visual Studio 2022
#### Linuxの場合
* Ubuntu 18.04 LTS / 20.04 LTS
* g++
#### Windows、Linux共通
* CUDA 12.1
* cuDNN 8.9
* TensorRT 8.6

※CUDA 10.0以上であれば変更可

### 学習部
上記USIエンジンのビルド環境に加えて以下が必要
* [Pytorch](https://pytorch.org/) 1.6以上
* Python 3.7以上 ([Anaconda](https://www.continuum.io/downloads))
* CUDA (PyTorchに対応したバージョン)
* cuDNN (CUDAに対応したバージョン)

## 謝辞
* 将棋の局面管理、合法手生成に、[Apery](https://github.com/HiraokaTakuya/apery)のソースコードを使用しています。
* モンテカルロ木探索の実装は囲碁プログラムの[Ray+Rn](https://github.com/zakki/Ray)の実装を参考にしています。
* 探索部の一部にLeela Chess Zeroのソースコードを流用しています。
* 王手生成などに、[やねうら王](https://github.com/yaneurao/YaneuraOu)のソースコードを流用しています。

## 学習設定（train.py）
`dlshogi/train.py` は `dlshogi/config.yaml` を既定設定として読み込み、コマンドライン引数で上書きできます。

例:
```bash
python -m dlshogi.train --config dlshogi/config.yaml --lr 0.0005 --epoch 10
```

優先順位:
1. CLI引数
2. `--config` で指定した YAML
3. `train.py` 内のデフォルト値

また学習開始時に、実行ディレクトリへ `hparams.yaml` を保存します。
実行ディレクトリは以下の順で決まります。
1. `--checkpoint` の出力先ディレクトリ
2. `--model` の出力先ディレクトリ
3. カレントディレクトリ

### 並列導入ネットワーク（resnetx）
既存のネットワーク実装を維持したまま、実験用の新実装を並列導入しています。
`--network` で以下のように選択可能です。

```bash
# 新実装（並列導入）
--network resnetx10
--network resnetx20x256_fcl384_swish

# 既存実装（従来どおり）
--network resnet10_swish
--network wideresnet10
```

### 自己対局の実験オプション（Gumbel root）
`selfplay/self_play.cpp` には、既存の `hcpe3` 学習パイプラインを保ったまま
ルート探索を Gumbel 方式で試す実験オプションがあります。
内部ノードの探索は従来の PUCT のままで、ルートでは Gumbel ノイズ付きスコアで候補を並べ、
round ごとに候補を半減させながら探索予算を配る実装です。

```bash
--gumbel_root
--gumbel_scale 1.0
--gumbel_prior_weight 1.0
--gumbel_value_weight 1.0
--gumbel_visit_weight 0.25
--gumbel_export_temperature 1.0
```

`--gumbel_root` を指定しなければ、従来の自己対局挙動は変わりません。

CUDA/TensorRT 環境で短時間の動作確認をしたい場合は、
[`selfplay/run_gumbel_smoke_test.sh`](/Users/akio/Documents/GitHub/DeepLearningShogi/selfplay/run_gumbel_smoke_test.sh)
で通常 root と Gumbel root を小さい設定で流せます。

```bash
./selfplay/run_gumbel_smoke_test.sh \
  --model /path/to/model \
  --roots /path/to/roots.hcp \
  --out-dir /tmp/dlshogi-gumbel-smoke
```

`selfplay/bin/selfplay` ができていることが前提です。必要なら `--selfplay`、`--nodes`、`--playouts`、`--gpu-id` などで上書きできます。

## ライセンス
ライセンスはGPL3ライセンスとします。
