# Floodgate CSA から学習データ作成・学習・floodgate対局までの手順

この手順は、このリポジトリ (`DeepLearningShogi`) の既存スクリプトを前提にしています。

- CSA変換: `dlshogi/utils/csa_to_hcpe.py`, `dlshogi/utils/csa_to_hcpe3.py`
- 学習: `dlshogi/train.py`
- モデル変換: `dlshogi/convert_model_to_onnx.py`
- 対局エンジン: `usi` (USIエンジン)

## 0. 前提（学習: WSL2 / 運用: Windows）

- Docker / Docker Compose
- NVIDIA GPUを使う場合は NVIDIA Container Toolkit
- floodgateクライアントは別途必要（このリポジトリには含まれない）
- 学習・前処理は **WSL2 上**で実行
- 将棋所経由のfloodgate運用は **Windows 上**で実行

## 1. WSL2上でDocker環境を構築

最小の開発用 Dockerfile 例（`Dockerfile`）:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir uv
WORKDIR /work/DeepLearningShogi
```

ビルド:

```bash
docker build -t dlshogi-dev:cuda12.1 .
```

起動（GPUあり）:

```bash
docker run --rm -it --gpus all \
  -v /path/to/DeepLearningShogi:/work/DeepLearningShogi \
  -v /path/to/data:/data \
  -w /work/DeepLearningShogi \
  dlshogi-dev:cuda12.1 bash
```

起動（CPUのみ）:

```bash
docker run --rm -it \
  -v /path/to/DeepLearningShogi:/work/DeepLearningShogi \
  -v /path/to/data:/data \
  -w /work/DeepLearningShogi \
  dlshogi-dev:cuda12.1 bash
```

## 2. uv仮想環境を作成（インストール前チェック付き）

コンテナ内で実行:

```bash
uv venv .venv
source .venv/bin/activate
```

まず、主要ライブラリが既に入っているか確認:

```bash
uv run python - <<'PY'
mods = ["numpy", "scipy", "pandas", "matplotlib", "cshogi", "torch"]
for m in mods:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception:
        print(f"[MISSING] {m}")
PY
```

`[MISSING]` のみインストール:

```bash
uv pip install -U pip setuptools wheel
uv pip install numpy scipy pandas matplotlib
uv pip install "git+https://github.com/akio-kobayashi/cshogi.git"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

再チェック:

```bash
uv run python - <<'PY'
mods = ["numpy", "scipy", "pandas", "matplotlib", "cshogi", "torch"]
for m in mods:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[NG] {m}: {e}")
PY
```

ディレクトリ例:

```bash
export DLSHOGI_ROOT=/path/to/DeepLearningShogi
export DATA_ROOT=/path/to/data
mkdir -p $DATA_ROOT/{csa_raw,csa_train,csa_val,hcpe,models,onnx,logs}
cd $DLSHOGI_ROOT
```

## 3. ダウンロード済み CSA を配置

この手順では **CSAファイルは事前にダウンロード済み** を前提にします。  
ダウンロード済みCSAを `csa_raw` にコピーしてください。

```bash
cp -r /path/to/downloaded_csa_dir/* $DATA_ROOT/csa_raw/
```

学習/検証を分割します（例: 9:1）。

```bash
mkdir -p $DATA_ROOT/csa_train $DATA_ROOT/csa_val
find $DATA_ROOT/csa_raw -type f -name "*.csa" | shuf > $DATA_ROOT/csa_all.txt
total=$(wc -l < $DATA_ROOT/csa_all.txt)
train_n=$(( total * 9 / 10 ))
head -n $train_n $DATA_ROOT/csa_all.txt > $DATA_ROOT/csa_train.txt
tail -n +$((train_n + 1)) $DATA_ROOT/csa_all.txt > $DATA_ROOT/csa_val.txt

# 元ディレクトリ構造を保ってコピー
while read -r f; do
  rel="${f#$DATA_ROOT/csa_raw/}"
  mkdir -p "$DATA_ROOT/csa_train/$(dirname "$rel")"
  cp "$f" "$DATA_ROOT/csa_train/$rel"
done < $DATA_ROOT/csa_train.txt

while read -r f; do
  rel="${f#$DATA_ROOT/csa_raw/}"
  mkdir -p "$DATA_ROOT/csa_val/$(dirname "$rel")"
  cp "$f" "$DATA_ROOT/csa_val/$rel"
done < $DATA_ROOT/csa_val.txt
```

品質確認（手数・千日手比率など）:

```bash
find $DATA_ROOT/csa_train -type f -name "*.csa" | wc -l
find $DATA_ROOT/csa_val -type f -name "*.csa" | wc -l
uv run python dlshogi/utils/stat_csa_in_dir.py $DATA_ROOT/csa_train --rating 3800
uv run python dlshogi/utils/stat_csa_in_dir.py $DATA_ROOT/csa_val --rating 3800
```

## 4. CSA -> 学習データへ変換

`train.py` の引数仕様上、以下の組み合わせが実用的です。

- 学習データ: `hcpe3` (`csa_to_hcpe3.py`)
- 検証データ: `hcpe` (`csa_to_hcpe.py`)

### 4.1 学習データ（hcpe3）

```bash
uv run python dlshogi/utils/csa_to_hcpe3.py \
  $DATA_ROOT/csa_train \
  $DATA_ROOT/hcpe/train_floodgate.hcpe3 \
  --filter_moves 50 \
  --filter_rating 3800 \
  --uniq
```

オプション補足:

- `--skip_opening`: 評価値コメントなし序盤を除外
- `--out_maxmove`, `--out_mate` などは用途に応じて追加

### 4.2 検証データ（hcpe）

```bash
uv run python dlshogi/utils/csa_to_hcpe.py \
  $DATA_ROOT/csa_val \
  $DATA_ROOT/hcpe/val_floodgate.hcpe \
  --filter_moves 50 \
  --filter_rating 3500 \
  --uniq
```

## 5. モデル学習

`dlshogi/train.py` で学習します。

```bash
uv run python -m dlshogi.train \
  $DATA_ROOT/hcpe/train_floodgate.hcpe3 \
  $DATA_ROOT/hcpe/val_floodgate.hcpe \
  --network resnet10_swish \
  --batchsize 1024 \
  --testbatchsize 1024 \
  --epoch 10 \
  --gpu 0 \
  --lr 0.001 \
  --eval_interval 1000 \
  --checkpoint "$DATA_ROOT/models/checkpoint-{epoch:03}.pth" \
  --model "$DATA_ROOT/models/model-{epoch:03}.npz" \
  --log "$DATA_ROOT/logs/train.log"
```

補足:

- `--model` を指定すると、最終的に `npz` が出力されます。
- 再開時は `--resume <checkpoint.pth>` を使用。

## 6. 学習モデル(npz)をONNXへ変換

USIエンジン側の既定 `DNN_Model` は `model.onnx` です。

```bash
uv run python -m dlshogi.convert_model_to_onnx \
  $DATA_ROOT/models/model-010.npz \
  $DATA_ROOT/onnx/model-010.onnx \
  --network resnet10_swish \
  --gpu 0
```

必要に応じて:

- `--fuse`
- `--fixed_batchsize 1`

## 7. USIエンジンをビルドしてモデルを設定

```bash
cd $DLSHOGI_ROOT/usi
make -j
```

起動先GUI/対局マネージャで以下を設定:

- `DNN_Model` = `$DATA_ROOT/onnx/model-010.onnx`
- （必要なら）`Book_File`, `USI_Ponder` など

### 7.1 配布バイナリを使う（推奨）

Visual Studio での Windows ネイティブビルドを避けたい場合は、公式 Releases の配布バイナリを利用します。

- DeepLearningShogi Releases: `https://github.com/TadaoYamaoka/DeepLearningShogi/releases`

手順:

1. Releases から Windows 向けビルド済みアーカイブ（`usi` / `usi_onnxruntime` を含むもの）を取得して展開
2. 将棋所にエンジン実行ファイル（`.exe`）を登録
3. 将棋所の USI オプションで `DNN_Model` を学習済みONNXへ設定
4. 必要に応じて `Book_File` や `USI_Ponder` を設定

補足:

- 配布バイナリ利用時も、モデル学習・ONNX変換は本手順（Docker + uv）で進められます。
- エンジン本体だけを配布バイナリに差し替える運用が最も簡単です。

### 7.2 Windows + 将棋所 + OnnxRuntime の具体手順（運用）

1. Releases から `usi_onnxruntime` を含むアーカイブを展開する  
2. 1つのフォルダに以下を集約する  
   - OnnxRuntime版エンジン実行ファイル（配布物の実ファイル名）
   - 同梱DLL（配布物に含まれる `.dll` 一式）
   - 学習済みONNX（例: `model-010.onnx`）
   - （任意）定跡ファイル `book.bin`
3. 展開先で実行ファイル名を確認する（PowerShell）  
   - `Get-ChildItem -Filter *.exe | Select-Object Name`
4. 将棋所で「エンジン設定」から上記 `.exe` を登録する  
5. 将棋所のエンジンオプションで最低限以下を設定する  
   - `DNN_Model` = `C:\\path\\to\\model-010.onnx`  
   - `Book_File` = `C:\\path\\to\\book.bin`（使う場合）  
   - `USI_Ponder` = `false`（最初は安定優先）
6. 登録後、将棋所の「エンジン情報表示」でUSI応答を確認する  
   - `usi` 応答が返る  
   - `isready` に `readyok` が返る  
7. 将棋所で1局ローカル対局し、以下を確認する  
   - 起動直後に落ちない  
   - 思考開始でONNX読込エラーが出ない  
   - 指し手が返る

トラブル時の確認:

- DLL不足: `.exe` と同じフォルダに配布同梱DLLが揃っているか確認  
- パス誤り: `DNN_Model` は絶対パスで設定（`\` 区切り）  
- モデル不整合: 学習ネットワーク構成とONNX変換時 `--network` が一致しているか確認  
- GPU未使用でよい場合: まずCPU/既定設定で起動確認してから最適化設定を足す

## 8. ローカルで事前検証（推奨）

floodgate投入前に、最低限次を確認:

- エンジン起動時に `DNN_Model` 読み込みエラーがない
- 数局の自己対局/ローカル対局で即落ちしない
- 思考時間制約下で応答遅延が致命的でない

## 9. floodgateへ投入（Windows側）

このリポジトリには floodgate 接続クライアント本体は含まれません。
別途、運用中の floodgate クライアント（例: shogi-server系クライアント）で以下を設定します。

- エンジン実行ファイル（Windows）: `C:\path\to\usi_onnxruntime.exe`（配布バイナリ）
- USIオプション: `DNN_Model`, `Book_File`, `USI_Ponder` など
- floodgateアカウント情報（ID/パスワード）
- 接続先ホスト/ポート

運用フロー:

1. クライアント起動
2. サーバ接続
3. 待機/対局開始
4. ログ監視（接続断・クラッシュ・時間切れ）

## 10. 更新サイクル（実運用）

1. 新しいfloodgate CSAを定期取得
2. `csa_to_hcpe3.py` / `csa_to_hcpe.py` で再生成
3. 学習を再実行
4. ONNX再変換
5. `DNN_Model` 差し替え
6. floodgate再投入

---

## トラブルシュート

- `csa_to_hcpe3.py` で棋譜がほとんど落ちる
  - `--filter_rating`, `--filter_moves` が厳しすぎないか確認
  - 評価値コメントがないCSAは除外される点に注意

- `train.py` がデータ形式で落ちる
  - 学習側が `hcpe3`、検証側が `hcpe` になっているか確認

- USI起動時にモデル読込失敗
  - ONNXのパス/権限/ネットワーク構造(`--network`)の整合を確認

- floodgateで不安定
  - まずローカル長時間対局で安定性を確認してから投入
