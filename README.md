# pusht-compass

学習済みの PushT に、ゴールを基準にした座標系（コンパス）を渡し、学習し直さずに別のゴールへ T を運ばせる。
方策は [lerobot/diffusion_pusht_keypoints](https://huggingface.co/lerobot/diffusion_pusht_keypoints) で、推論だけに使う。

## 準備と実行

```sh
uv sync
uv run python scripts/check_baseline.py --device mps   # 読み込みの確認（公開値 71% と比べる）
uv run python scripts/predict_grid.py                  # 格子の予想（方策は動かさない）
uv run python scripts/prediction_intervals.py          # 予想の範囲
uv run python scripts/run_conditions.py --device mps   # 条件の表
uv run python scripts/run_grid.py --device mps         # 格子（約 20 分）
uv run python scripts/judge.py                         # 予想との照合
```

`--device cpu` でも動く。
seed と乱数の種は固定している。

## 中身

| 場所 | 中身 |
| --- | --- |
| `src/compass/policy_loader.py` | 2024 年形式の重みを LeRobot 0.6.1 に読み込み、重みの中の統計で正規化する |
| `src/compass/se2.py`、`agent.py` | コンパス（ゴールを学習時の位置に重ねる剛体変換）と、それを挟んで方策を動かすエージェント |
| `src/compass/rollout.py` | seed を束ねて同時に回す。ゴールの移動、T の移動、観測の雑音を扱う |
| `src/compass/support.py`、`geometry.py` | 初期配置が学習データの範囲に収まるか、ゴールの T が壁の内側に収まるか |
| `scripts/` | 実験、予想、判定 |

## ライセンスと出典

コードは Apache-2.0。  
LeRobot、gym-pusht、学習済みの方策も Apache-2.0。  
動画の蜂の場面は、作者の別の自作シミュレーション（非公開）から書き出した。  
コードと文章の下書きには AI のコーディング支援を使い、問いの設定、予想の確認、採否は人が判断した。
