# scripts/archives — 引退スクリプト置き場

役目を終えた一回限りのスクリプトを `git mv` でここへ退避する。**削除ではなくアーカイブ**
する方針 (調査の経緯・再現性を残すため。履歴は git で追える)。

## 退避の基準

以下に該当し、かつ**現役スクリプトから import されていない**ものを移す:

- **probe / 調査スクリプト** (`dollma_probe*` / `dollma_d1_*` / `dollma_find_*`): 計測・調査が
  完了し結論が docs に反映済みのもの。
- **引退した実験パイプライン**: 研究結論が出て不採用 or 後継に置換されたもの。
  - `dollma_b_seedsweep*` (施策B 500版 → b2000/b10k に置換)
  - `dollma_c_seedsweep*` (施策C 完了)
  - `dollma_distill_pairs.py` (蒸留 D2/D4 不採用)
  - `dollma_d6_teacher_cache.py` + `test_dollma_d6_teacher_cache.py` (外部教師蒸留 D6 不採用)

## 注意

- ここへ移したスクリプトは**再実行を想定しない**。再び動かす場合、現役モジュール
  (`dollma_make_pairs` 等) を import するものは `sys.path` 調整が要る (退避で相対位置が変わるため)。
- **現役から import される基盤モジュールは移さない**: `dollma_make_pairs` /
  `dollma_make_eval_diverse` / `dollma_make_diverse_train` / `train_bitnet` など。
- テストランナー (meson) は C++ 専用で、ここの Python は参照しない。
