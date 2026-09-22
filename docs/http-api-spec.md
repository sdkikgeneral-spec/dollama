# HTTP API 仕様 — dollama サーバー

## 概要

OpenAI Images API 互換 HTTP サーバー。**配管は自作しない** — HTTP は
**cpp-httplib** (単一ヘッダ・Winsock2/POSIX を内部吸収)、JSON は **nlohmann/json**
(ヘッダオンリー) を使う。重量級フレームワークは不使用・単一バイナリの方針は
ヘッダオンリー採用で維持。自作は HW 研究コアに限定 (CLAUDE.md「実装方針」参照)。
エンドポイント実装は `src/server/api.cpp`。

## エンドポイント

### POST /v1/images/generations

txt2img 生成。

**リクエスト**:

```json
{
  "prompt": "1girl, silver hair, magical girl",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json"
}
```

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `prompt` | string | 必須 | 日本語 or 英語 (LLM が danbooru タグに変換) |
| `n` | int | 1 | 生成枚数 (現在 1 のみ対応) |
| `size` | string | `"1024x1024"` | `"1024x1024"` 固定 |
| `negative_prompt` | string | `""` | (拡張フィールド、OpenAI 非標準) |
| `steps` | int | 20 | (拡張フィールド) |
| `guidance_scale` | number | backend 既定 (SDXL は **7.5**) | (拡張フィールド・E-2) CFG スケール。省略時は従来どおり `cfg=0.0f` を backend へ渡し、backend 側の既定にフォールバックする (`sdxl_backend.hpp` は `cfg<=0` のとき `kGuidanceScale=7.5f`)。★**0 以下を明示指定した場合も同契約で既定 7.5 になる** (0 指定で CFG を無効化することはできない)。数値以外は 400。★**効くのは段1 (`BackendImageGenerator`) 経路のみ** — 生成器は 3 段 DI (`src/server/cli_generate.hpp` の 3 段フォールバック) で、段2 `PipelineGenerator` / 段3 `StubGenerator` へ落ちた場合、これらには CFG の概念自体が無く (`req.guidance_scale` の参照が 0 件) **400 にもならず黙って無視される**。段1 かどうかの判定は `[gen] seed=` 行の有無で行う (段2/3 はこの行を出さない) |
| `seed` | 非負整数 (uint64) | env `DOLLAMA_SEED` → 時刻ベース | (拡張フィールド・E-2) 乱数シード。解決順は **`seed` > env `DOLLAMA_SEED` > 時刻ベース**の 3 段 (`backend_image_generator.hpp`)。実効値は `[gen] seed=<値>(req|env|time)` としてログに出る。★**この行の出力先は stderr** (`backend_image_generator.hpp` の `std::clog`)。一方、段2/3 が名乗る行は stdout (`src/server/cli_generate.hpp` の `log` ストリーム = 呼び出し元 `src/main.cpp` が `std::cout` を渡す) — **stdout だけを grep すると段1 でも `[gen] seed=` が 0 件になり段2/3 と誤判定する。取得時は必ず `2>&1` すること** (本注記は seed 行にのみ置く。`guidance_scale` 行の同判定も同じ)。非負整数以外 (負数・小数・文字列) は 400。★**効くのは段1 (`BackendImageGenerator`) 経路のみ** — 段2 `PipelineGenerator` は `req.seed` を一切読まず内部で seed を決め (`src/server/pipeline_generator.hpp` の seed 決定箇所。「GenRequest に seed フィールドが無いため内部で決める」という **stale コメントのまま**)、段3 `StubGenerator` は **seed の概念自体を持たない** (`src/server/stub_generator.hpp` に seed/RNG が 0 件・出力は `fnv1a(prompt) ^ fnv1a(negative_prompt)*16777619` 由来の決定的な base 色に x/y 方向のグラデーションを載せたダミー画像)。いずれも `req.seed` は**黙って無視**される。判定は `[gen] seed=` 行の有無で行う (段2/3 はこの行を出さない) |
| `preset_prefix` | bool | `true` | (拡張フィールド・2-6e) preset 付帯の prompt_prefix/negative_prefix を自動付与するか。`false` で OFF (CLI `--no-preset-prefix` 相当)。真偽値以外は 400 |
| `loras` | array of `{name: string, strength: number}` | `[]` (未指定 = 空 = 従来経路) | (拡張フィールド・L-2) ランタイム LoRA。`strength` 省略時 1.0。`name` は `DOLLAMA_LORA_DIR` (既定 `models/loras`) 配下の `<name>.safetensors` へ解決され、**許可文字 `[A-Za-z0-9_.-]`・空/先頭ドット禁止**で path traversal を封じる (`sdxl_backend.hpp` の `resolve_lora_path`)。配列でない / 要素が `{"name": 非空文字列}` でない / `strength` が数値でない場合は 400。不正名・重み未解決は生成時に `std::invalid_argument` → 400 |
| `response_format` | string | `"b64_json"` | `"b64_json"` or `"url"` (url は未対応) |

**レスポンス (200 OK)**:

```json
{
  "created": 1700000000,
  "data": [
    {
      "b64_json": "<base64 encoded PNG>"
    }
  ]
}
```

**エラーレスポンス**:

```json
{
  "error": {
    "message": "エラーの説明",
    "type": "invalid_request_error",
    "code": null
  }
}
```

---

### POST /v1/images/edits

img2img 生成 (入力画像を latent encode して編集)。

**リクエスト** (multipart/form-data):

| フィールド | 型 | 説明 |
|---|---|---|
| `image` | file (PNG) | 入力画像 (1024×1024 PNG) |
| `prompt` | string | 編集プロンプト |
| `n` | int | 生成枚数 |
| `strength` | float (0-1) | ノイズ量 (1.0 = txt2img 相当) |

**レスポンス**: `/v1/images/generations` と同形式。

---

### GET /health

サーバー死活確認。

**レスポンス (200 OK)**:

```json
{ "status": "ok" }
```

---

### GET /v1/models

利用可能モデル一覧 (OpenAI 互換)。`data` は常に 1 要素で、`id` は生成器の `model_id()` をそのまま返す
(`src/server/api.cpp` の `/v1/models` ハンドラ)。

**レスポンス (200 OK)**:

```json
{
  "data": [
    { "id": "sdxl-1.0/illustrious-xl", "object": "model" }
  ]
}
```

**`id` の形 (2-6f)**: `SDXLBackend` 経路では `compose_model_id(preset)`
(`src/server/sdxl_backend.hpp`) が `preset` の有無で切り替える。

| 条件 | `id` |
|---|---|
| preset が解決できた (既定出荷 = `illustrious-xl`) | `sdxl-1.0/<preset>` → 既定は **`sdxl-1.0/illustrious-xl`** |
| `--preset base` / `DOLLAMA_BACKEND_PRESET=base` の明示 | `sdxl-1.0` (無印) |
| preset 名が `models/presets/` に見つからず base へフォールバック | `sdxl-1.0` (無印・`cli_generate.hpp` が `cfg.preset` を空にするため) |
| フォールバック生成器 (`pipeline_generator.hpp` / `stub_generator.hpp` / `txt2img_generator.hpp`) | `sdxl-1.0` (無印・2-6f では追随させていない) |

UI (2-6f) はこの `id` をエンドポイント選択肢のラベルに出すため、**クライアントは `sdxl-1.0` 固定を
前提にしないこと**。

---

## サーバー実装仕様

### ポート・プロトコル

- デフォルトポート: `8080` (コマンドライン引数で変更可)
- プロトコル: HTTP/1.1 (HTTPS 非対応)
- バインド: `127.0.0.1` のみ (LAN 公開しない)

### 実装方針 (cpp-httplib + nlohmann/json)

```cpp
// src/server/api.cpp — cpp-httplib でルーティング、nlohmann/json で入出力
#include <httplib.h>
#include <nlohmann/json.hpp>

httplib::Server svr;
svr.Post("/v1/images/generations",
    [&](const httplib::Request& req, httplib::Response& res)
    {
        auto body = nlohmann::json::parse(req.body);
        // body から prompt/steps/size を取り出しパイプラインへ
        // 結果 PNG を base64 化して JSON で返す
    });
svr.listen("127.0.0.1", 8080);
```

- ルーティング・ソケット・スレッド処理は cpp-httplib に委譲 (accept ループ・
  Content-Length 読み切りも内部処理)。手書き Winsock2 は不要。
- リクエスト/レスポンス JSON は nlohmann/json でパース・生成 (手書きパーサ不使用)。

### レスポンスヘッダ

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: <len>
Connection: close
```

### Base64 エンコード

PNG → base64 は cpp-httplib 付属のヘルパ、または数十行の小物で済ます
(自作する価値が薄い配管)。

### パイプラインとの接続

```
httplib ハンドラ (Post コールバック)
  ↓ nlohmann/json でリクエスト解析
  ↓ パイプラインキューに push (llm_to_clip_queue の手前)
  ↓ 結果キューを pop_wait (タイムアウト 60s)
  ↓ PNG エンコード → base64 → JSON レスポンス返却
```

結果は `std::promise<std::vector<uint8_t>>` で非同期受け渡し。

## コマンドライン引数

```
dollama [--port 8080] [--steps 20] [--width 1024] [--height 1024]
        [--seed <uint64>] [--cfg <float>]
```

- `--seed` / `--cfg` (E-2) は CLI 生成経路で上表の `seed` / `guidance_scale` と**同じ
  `GenRequest` フィールド**に入る (`src/main.cpp`)。未指定なら nullopt = 従来経路。
  値のパースに失敗した場合は**未指定として扱う** (エラーにしない)。
- ★**`--http` 起動時は `--seed` / `--cfg` は使われない (サーバ既定にはならない)**。`src/main.cpp` は
  `http_mode` なら `GenRequest` を組む前に `start_server` へ return するため、両引数は**パースされるだけで
  捨てられる** (HTTP 経路の `GenRequest` は `src/server/api.cpp` 側でしか組まれない)。
  HTTP で seed を固定するには body の `seed` か env `DOLLAMA_SEED` を使う。
- 上の並びは抜粋。CLI 引数の全量は `src/main.cpp` の引数分岐を参照。

## 将来拡張 (現フェーズ対象外)

- Server-Sent Events (SSE) でステップ進捗をストリーミング
- WebSocket 対応
- バッチ処理 (n > 1)
- HTTPS (TLS)
