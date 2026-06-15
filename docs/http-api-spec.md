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

利用可能モデル一覧 (OpenAI 互換)。

**レスポンス (200 OK)**:

```json
{
  "data": [
    { "id": "sdxl-1.0", "object": "model" }
  ]
}
```

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
```

## 将来拡張 (現フェーズ対象外)

- Server-Sent Events (SSE) でステップ進捗をストリーミング
- WebSocket 対応
- バッチ処理 (n > 1)
- HTTPS (TLS)
