# HTTP API 仕様 — dollama サーバー

## 概要

Winsock2 を直接使った OpenAI Images API 互換 HTTP サーバー。
フレームワーク不使用、`src/server/http.cpp` にスクラッチ実装する。

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

### Winsock2 実装方針

```cpp
// src/server/http.hpp
class HttpServer
{
public:
    explicit HttpServer(uint16_t port = 8080);
    void start();   // ブロッキング; 内部でスレッドプールを回す
    void stop();

private:
    SOCKET listen_sock_;
    std::atomic<bool> running_{false};
    void handle_client(SOCKET client);
    std::string dispatch(const HttpRequest& req);
};
```

- `accept()` ループ → 1接続 = 1スレッド (同時接続数が少ないため)
- リクエストボディは `Content-Length` で読み切る (チャンク転送非対応)
- JSON は手書きパース (`sscanf` + 文字列検索) または minijson (ヘッダオンリー)

### レスポンスヘッダ

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: <len>
Connection: close
```

### Base64 エンコード

PNG → base64 は STL のみで実装 (`src/server/base64.hpp`)。

### パイプラインとの接続

```
HttpServer::handle_client()
  ↓ リクエスト解析
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
