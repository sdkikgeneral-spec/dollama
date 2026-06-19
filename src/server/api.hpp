// OpenAI Images 互換 HTTP API (cpp-httplib + nlohmann/json)
//
// HTTP サーバ層。生成器は IImageGenerator interface 越しに受け取り、中身は知らない。
// ルーティング・ソケット・スレッドは cpp-httplib に委譲。CUDA / OpenVINO 非依存。
//
// テストから再利用できるよう、ハンドラ登録を register_routes() に分離し、
// httplib::Server を外部から渡せる形にする (テストは bind_to_any_port で起動)。
#pragma once

#include <string>

#include <httplib.h>

#include "server/generator.hpp"

namespace dollama
{

// 与えた httplib::Server に全ルートを登録する。
// gen は呼び出し側が生存を保証する (参照を保持する)。
void register_routes(httplib::Server& svr, IImageGenerator& gen);

// 127.0.0.1:port でブロッキング起動する (本番用ヘルパ)。
// 戻り値: listen に成功し正常終了したら 0、bind 失敗なら 1。
int start_server(IImageGenerator& gen, const std::string& host, int port);

} // namespace dollama
