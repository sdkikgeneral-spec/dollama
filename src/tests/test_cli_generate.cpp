// CLI 生成モードの生成器 DI + PNG 書き出しヘルパ単体テスト。
//
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 重み/OV アセットが無い CI 環境では build_image_generator() が必ず
// StubGenerator にフォールバックして有効 PNG を返すことを保証する
// (NPU/CUDA/OV に依存しない・Stub 経路のみで完結)。
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "server/cli_generate.hpp"
#include "server/generator.hpp"

namespace dollama
{

// PNG マジック先頭 4 バイト (0x89 'P' 'N' 'G')。
static bool starts_with_png_magic(const std::vector<uint8_t>& b)
{
    return b.size() >= 4 && b[0] == 0x89 && b[1] == 0x50 &&
           b[2] == 0x4E && b[3] == 0x47;
}

// build_image_generator() が生成器を返し、generate() が有効 PNG を返す。
static bool test_build_and_generate()
{
    // ログは捨てる (テスト出力を汚さない)。
    std::ostringstream sink;
    std::unique_ptr<IImageGenerator> gen = build_image_generator(sink);
    if (!gen)
    {
        std::cerr << "[cli] build_image_generator が nullptr を返した\n";
        return false;
    }

    GenRequest req{"1girl, solo, long hair", "", 1, 2, 64, 64};
    GenResult r = gen->generate(req);

    if (r.png_bytes.empty())
    {
        std::cerr << "[cli] png_bytes が空\n";
        return false;
    }
    if (!starts_with_png_magic(r.png_bytes))
    {
        std::cerr << "[cli] PNG マジック不一致\n";
        return false;
    }
    std::cout << "  [cli] generate OK (" << r.png_bytes.size()
              << " bytes, model=" << gen->model_id() << ")\n";
    return true;
}

// write_png_file() で一時ファイルへ書き → 読み戻して先頭マジック一致を確認。
static bool test_write_png_file()
{
    std::ostringstream sink;
    std::unique_ptr<IImageGenerator> gen = build_image_generator(sink);
    if (!gen)
    {
        std::cerr << "[write] build_image_generator が nullptr を返した\n";
        return false;
    }
    GenRequest req{"test prompt", "", 1, 2, 32, 32};
    GenResult r = gen->generate(req);

    const std::string path = "test_cli_generate_tmp.png";
    if (!write_png_file(path, r.png_bytes))
    {
        std::cerr << "[write] write_png_file が失敗\n";
        return false;
    }

    // 読み戻して先頭マジックを照合。
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "[write] 書き出したファイルを開けない\n";
        std::remove(path.c_str());
        return false;
    }
    std::vector<uint8_t> back((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
    ifs.close();
    std::remove(path.c_str()); // 後始末

    if (back.size() != r.png_bytes.size())
    {
        std::cerr << "[write] 読み戻しサイズ不一致 (" << back.size()
                  << " != " << r.png_bytes.size() << ")\n";
        return false;
    }
    if (!starts_with_png_magic(back))
    {
        std::cerr << "[write] 読み戻し PNG マジック不一致\n";
        return false;
    }
    std::cout << "  [write] write_png_file roundtrip OK (" << back.size() << " bytes)\n";
    return true;
}

// build_image_generator() のログに DI 注入の各行 (matting / scorer) が出ることを確認。
// 開発機 (OV 無) では make_matter / make_scorer が nullptr を返すため、
// それぞれ「無効」行が出る (M-6 / B-5-3 の DI 配管が通っている非回帰)。
static bool test_di_log_lines()
{
    std::ostringstream log;
    std::unique_ptr<IImageGenerator> gen = build_image_generator(log);
    if (!gen)
    {
        std::cerr << "[log] build_image_generator が nullptr を返した\n";
        return false;
    }
    const std::string text = log.str();

    // M-6: matting の DI 行が出ていること (対称性の非回帰)。
    if (text.find("matting:") == std::string::npos)
    {
        std::cerr << "[log] ログに matting: 行が無い\n";
        return false;
    }
    // B-5-3: scorer の DI 行が出ていること (開発機 OV 無では「scorer: 無効」)。
    if (text.find("scorer:") == std::string::npos)
    {
        std::cerr << "[log] ログに scorer: 行が無い\n";
        return false;
    }
    std::cout << "  [log] matting/scorer DI ログ行 OK\n";
    return true;
}

} // namespace dollama

int main()
{
    using namespace dollama;
    bool ok = true;

    std::cout << "=== test_cli_generate ===\n";
    ok = test_build_and_generate() && ok;
    ok = test_write_png_file() && ok;
    ok = test_di_log_lines() && ok;

    if (ok)
    {
        std::cout << "ALL PASSED\n";
        return 0;
    }
    std::cerr << "FAILED\n";
    return 1;
}
