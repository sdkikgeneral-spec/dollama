// safetensors 重みローダー 単体テスト + ベンチ
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 期待値は scripts/dollma_make_safetensors_fixture.py が生成した
// src/tests/data/golden.safetensors の既知値 (スクリプト stdout 参照)。
//   t_f32  F32  [2,3] = arange(6) = {0,1,2,3,4,5}
//   t_f16  F16  [4]   = {1.5, -2.0, 0.0, 65504.0}
//   t_i64  I64  [2,2] = {1,2,3,4}
//   t_i8   I8   [3]   = {-128, 0, 127}
//   t_bf16 BF16 [2]   = {1.0, -2.5}  bf16 bits LE = {0x3f80, 0xc020}
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "io/safetensors.hpp"

#ifndef GOLDEN_PATH
#define GOLDEN_PATH "data/golden.safetensors"
#endif

namespace dollama {

static const std::string g_golden = GOLDEN_PATH;

// ----------------------------------------------------------------
// names 一覧に 5 テンソルが揃い、__metadata__ が混入しないこと
// ----------------------------------------------------------------
static bool test_names()
{
    SafeTensors st(g_golden);
    std::vector<std::string> ns = st.names();

    // 5 テンソル
    if (ns.size() != 5)
    {
        std::cerr << "[test_names] テンソル数が 5 でない: " << ns.size() << "\n";
        for (const auto& n : ns)
        {
            std::cerr << "  got: " << n << "\n";
        }
        return false;
    }
    const char* want[] = {"t_bf16", "t_f16", "t_f32", "t_i64", "t_i8"};
    for (const char* w : want)
    {
        if (!st.contains(w))
        {
            std::cerr << "[test_names] '" << w << "' が見つからない\n";
            return false;
        }
    }
    // __metadata__ がテンソルとして混入しないこと
    if (st.contains("__metadata__"))
    {
        std::cerr << "[test_names] __metadata__ がテンソルとして混入\n";
        return false;
    }
    std::cout << "[test_names] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// 各テンソルの dtype / shape が期待と一致
// ----------------------------------------------------------------
static bool test_dtype_shape()
{
    SafeTensors st(g_golden);

    struct Expect
    {
        const char*         name;
        StDtype             dt;
        std::vector<size_t> shape;
    };
    std::vector<Expect> exp = {
        {"t_f32",  StDtype::F32,  {2, 3}},
        {"t_f16",  StDtype::F16,  {4}},
        {"t_i64",  StDtype::I64,  {2, 2}},
        {"t_i8",   StDtype::I8,   {3}},
        {"t_bf16", StDtype::BF16, {2}},
    };

    for (const auto& e : exp)
    {
        if (st.dtype(e.name) != e.dt)
        {
            std::cerr << "[test_dtype_shape] " << e.name << " dtype 不一致: got "
                      << st_dtype_name(st.dtype(e.name)) << " want "
                      << st_dtype_name(e.dt) << "\n";
            return false;
        }
        if (st.shape(e.name) != e.shape)
        {
            std::cerr << "[test_dtype_shape] " << e.name << " shape 不一致\n";
            return false;
        }
    }
    std::cout << "[test_dtype_shape] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// F32 [2,3] の中身を float 値で突合
// ----------------------------------------------------------------
static bool test_f32_values()
{
    SafeTensors st(g_golden);
    size_t nb = 0;
    const uint8_t* p = st.tensor_bytes("t_f32", nb);
    if (nb != 6 * sizeof(float))
    {
        std::cerr << "[test_f32_values] バイト数不一致: " << nb << "\n";
        return false;
    }
    float vals[6];
    std::memcpy(vals, p, nb);
    for (int i = 0; i < 6; ++i)
    {
        if (vals[i] != static_cast<float>(i))
        {
            std::cerr << "[test_f32_values] vals[" << i << "]=" << vals[i]
                      << " want " << i << "\n";
            return false;
        }
    }
    std::cout << "[test_f32_values] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// I64 [2,2] の中身を int64 値で突合
// ----------------------------------------------------------------
static bool test_i64_values()
{
    SafeTensors st(g_golden);
    size_t nb = 0;
    const uint8_t* p = st.tensor_bytes("t_i64", nb);
    if (nb != 4 * sizeof(int64_t))
    {
        std::cerr << "[test_i64_values] バイト数不一致: " << nb << "\n";
        return false;
    }
    int64_t vals[4];
    std::memcpy(vals, p, nb);
    int64_t want[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i)
    {
        if (vals[i] != want[i])
        {
            std::cerr << "[test_i64_values] vals[" << i << "]=" << vals[i]
                      << " want " << want[i] << "\n";
            return false;
        }
    }
    std::cout << "[test_i64_values] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// I8 [3] の中身を int8 値で突合 (符号付き境界)
// ----------------------------------------------------------------
static bool test_i8_values()
{
    SafeTensors st(g_golden);
    size_t nb = 0;
    const uint8_t* p = st.tensor_bytes("t_i8", nb);
    if (nb != 3)
    {
        std::cerr << "[test_i8_values] バイト数不一致: " << nb << "\n";
        return false;
    }
    int8_t vals[3];
    std::memcpy(vals, p, nb);
    int8_t want[3] = {-128, 0, 127};
    for (int i = 0; i < 3; ++i)
    {
        if (vals[i] != want[i])
        {
            std::cerr << "[test_i8_values] vals[" << i << "]=" << static_cast<int>(vals[i])
                      << " want " << static_cast<int>(want[i]) << "\n";
            return false;
        }
    }
    std::cout << "[test_i8_values] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// F16 [4] と BF16 [2] を raw 16bit ビットパターンで突合
// (float 変換せずビットで照合する → ローダーの生バイト透過性を検証)
// ----------------------------------------------------------------
static bool test_half_bits()
{
    SafeTensors st(g_golden);

    // F16: numpy float16 ビット。1.5=0x3e00, -2.0=0xc000, 0.0=0x0000, 65504=0x7bff
    {
        size_t nb = 0;
        const uint8_t* p = st.tensor_bytes("t_f16", nb);
        if (nb != 4 * 2)
        {
            std::cerr << "[test_half_bits] f16 バイト数不一致: " << nb << "\n";
            return false;
        }
        uint16_t bits[4];
        std::memcpy(bits, p, nb);
        uint16_t want[4] = {0x3e00, 0xc000, 0x0000, 0x7bff};
        for (int i = 0; i < 4; ++i)
        {
            if (bits[i] != want[i])
            {
                std::cerr << "[test_half_bits] f16 bits[" << i << "]=0x"
                          << std::hex << bits[i] << " want 0x" << want[i] << std::dec << "\n";
                return false;
            }
        }
    }

    // BF16: 1.0=0x3f80, -2.5=0xc020
    {
        size_t nb = 0;
        const uint8_t* p = st.tensor_bytes("t_bf16", nb);
        if (nb != 2 * 2)
        {
            std::cerr << "[test_half_bits] bf16 バイト数不一致: " << nb << "\n";
            return false;
        }
        uint16_t bits[2];
        std::memcpy(bits, p, nb);
        uint16_t want[2] = {0x3f80, 0xc020};
        for (int i = 0; i < 2; ++i)
        {
            if (bits[i] != want[i])
            {
                std::cerr << "[test_half_bits] bf16 bits[" << i << "]=0x"
                          << std::hex << bits[i] << " want 0x" << want[i] << std::dec << "\n";
                return false;
            }
        }
    }
    std::cout << "[test_half_bits] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// エラー系: 存在しないファイルで例外
// ----------------------------------------------------------------
static bool test_missing_file_throws()
{
    bool threw = false;
    try
    {
        SafeTensors st("__no_such_file__.safetensors");
        (void)st;
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    if (!threw)
    {
        std::cerr << "[test_missing_file_throws] 例外が飛ばなかった\n";
        return false;
    }
    std::cout << "[test_missing_file_throws] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// エラー系: ヘッダ長を過大に細工した一時ファイルで例外
// ----------------------------------------------------------------
static bool test_corrupt_header_throws()
{
    // 正常ファイルを読み込み、先頭 8 バイト (ヘッダ長) を巨大値に書換えて一時保存
    std::ifstream in(g_golden, std::ios::binary);
    if (!in)
    {
        std::cerr << "[test_corrupt_header_throws] golden を開けない\n";
        return false;
    }
    std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    in.close();
    if (buf.size() < 8)
    {
        std::cerr << "[test_corrupt_header_throws] golden が小さすぎる\n";
        return false;
    }

    // 先頭 8 バイト LE をファイルサイズを超える値に細工
    uint64_t huge = 0xFFFFFFFFFFFFFFFFull;
    std::memcpy(buf.data(), &huge, 8);

    std::string tmp = "corrupt_tmp.safetensors";
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    }

    bool threw = false;
    try
    {
        SafeTensors st(tmp);
        (void)st;
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    std::remove(tmp.c_str());

    if (!threw)
    {
        std::cerr << "[test_corrupt_header_throws] 破損ヘッダで例外が飛ばなかった\n";
        return false;
    }
    std::cout << "[test_corrupt_header_throws] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// エラー系: 存在しないテンソル名アクセスで例外
// ----------------------------------------------------------------
static bool test_unknown_tensor_throws()
{
    SafeTensors st(g_golden);
    bool threw = false;
    try
    {
        size_t nb = 0;
        st.tensor_bytes("__nope__", nb);
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    if (!threw)
    {
        std::cerr << "[test_unknown_tensor_throws] 例外が飛ばなかった\n";
        return false;
    }
    std::cout << "[test_unknown_tensor_throws] PASSED\n";
    return true;
}

// ----------------------------------------------------------------
// ベンチ: golden を N 回ロードして µs/op を出力
// PASS/FAIL 判定はせず数値のみ出力する。
// ----------------------------------------------------------------
static void bench_load()
{
    const int iters = 10'000;
    volatile size_t sink = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        SafeTensors st(g_golden);
        sink += st.names().size();
    }
    auto t1 = std::chrono::steady_clock::now();

    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "[bench_load] " << (us / iters) << " us/op ("
              << iters << " iters, sink=" << sink << ")\n";
}

} // namespace dollama

int main()
{
    bool ok = true;
    ok = dollama::test_names()                 && ok;
    ok = dollama::test_dtype_shape()           && ok;
    ok = dollama::test_f32_values()            && ok;
    ok = dollama::test_i64_values()            && ok;
    ok = dollama::test_i8_values()             && ok;
    ok = dollama::test_half_bits()             && ok;
    ok = dollama::test_missing_file_throws()   && ok;
    ok = dollama::test_corrupt_header_throws() && ok;
    ok = dollama::test_unknown_tensor_throws() && ok;

    // ベンチ (PASS/FAIL 判定には含めない。数値出力のみ)
    dollama::bench_load();

    if (!ok)
    {
        std::cerr << "[test_safetensors] FAILED\n";
        return 1;
    }
    std::cout << "[test_safetensors] ALL PASSED\n";
    return 0;
}
