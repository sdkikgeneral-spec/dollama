#!/usr/bin/env python3
# safetensors C++ ローダー (src/io/safetensors.hpp) のゴールデンフィクスチャ生成。
#
# 既知の小テンソルを dtype 網羅で書き出し、C++ テスト (test_safetensors.cpp) が
# バイト/値で突合できるようにする。各テンソルの dtype/shape/期待値を stdout に出力する。
#
# 出力先: src/tests/data/golden.safetensors
#
# BF16 は numpy に型が無いので torch.bfloat16 で書き出す (この環境には torch がある)。
# 値は固定の既知値 (arange / ハードコード) にする。

import os
import struct

import numpy as np
from safetensors.numpy import save_file as save_file_np

# BF16 用 (torch がある環境を前提。無ければ uint16 ビット構成にフォールバック)
try:
    import torch
    from safetensors.torch import save_file as save_file_torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "tests", "data")
OUT_PATH = os.path.abspath(os.path.join(OUT_DIR, "golden.safetensors"))


def f32_to_bf16_bits(x: float) -> int:
    # float32 を bf16 (上位16bit、round-to-nearest-even) へ変換しビットを返す。
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    # round to nearest even
    rounding_bias = 0x7FFF + ((b >> 16) & 1)
    b = (b + rounding_bias) >> 16
    return b & 0xFFFF


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- numpy で書ける dtype 群 (固定既知値) ----
    t_f32 = np.arange(6, dtype=np.float32).reshape(2, 3)          # [0,1,2,3,4,5]
    t_f16 = np.array([1.5, -2.0, 0.0, 65504.0], dtype=np.float16)  # [4]
    t_i64 = np.array([[1, 2], [3, 4]], dtype=np.int64)            # [2,2]
    t_i8 = np.array([-128, 0, 127], dtype=np.int8)               # [3]

    np_tensors = {
        "t_f32": t_f32,
        "t_f16": t_f16,
        "t_i64": t_i64,
        "t_i8": t_i8,
    }

    # ---- BF16 [2] ----
    # 既知値: [1.0, -2.5]
    bf16_vals = [1.0, -2.5]
    bf16_bits = [f32_to_bf16_bits(v) for v in bf16_vals]

    if HAVE_TORCH:
        # torch.bfloat16 で書く。全テンソルを torch 経由で保存する
        # (safetensors は1ファイル1フレームワークなので混在させない)。
        torch_tensors = {k: torch.from_numpy(v.copy()) for k, v in np_tensors.items()}
        torch_tensors["t_bf16"] = torch.tensor(bf16_vals, dtype=torch.bfloat16)
        save_file_torch(torch_tensors, OUT_PATH)
        bf16_method = "torch.bfloat16"
    else:
        # フォールバック: bf16 を uint16 ビットで詰め、後で dtype を BF16 に書換える
        raise RuntimeError("torch が無い環境。BF16 書き出しに torch が必要")

    print("=== golden.safetensors 生成完了 ===")
    print("path:", OUT_PATH)
    print("bf16 method:", bf16_method)
    print()

    def dump(name, dtype, shape, raw_bytes_hint, values):
        print(f"[{name}] dtype={dtype} shape={list(shape)}")
        print(f"  values={values}")
        print(f"  raw_le_bytes(first16)={raw_bytes_hint[:16].hex()}")
        print()

    dump("t_f32", "F32", t_f32.shape, t_f32.tobytes(), t_f32.flatten().tolist())
    dump("t_f16", "F16", t_f16.shape, t_f16.tobytes(), t_f16.flatten().tolist())
    dump("t_i64", "I64", t_i64.shape, t_i64.tobytes(), t_i64.flatten().tolist())
    dump("t_i8", "I8", t_i8.shape, t_i8.tobytes(), t_i8.flatten().tolist())

    bf16_raw = b"".join(struct.pack("<H", b) for b in bf16_bits)
    print(f"[t_bf16] dtype=BF16 shape=[2]")
    print(f"  values={bf16_vals}")
    print(f"  bf16_bits_le={[hex(b) for b in bf16_bits]}")
    print(f"  raw_le_bytes={bf16_raw.hex()}")
    print()

    # ファイルサイズ・ヘッダ長も出す (テストのデバッグ用)
    with open(OUT_PATH, "rb") as f:
        head = f.read(8)
        n = struct.unpack("<Q", head)[0]
        f.seek(0)
        total = len(f.read())
    print(f"header_len(N)={n}  total_file_bytes={total}  data_offset={8 + n}")


if __name__ == "__main__":
    main()
