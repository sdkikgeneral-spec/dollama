# openvino_tokenizers.dll の絶対パスを stdout に出力する補助スクリプト。
# meson の run_command から呼び、TextConditioner テストの -DTOKENIZERS_DLL に埋め込む。
# パッケージが無い環境では何も出力せず終了コード 0 で抜ける (テスト側で [SKIP])。
import os
import sys

try:
    import openvino_tokenizers as t
except Exception:
    sys.exit(0)

dll = os.path.join(os.path.dirname(t.__file__), "lib", "openvino_tokenizers.dll")
if os.path.exists(dll):
    # MSVC の -D 文字列マクロでバックスラッシュが壊れないよう / に正規化する。
    sys.stdout.write(dll.replace("\\", "/"))
