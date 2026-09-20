// E-0: VAE scaling_factor の server/ 層 (cpp) 側共有既定値 — 単独の極小ヘッダ。
//
// ----------------------------------------------------------------
// なぜ preset.hpp に同居させず、独立した極小ヘッダにしているか (重要・レビュー是正③):
//   diffusion_runner.hpp / diffusion_backend.hpp は diffusion_runner.cu (nvcc 経由の
//   .cu TU) から include される。nvcc のホストコンパイルは CUDA13+MSVC の既知の制約
//   (reference_cuda_meson_build) で /std:c++14 に固定されており、server/preset.hpp が
//   使う <filesystem> / <optional> (いずれも C++17) をこの経路へ引き込むと
//   「namespace "std" has no member "filesystem"」等でビルドが壊れる
//   (実際に一度このビルド破壊を踏んでいる — この経緯で本ヘッダを切り出した)。
//
//   本ヘッダは #include 依存を一切持たない定数 1 個だけにすることで、
//   diffusion_runner.cu (.cu・/std:c++14) からも preset.hpp 側 (cpp・任意の std バージョン)
//   からも安全に include できるようにする。
//
//   infer/diffusion.cuh 側の kDefaultVaeScalingFactor (CUDA 層の正典) とは名前を分けている
//   (同一 TU に両方が同時 include されたときの「同一 namespace 内再定義」エラーを避けるため)。
//   値の一致は server/diffusion_runner.cu の static_assert がコンパイル時に保証する。
// ----------------------------------------------------------------
#pragma once

namespace dollama
{

// server/ 層 (BackendConfig / make_diffusion_runner / SDXLBackend / preset.json パース) が
// 共有する VAE scaling_factor の既定値。infer/diffusion.cuh の kDefaultVaeScalingFactor と
// 同値でなければならない (値を変えるときは両方揃えること)。
constexpr float kServerDefaultVaeScalingFactor = 0.13025f;

} // namespace dollama
