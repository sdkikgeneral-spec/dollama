# G-10k T6r: nsys cuda_gpu_kern_sum CSV (t6r/*.csv) から per-call 内訳表を生成する (iters=50 で割る)。
#   python docs/logs/g10k-resnet/scripts/t6r_table.py > docs/logs/g10k-resnet/t6r/breakdown_table.txt
import csv, os, re
d = os.path.join(os.path.dirname(__file__), '..', 't6r')
ITERS = 50
def cat(name):
    if 'im2col' in name: return 'im2col'
    if 'cutlass' in name or 'nvjet' in name or 'gemm' in name.lower(): return 'GEMM'
    if 'bias' in name: return 'bias'
    if 'scatter' in name: return 'scatter'
    return 'other'
shapes = ['rep_320_128', 'rep_640_64', 'rep_1280_32', 'G4_band_640to320_128', 'unet_c320_64']
cols = ['im2col', 'GEMM', 'bias', 'scatter', 'other']
print(f"{'shape':<22} {'mode':<8} " + ' '.join(f'{c:>9}' for c in cols) + f" {'sum':>9}  GEMM kernel")
res = {}
for s in shapes:
    for m in ('batched', 'seq'):
        rows = list(csv.DictReader(open(os.path.join(d, f'{s}_{m}_cuda_gpu_kern_sum.csv'), encoding='utf-8-sig')))
        acc = {c: 0.0 for c in cols}; gk = []
        for r in rows:
            c = cat(r['Name']); acc[c] += float(r['Total Time (ns)']) / 1e3 / ITERS
            if c == 'GEMM': gk.append(re.sub(r'^void ', '', r['Name'])[:60] + f" x{int(r['Instances'])//ITERS}/call")
        tot = sum(acc.values()); res[(s, m)] = (acc, tot)
        print(f"{s:<22} {m:<8} " + ' '.join(f'{acc[c]:9.1f}' for c in cols) + f" {tot:9.1f}  {'; '.join(gk)}")
print('\n(単位 us / call・nsys GPU カーネル時間の合計・cudaEvent 計測とは別物)')
print(f"\n{'shape':<22} {'batched/seq (kernel sum)':>26} {'batch-only overhead (us)':>26}  内訳: bias差 / scatter / im2col差 / GEMM差")
for s in shapes:
    b, tb = res[(s, 'batched')]; q, tq = res[(s, 'seq')]
    print(f"{s:<22} {tb/tq:26.3f} {tb-tq:26.1f}  {b['bias']-q['bias']:+.1f} / {b['scatter']:+.1f} / {b['im2col']-q['im2col']:+.1f} / {b['GEMM']-q['GEMM']:+.1f}")
