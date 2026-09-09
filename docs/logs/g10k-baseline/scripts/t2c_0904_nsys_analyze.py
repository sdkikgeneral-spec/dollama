# G-10k T2c : timeline analysis of the nsys capture (post-processing only, no GPU work).
# Reports: total kernel GPU time, kernel-activity clusters (= generation phases),
#          wall vs sum(kernel) per cluster (launch gap), launch counts, memcpy.
import sqlite3, sys, collections

db = r'E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys.sqlite'
con = sqlite3.connect(db)
cur = con.cursor()

def tables():
    return {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
T = tables()
print("=== sqlite ===")
print("db =", db)
print("has KERNEL:", 'CUPTI_ACTIVITY_KIND_KERNEL' in T,
      " MEMCPY:", 'CUPTI_ACTIVITY_KIND_MEMCPY' in T,
      " RUNTIME:", 'CUPTI_ACTIVITY_KIND_RUNTIME' in T)

# ---- global kernel totals ----
cur.execute("SELECT COUNT(*), SUM(end-start), MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
kn, ksum, kmin, kmax = cur.fetchone()
print()
print("=== kernels (whole capture) ===")
print(f"kernel_instances = {kn}")
print(f"sum_kernel_gpu_time = {ksum/1e9:.6f} s")
print(f"kernel_activity_span (first start .. last end) = {(kmax-kmin)/1e9:.6f} s")
print(f"gap_over_span = {((kmax-kmin)-ksum)/1e9:.6f} s "
      f"({100.0*((kmax-kmin)-ksum)/(kmax-kmin):.2f} % of span)")

# ---- memcpy ----
if 'CUPTI_ACTIVITY_KIND_MEMCPY' in T:
    print()
    print("=== memcpy (whole capture) ===")
    q = """SELECT m.copyKind, COUNT(*), SUM(m.end-m.start), SUM(m.bytes)
           FROM CUPTI_ACTIVITY_KIND_MEMCPY m GROUP BY m.copyKind ORDER BY 3 DESC"""
    kindname = {1:'H2D', 2:'D2H', 8:'D2D', 10:'D2D(p2p)'}
    for ck, c, t, b in cur.execute(q):
        print(f"  copyKind={ck} ({kindname.get(ck,'?')}) count={c} time={t/1e9:.6f}s bytes={b/1e6:.3f}MB")

# ---- cluster kernels into phases (gap threshold) ----
rows = list(cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
GAP = 20_000_000  # 20 ms
clusters = []
cs, ce, csum, cnt = rows[0][0], rows[0][1], 0, 0
for s, e in rows:
    if s - ce > GAP:
        clusters.append((cs, ce, csum, cnt))
        cs, ce, csum, cnt = s, e, 0, 0
    csum += (e - s)
    cnt += 1
    if e > ce:
        ce = e
clusters.append((cs, ce, csum, cnt))

print()
print(f"=== kernel-activity clusters (split where consecutive-kernel gap > {GAP/1e6:.0f} ms) ===")
print(f"{'#':>3} {'t0_rel_s':>10} {'span_s':>10} {'sum_kern_s':>11} {'gap_s':>9} {'busy%':>7} {'kernels':>9}")
for i, (s, e, ks, n) in enumerate(clusters, 1):
    span = e - s
    print(f"{i:>3} {(s-kmin)/1e9:10.3f} {span/1e9:10.3f} {ks/1e9:11.3f} "
          f"{(span-ks)/1e9:9.3f} {100.0*ks/span:7.2f} {n:>9}")

# ---- top kernels within the two largest clusters ----
big = sorted(clusters, key=lambda c: -(c[1]-c[0]))[:3]
big = sorted(big, key=lambda c: c[0])
names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}
for i, (s, e, ks, n) in enumerate(big, 1):
    print()
    print(f"=== cluster t0_rel={(s-kmin)/1e9:.3f}s span={(e-s)/1e9:.3f}s : top kernels ===")
    q = f"""SELECT demangledName, COUNT(*), SUM(end-start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= {s} AND end <= {e}
            GROUP BY demangledName ORDER BY 3 DESC LIMIT 15"""
    for nid, c, t in cur.execute(q):
        print(f"  {t/1e9:9.4f}s  n={c:>6}  {names.get(nid,'?')[:100]}")

# ---- B=2 evidence: grid dims of the fast attention kernel ----
print()
print("=== grid/block dims (B=2 evidence) for selected kernels ===")
for pat in ('attention_flash_wmma_fast_fp16', 'k_split_heads', 'im2col_fp16'):
    q = f"""SELECT gridX, gridY, gridZ, blockX, blockY, blockZ, COUNT(*), SUM(end-start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.demangledName
            WHERE s.value LIKE '%{pat}%'
            GROUP BY gridX, gridY, gridZ, blockX, blockY, blockZ
            ORDER BY 7 DESC LIMIT 8"""
    print(f"  -- {pat} --")
    for gx, gy, gz, bx, by, bz, c, t in cur.execute(q):
        print(f"     grid=({gx},{gy},{gz}) block=({bx},{by},{bz}) n={c} t={t/1e9:.4f}s")

# ---- runtime API launch counts ----
if 'CUPTI_ACTIVITY_KIND_RUNTIME' in T:
    print()
    print("=== CUDA runtime API calls (top by count) ===")
    q = """SELECT s.value, COUNT(*), SUM(r.end-r.start)
           FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
           GROUP BY s.value ORDER BY 2 DESC LIMIT 12"""
    for nm, c, t in cur.execute(q):
        print(f"  n={c:>8} cpu_time={t/1e9:9.4f}s  {nm}")
print()
print("=== end ===")
