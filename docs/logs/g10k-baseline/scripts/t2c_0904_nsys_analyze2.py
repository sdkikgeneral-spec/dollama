# G-10k T2c : per-generation timeline split (post-processing only).
# Generation boundary marker = the final image D2H copy (1024*1024*3 halves = 6,291,456 B).
import sqlite3, sys

db = sys.argv[1] if len(sys.argv) > 1 else r'E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys.sqlite'
con = sqlite3.connect(db); cur = con.cursor()
names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}

print("db =", db)
cur.execute("SELECT MIN(start), MAX(end), COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
kmin, kmax, kn, ksum = cur.fetchone()

print()
print("=== D2H copies >= 1 MB (image readback markers) ===")
marks = []
for s, e, b in cur.execute("SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                           "WHERE copyKind=2 AND bytes>=1000000 ORDER BY start"):
    print(f"  t_rel={(s-kmin)/1e9:9.4f}s  bytes={b}  dur={(e-s)/1e6:.3f}ms")
    marks.append(e)

print()
print("=== H2D copies >= 8 MB (weight-upload markers: first/last) ===")
big = list(cur.execute("SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                       "WHERE copyKind=1 AND bytes>=8000000 ORDER BY start"))
if big:
    print(f"  count={len(big)}  first_t_rel={(big[0][0]-kmin)/1e9:.4f}s  "
          f"last_end_t_rel={(big[-1][1]-kmin)/1e9:.4f}s  "
          f"sum_bytes={sum(b for _,_,b in big)/1e6:.1f}MB")

def window(lo, hi, tag):
    cur.execute("SELECT COUNT(*), SUM(end-start), MIN(start), MAX(end) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (lo, hi))
    n, s, a, b = cur.fetchone()
    span = hi - lo
    print()
    print(f"=== {tag} ===")
    print(f"  window            = [{(lo-kmin)/1e9:.4f}s .. {(hi-kmin)/1e9:.4f}s]  wall={span/1e9:.4f} s")
    print(f"  kernels           = {n}")
    print(f"  sum_kernel_time   = {s/1e9:.4f} s")
    print(f"  wall - sum_kernel = {(span-s)/1e9:.4f} s   (launch gap; {100.0*(span-s)/span:.2f} % of wall)")
    print(f"  GPU busy          = {100.0*s/span:.2f} %")
    cur.execute("SELECT COUNT(*), SUM(end-start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE start>=? AND end<=? AND copyKind=1", (lo, hi))
    print("  H2D in window     =", cur.fetchone())
    cur.execute("SELECT COUNT(*), SUM(end-start), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE start>=? AND end<=? AND copyKind=8", (lo, hi))
    print("  D2D in window     =", cur.fetchone())
    print("  top kernels:")
    q = ("SELECT demangledName, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
         "WHERE start>=? AND end<=? GROUP BY demangledName ORDER BY 3 DESC LIMIT 15")
    for nid, c, t in cur.execute(q, (lo, hi)):
        print(f"    {t/1e9:8.4f}s  {100.0*t/s:5.2f}%  n={c:>6}  {names.get(nid,'?')[:95]}")
    return n, s, span

if len(marks) >= 2:
    # gen2 = strictly between the two image readbacks -> exactly one generate_txt2img
    window(marks[0], marks[1], "generation #2 (between image1 D2H end and image2 D2H end)")
    lo = big[-1][1] if big else kmin
    window(lo, marks[0], "generation #1 (last big H2D end .. image1 D2H end)")

print()
print("=== whole capture ===")
print(f"  kernel_instances = {kn}")
print(f"  sum_kernel_time  = {ksum/1e9:.4f} s")
print(f"  activity_span    = {(kmax-kmin)/1e9:.4f} s")
print("=== end ===")
