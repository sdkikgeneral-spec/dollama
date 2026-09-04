# G-10k T2c : split generation #2 into (UNet 20 steps) vs (VAE decode) and count launches/step.
# Boundary = end of the LAST dollama::attention_flash_wmma_fast_fp16 in the window
#            (the UNet transformer blocks are the only user of the *_fast_* kernel;
#             the VAE mid-block uses the non-fast dollama::attention_flash_wmma_fp16).
import sqlite3, sys
db = sys.argv[1]
con = sqlite3.connect(db); cur = con.cursor()
names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}
cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"); kmin = cur.fetchone()[0]

marks = [e for (e,) in cur.execute(
    "SELECT end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2 AND bytes>=6000000 ORDER BY start")]
lo, hi = marks[0], marks[1]          # generation #2

cur.execute("""SELECT MAX(k.end) FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON s.id=k.demangledName
               WHERE k.start>=? AND k.end<=? AND s.value LIKE '%attention_flash_wmma_fast_fp16%'""", (lo, hi))
split = cur.fetchone()[0]

def seg(a, b, tag, steps=None):
    cur.execute("SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=?", (a, b))
    n, s = cur.fetchone()
    wall = b - a
    print()
    print(f"=== {tag} ===")
    print(f"  window   = [{(a-kmin)/1e9:.4f}s .. {(b-kmin)/1e9:.4f}s]  wall = {wall/1e9:.4f} s")
    print(f"  kernels  = {n}" + (f"   -> {n/steps:.1f} launches / diffusion step" if steps else ""))
    print(f"  sum_kern = {s/1e9:.4f} s   wall-sum_kern = {(wall-s)/1e9:.4f} s "
          f"({100.0*(wall-s)/wall:.2f} % of wall)   busy = {100.0*s/wall:.2f} %")
    q = ("SELECT demangledName, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL "
         "WHERE start>=? AND end<=? GROUP BY demangledName ORDER BY 3 DESC LIMIT 12")
    for nid, c, t in cur.execute(q, (a, b)):
        print(f"    {t/1e9:8.4f}s {100.0*t/s:6.2f}%  n={c:>6}  {names.get(nid,'?')[:92]}")
    return n, s, wall

seg(lo, split, "generation #2 : UNet part (20 diffusion steps, batch2 B=2)", steps=20)
seg(split, hi, "generation #2 : VAE decode part (everything after the last UNet attention)")

# per-kernel-name attention detail
print()
print("=== attention kernels in generation #2 (grid = (qtiles, B*H, 1)) ===")
q = """SELECT s.value, k.gridX, k.gridY, k.blockX, COUNT(*), SUM(k.end-k.start), AVG(k.end-k.start)
       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
       WHERE k.start>=? AND k.end<=? AND s.value LIKE '%attention_flash%'
       GROUP BY s.value, k.gridX, k.gridY, k.blockX ORDER BY 6 DESC"""
for nm, gx, gy, bx, c, t, avg in cur.execute(q, (lo, hi)):
    short = nm.split('(')[0]
    print(f"  {short:48s} grid=({gx},{gy}) block={bx} n={c:>5} total={t/1e9:7.4f}s avg={avg/1e6:8.3f}ms")
print()
print("=== end ===")
