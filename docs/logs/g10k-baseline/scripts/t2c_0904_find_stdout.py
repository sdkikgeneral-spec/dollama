import sqlite3, re
db = r'E:\Develop\logs\g10k-t2c\nsys\g10k_t2c_nsys2.sqlite'
con = sqlite3.connect(db); cur = con.cursor()
tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("n_tables =", len(tabs))
hits = [t for t in tabs if re.search(r'stdout|output|stream|console|target', t, re.I)]
print("candidate tables:", hits)
for t in hits:
    try:
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: rows={n}")
    except Exception as ex:
        print(f"  {t}: {ex}")
# search StringIds for our marker
try:
    rows = list(cur.execute("SELECT value FROM StringIds WHERE value LIKE '%[S4]%' LIMIT 20"))
    print("StringIds hits for [S4]:", len(rows))
    for r in rows[:20]:
        print("   ", r[0][:160])
except Exception as ex:
    print("StringIds search failed:", ex)
