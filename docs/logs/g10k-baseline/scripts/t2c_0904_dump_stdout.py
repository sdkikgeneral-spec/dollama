import sqlite3, sys
db = sys.argv[1]
con = sqlite3.connect(db); cur = con.cursor()
cols = [r[1] for r in cur.execute("PRAGMA table_info(ProcessStreams)")]
print("ProcessStreams cols:", cols)
names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}
for row in cur.execute("SELECT * FROM ProcessStreams"):
    d = dict(zip(cols, row))
    print("---- row:", {k: v for k, v in d.items() if k not in ('content','filenameId','contentId')})
    for k in ('contentId', 'content', 'filenameId'):
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, int) and v in names:
                v = names[v]
            print(f"[{k}]")
            print(v if isinstance(v, str) else repr(v)[:200])
