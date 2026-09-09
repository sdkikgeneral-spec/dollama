# Recover the profiled target's own stdout/stderr from the nsys sqlite (ProcessStreams).
# nsys 2026.1.3 on Windows does NOT pass the child's stdout through to the redirected
# console even with --show-output=true; it stores it in the report instead.
import sqlite3, sys
db, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db); cur = con.cursor()
names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}
with open(out, 'w', encoding='utf-8', newline='\n') as f:
    f.write(f"# recovered from {db} (table ProcessStreams)\n")
    for gpid, fid, cid in cur.execute("SELECT globalPid, filenameId, contentId FROM ProcessStreams"):
        fn = names.get(fid, str(fid))
        body = names.get(cid, '') if isinstance(cid, int) else (cid or '')
        f.write(f"\n===== stream: {fn} (globalPid={gpid}) =====\n")
        f.write(body if body else "(empty)\n")
print("wrote", out)
