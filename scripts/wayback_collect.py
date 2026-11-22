"""
Archived web footprint of every TM-RugPull project, from the Internet Archive.

For each project the script asks the Wayback Machine CDX API on which days the project's
website (any page on its host) and its X/Twitter profile page were captured, and counts the
distinct capture days up to the start date and up to the lifetime midpoint:

    wb_site_start, wb_site_mid   website host
    wb_x_start,    wb_x_mid      twitter.com/<handle> plus x.com/<handle>

Each count is given twice: for the start date recorded in the dataset, and (columns ending
in _oc) for the on-chain start date, i.e. the token's first transfer.

A capture carries the date on which it was made and never changes, so the counts are
point-in-time values that anyone can recompute. An empty cell means "not measurable" (no
website or handle recorded, or the website is a page on a shared platform); 0 means the
archive holds no capture up to that date.

Run on any computer with Python 3 and internet access (no extra packages needed):

    python wayback_collect.py                 # reads wayback_targets.csv, writes wayback_counts.csv
    python wayback_collect.py --recheck 40    # measure 40 random projects again (repeatability)
    python wayback_collect.py --rezero        # measure again every project with a zero count
                                              # (writes wayback_counts_rezero.csv)

The script can be stopped and started again; finished rows are skipped. It sends about one
request per second, so the full run takes one to two hours.
"""
import csv
import datetime
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = os.path.join(HERE, "wayback_targets.csv")
OUT = os.path.join(HERE, "wayback_counts.csv")
CDX = "https://web.archive.org/cdx/search/cdx"
COLS = ["row", "website_host", "x_handle", "start", "mid", "start_oc", "mid_oc",
        "wb_site_start", "wb_site_mid", "wb_site_start_oc", "wb_site_mid_oc", "wb_site_first",
        "wb_x_start", "wb_x_mid", "wb_x_start_oc", "wb_x_mid_oc", "wb_x_first", "collected_at", "note"]
PAUSE = 1.0


def capture_days(url, match, to):
    """Distinct capture days (YYYYMMDD) of `url` up to `to`, successful captures only."""
    q = {"url": url, "to": to, "fl": "timestamp", "collapse": "timestamp:8", "filter": "statuscode:200",
         "limit": "100000"}
    if match:
        q["matchType"] = match
    full = CDX + "?" + urllib.parse.urlencode(q)
    confirmed = False
    for attempt in range(7):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "TM-RugPull dataset research (academic)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                text = r.read().decode("utf-8", "replace")
            time.sleep(PAUSE)
            days = sorted({ln[:8] for ln in text.split("\n") if ln[:8].isdigit()})
            if not days and not confirmed:          # an empty answer can be a transient glitch
                confirmed = True
                time.sleep(5)
                continue
            return days
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 504):
                time.sleep(30 * (attempt + 1))
                continue
            if e.code in (403, 404):            # host excluded from the archive, or nothing archived
                time.sleep(PAUSE)
                if not confirmed:
                    confirmed = True
                    time.sleep(5)
                    continue
                return [] if e.code == 404 else None
            time.sleep(10)
        except Exception:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("no answer from the archive for " + url)


def measure(t):
    row = {k: t.get(k, "") for k in ("row", "website_host", "x_handle", "start", "mid", "start_oc", "mid_oc")}
    last = max(t["mid"], t.get("mid_oc") or "")

    def counts(prefix, days):
        row[prefix + "_start"] = sum(d <= t["start"] for d in days)
        row[prefix + "_mid"] = sum(d <= t["mid"] for d in days)
        if t.get("start_oc"):
            row[prefix + "_start_oc"] = sum(d <= t["start_oc"] for d in days)
            row[prefix + "_mid_oc"] = sum(d <= t["mid_oc"] for d in days)
        row[prefix + "_first"] = days[0] if days else ""

    row["collected_at"] = datetime.date.today().isoformat()
    notes = []
    if t["website_host"]:
        days = capture_days(t["website_host"], "host", last)
        if days is None:
            notes.append("website excluded from the archive")
        else:
            counts("wb_site", days)
    else:
        notes.append(t.get("website_note") or "no website")
    if t["x_handle"]:
        days = set()
        for dom in ("twitter.com/", "x.com/"):
            d = capture_days(dom + t["x_handle"], None, last)
            days |= set(d or [])
        counts("wb_x", sorted(days))
    else:
        notes.append("no X handle")
    row["note"] = "; ".join(notes)
    return row


def main():
    recheck = int(sys.argv[sys.argv.index("--recheck") + 1]) if "--recheck" in sys.argv else 0
    rezero = "--rezero" in sys.argv
    targets = list(csv.DictReader(open(TARGETS, newline="", encoding="utf-8")))
    out = OUT.replace(".csv", "_recheck.csv") if recheck else OUT.replace(".csv", "_rezero.csv") if rezero else OUT
    if rezero:
        zero = {r["row"] for r in csv.DictReader(open(OUT, newline="", encoding="utf-8"))
                if "0" in (r["wb_site_mid"], r["wb_site_mid_oc"], r["wb_x_mid"], r["wb_x_mid_oc"])}
        targets = [t for t in targets if t["row"] in zero]
    if recheck:
        random.seed(2026)
        targets = random.sample(targets, min(recheck, len(targets)))
    done = set()
    if os.path.exists(out):
        done = {r["row"] for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    new = not os.path.exists(out)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        fails = 0
        for n, t in enumerate(targets, 1):
            if t["row"] in done:
                continue
            try:
                row = measure(t)
                fails = 0
            except Exception as exc:
                fails += 1
                print(f"[{n}/{len(targets)}] row {t['row']}: skipped for now ({exc})", flush=True)
                if fails >= 5:
                    print("five rows in a row failed; the archive may be down. Run the script again later.")
                    break
                continue
            w.writerow({c: row.get(c, "") for c in COLS})
            f.flush()
            print(f"[{n}/{len(targets)}] row {t['row']} {t['website_host'] or '-'}: site {row.get('wb_site_start', '')}/"
                  f"{row.get('wb_site_mid', '')}  x {row.get('wb_x_start', '')}/{row.get('wb_x_mid', '')}", flush=True)
    left = len(targets) - len(done) - sum(1 for _ in open(out, encoding="utf-8")) + 1 + len(done)
    print("finished this pass." + (f" {left} rows were skipped: run the script again to retry them." if left > 0 else " All rows done.") + " File:", out)


if __name__ == "__main__":
    main()
