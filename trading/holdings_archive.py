"""
Tax evidence for the German partial exemption (Teilfreistellung) of the 2x ETFs.

§ 20 Abs. 4 InvStG: if the investor proves that a fund actually held more than 50 % of its
assets in equities (Kapitalbeteiligungen) throughout the calendar year, the 30 % equity-fund
exemption applies on request in the tax assessment. SSO/QLD are US funds without a German
equity-fund status, so the proof has to come from their holdings.

Stores the issuer's official daily holdings file of every 2x ETF of the core strategy unchanged
(tax_evidence/holdings/<year>/<FUND>_<as-of>.csv) and appends its equity quota to
tax_evidence/equity_quota.csv. Quota = listed stocks / net assets (sum of all market values incl.
"Net Other Assets"); swaps/futures are exposure, not equity. REITs are counted as stocks here
although they are no Kapitalbeteiligungen (about 2-3 % of the S&P 500).

    python -m trading.holdings_archive
"""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from wf_backtest.core_strategy import CONFIG as CORE_CFG

log = logging.getLogger("holdings_archive")

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "tax_evidence"
QUOTA_LOG = EVIDENCE_DIR / "equity_quota.csv"
PUBLIC_LOG = ROOT / "docs" / "data" / "equity_quota.csv"   # copy shown on the dashboard
SOURCE_URL = "https://accounts.profunds.com/etfdata/ByFund/{fund}-psdlyhld.csv"
FUNDS = sorted({sl["etf_2x"] for sl in CORE_CFG["sleeves"]})
WARN_QUOTA = 0.55                 # legal limit is > 50 %; warn early
LOG_FIELDS = ["as_of", "fund", "equity_quota", "equities_usd", "net_assets_usd", "n_stocks",
              "other_securities_usd", "net_other_assets_usd", "derivatives_exposure_usd",
              "source", "file", "sha256", "fetched_utc"]
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/126 Safari/537.36",
                "Accept": "text/csv,*/*"}
REQUIRED_COLS = ["Fund Ticker", "Security Ticker", "Security Description", "Coupon",
                 "Maturity Date", "Exposure Value (Notional + G/L)", "Market Value"]


def download(fund: str, retries: int = 3) -> bytes:
    url = SOURCE_URL.format(fund=fund)
    last = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(Request(url, headers=HTTP_HEADERS), timeout=30) as resp:
                return resp.read()
        except Exception as e:  # network / server errors
            last = e
            log.warning(f"{fund}: download attempt {attempt}/{retries} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"{fund}: download failed: {last}")


def _num(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


def parse_holdings(raw: bytes, fund: str) -> dict:
    """Summary of an issuer holdings file; raises ValueError if the format is not as expected."""
    lines = raw.decode("utf-8-sig", errors="replace").splitlines()
    m = next((re.search(r"AS OF (\d{1,2}/\d{1,2}/\d{4})", ln) for ln in lines[:5] if "AS OF" in ln), None)
    if not m:
        raise ValueError("no 'AS OF' date in the file")
    as_of = datetime.strptime(m.group(1), "%m/%d/%Y").date()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Fund Ticker")), None)
    if start is None:
        raise ValueError("no header row in the file")
    rows = list(csv.reader(lines[start:]))
    col = {h.strip(): i for i, h in enumerate(rows[0])}
    missing = [c for c in REQUIRED_COLS if c not in col]
    if missing:
        raise ValueError(f"missing columns {missing}")

    equities = other = net_other = derivatives = 0.0
    n_stocks = 0
    for r in rows[1:]:
        f = {c: (r[col[c]].strip() if col[c] < len(r) else "") for c in REQUIRED_COLS}
        if f["Fund Ticker"] != fund:
            continue
        mv = _num(f["Market Value"])
        if f["Security Description"].startswith("Net Other Assets"):
            net_other += mv or 0.0
        elif mv is None:                                   # swaps / futures: exposure only
            derivatives += _num(f["Exposure Value (Notional + G/L)"]) or 0.0
        elif f["Security Ticker"] and not f["Coupon"] and not f["Maturity Date"]:
            equities += mv
            n_stocks += 1
        else:                                              # T-bills etc.
            other += mv
    net_assets = equities + other + net_other
    if n_stocks == 0 or net_assets <= 0:
        raise ValueError(f"implausible content: {n_stocks} stocks, net assets {net_assets:,.0f}")
    return {"as_of": as_of.isoformat(), "fund": fund,
            "equity_quota": round(equities / net_assets, 4),
            "equities_usd": round(equities, 2), "net_assets_usd": round(net_assets, 2),
            "n_stocks": n_stocks, "other_securities_usd": round(other, 2),
            "net_other_assets_usd": round(net_other, 2),
            "derivatives_exposure_usd": round(derivatives, 2),
            "source": SOURCE_URL.format(fund=fund)}


def _load_done() -> set:
    if not QUOTA_LOG.exists():
        return set()
    with QUOTA_LOG.open(newline="", encoding="utf-8") as fh:
        return {(r["as_of"], r["fund"]) for r in csv.DictReader(fh)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    done, new, failed = _load_done(), [], []
    for fund in FUNDS:
        try:
            raw = download(fund)
            s = parse_holdings(raw, fund)
        except Exception as e:
            log.error(f"{fund}: {e}")
            failed.append(fund)
            continue
        if (s["as_of"], fund) in done:
            log.info(f"{fund}: holdings as of {s['as_of']} already archived")
            continue
        path = EVIDENCE_DIR / "holdings" / s["as_of"][:4] / f"{fund}_{s['as_of']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)                              # original file, unchanged
        s.update(file=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                 fetched_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        new.append(s)
        log.info(f"{fund}: as of {s['as_of']} equity quota {s['equity_quota']:.1%} "
                 f"({s['n_stocks']} stocks) -> {s['file']}")
        if s["equity_quota"] <= WARN_QUOTA:
            msg = (f"{fund} equity quota {s['equity_quota']:.1%} as of {s['as_of']} "
                   f"(Teilfreistellung needs > 50 % all year)")
            log.warning(msg)
            if os.environ.get("GITHUB_ACTIONS"):
                print(f"::warning::{msg}")

    if new:
        write_header = not QUOTA_LOG.exists()
        QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
        with QUOTA_LOG.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(new)
    if QUOTA_LOG.exists():
        PUBLIC_LOG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(QUOTA_LOG, PUBLIC_LOG)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
