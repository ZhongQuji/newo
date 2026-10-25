#!/usr/bin/env python3
"""Fetch the full Chinese Wikisource text of 《大越史記全書》 (Đại Việt sử ký toàn thư),
the Vietnamese official chronicle written in Classical Chinese, and probe it for
celestial-event records (日食 / 月食 / 行星犯 / 月犯 / 太白晝見 ...).

Why: the project log (S33 续 6) lists 大越史記 as the only one of six global
sources not acquired ("网站动态加载、拿不到正文").  zh.wikisource.org serves the
30 卷 as static wikitext through the MediaWiki API, which is reachable here.

Outputs (next to this script):
    raw/<卷名>.wikitext      original wikitext per 卷
    text/<卷名>.txt          plain text (templates / links / tags stripped)
    dvsktt_all.txt           all 卷 concatenated with "## 卷名" headers
    MANIFEST.json            per-卷 revision id, timestamp, sha256, chars, keyword counts
    probe.md                 keyword table + example lines with eclipse records

Only standard-library modules.  Usage:  python fetch_dvsktt.py [--skip-fetch]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://zh.wikisource.org/w/api.php"
ROOT = "大越史記全書"
HERE = Path(__file__).resolve().parent
RAW, TEXT = HERE / "raw", HERE / "text"

SUBPAGES = (
    ["卷首"]
    + [f"外紀卷之{n}" for n in "一二三四五"]
    + [f"本紀卷之{n}" for n in "一 二 三 四 五 六 七 八 九 十 十一 十二 十三 十四 十五 十六 十七 十八 十九".split()]
    + [f"續編卷之{n}" for n in "一二三四五"]
)

# Keyword families used by the project's five celestial layers plus a few
# phrasings specific to Vietnamese chronicles (e.g. 日有食之 / 日蝕 / 月蝕).
KEYWORDS = {
    "SE 日食": r"日有食之|日食|日蝕|日有蝕",
    "LE 月食": r"月有食之|月食|月蝕",
    "PL 行星犯": r"(?:熒惑|太白|歲星|填星|鎮星|辰星|木星|火星|土星|金星|水星)[^。；，]{0,6}?(?:犯|入|守|合)",
    "MO 月犯": r"月犯|月入|月掩|月食(?:熒惑|太白|歲星|填星|鎮星|辰星)",
    "VD 太白晝見": r"太白晝見|太白經天|太白晝現",
    "彗/孛/流星": r"彗星|星孛|長星|流星|星隕",
}

GZ = r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]"
# Finer-grained counts that tell how many records are datable to the day,
# which is what the project's SE/LE oracles need.
DAYLEVEL = {
    "SE 干支+朔+日食（干支定日）": GZ + r"[，,]?\s*朔[，,]?\s*日(?:有)?[食蝕]",
    "SE 月+朔+日食（无干支，朔日可由历表定）": r"(?<![子丑寅卯辰巳午未申酉戌亥])月[，,]?\s*朔[，,]?\s*日(?:有)?[食蝕]",
    "SE 晦+日食（外紀转录汉代记录，用汉历）": r"晦[，,]?\s*日(?:有)?[食蝕]",
    "SE 既（全食）": r"日(?:有)?[食蝕][，,]?\s*既|日有食之既",
    "LE 望+月食": r"望[，,]?\s*月(?:有)?[食蝕]",
    "LE 十五/十六日+月食": r"十[五六]日[，、,]?\s*(?:[^，。]{0,6}[，、,]\s*)?月(?:有)?[食蝕]",
    "LE 干支+月食": GZ + r"[，,]?\s*(?:望[，,]?\s*)?月(?:有)?[食蝕]",
}


def api_get(params: dict, retries: int = 4) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "verichrono-dvsktt-fetch/1.0 (research)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"API failed: {url}: {last}")


def fetch_page(title: str) -> dict:
    """Return wikitext plus revision metadata for the latest revision."""
    d = api_get({"action": "query", "prop": "revisions", "titles": title,
                 "rvprop": "ids|timestamp|content", "rvslots": "main"})
    page = d["query"]["pages"][0]
    if page.get("missing"):
        return {"missing": True}
    rev = page["revisions"][0]
    return {"revid": rev["revid"], "timestamp": rev["timestamp"], "wikitext": rev["slots"]["main"]["content"]}


def strip_wikitext(wt: str) -> str:
    t = wt
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    # {{*|...}} is Wikisource's inline small-note template: keep the note text in brackets
    t = re.sub(r"\{\{\*\|([^{}]*)\}\}", r"〔\1〕", t)
    # drop remaining templates (header, PD, Textquality ...) possibly nested two levels
    for _ in range(3):
        t = re.sub(r"\{\{(?:[^{}]|\{[^{}]*\})*\}\}", "", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)           # [[a|b]] -> b
    t = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", t)         # [url text] -> text
    t = re.sub(r"-\{([^{}]*)\}-", lambda m: m.group(1).split(";")[0].split(":")[-1], t)  # -{zh-hant:x;zh-hans:y}-
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\n【\1】\n", t, flags=re.M)
    t = re.sub(r"^[\*#:;]+\s*", "", t, flags=re.M)
    t = re.sub(r"'''?", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true", help="reuse raw/ wikitext already on disk")
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    TEXT.mkdir(exist_ok=True)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"https://zh.wikisource.org/wiki/{urllib.parse.quote(ROOT)}",
        "license": "Underlying text: public domain (後黎朝 1479/1665/1697 official chronicle; page tagged {{PD-old}}). "
                   "Wikisource transcription/corrections: CC BY-SA 4.0 — cite zh.wikisource.org and keep revision ids.",
        "text_quality_tag": "Wikisource marks the work {{Textquality|50%}} (incomplete proofreading); treat as OCR-grade.",
        "pages": [],
    }
    all_parts, examples = [], []
    totals = {k: 0 for k in KEYWORDS}
    for sub in SUBPAGES:
        title = f"{ROOT}/{sub}"
        rawp = RAW / f"{sub}.wikitext"
        if args.skip_fetch and rawp.exists():
            wt = rawp.read_text(encoding="utf-8")
            meta = {"revid": None, "timestamp": None}
        else:
            r = fetch_page(title)
            if r.get("missing"):
                print(f"MISSING {title}")
                manifest["pages"].append({"卷": sub, "missing": True})
                continue
            wt, meta = r["wikitext"], {"revid": r["revid"], "timestamp": r["timestamp"]}
            rawp.write_text(wt, encoding="utf-8")
            time.sleep(0.5)  # be polite
        txt = strip_wikitext(wt)
        (TEXT / f"{sub}.txt").write_text(txt, encoding="utf-8")
        counts = {k: len(re.findall(p, txt)) for k, p in KEYWORDS.items()}
        for k, v in counts.items():
            totals[k] += v
        # collect example lines for the two eclipse layers
        for line in txt.splitlines():
            if re.search(KEYWORDS["SE 日食"], line) or re.search(KEYWORDS["LE 月食"], line):
                if len(examples) < 40:
                    examples.append((sub, line.strip()[:160]))
        entry = {"卷": sub, "title": title, **meta, "wikitext_chars": len(wt), "text_chars": len(txt),
                 "sha256_text": sha256_text(txt), "keywords": counts}
        manifest["pages"].append(entry)
        all_parts.append(f"## {sub}\n{txt}")
        print(f"{sub:<8} chars={len(txt):>7}  " + "  ".join(f"{k.split()[0]}={v}" for k, v in counts.items()))
    full = "\n".join(all_parts)
    (HERE / "dvsktt_all.txt").write_text(full, encoding="utf-8")
    daylevel = {k: len(re.findall(p, full)) for k, p in DAYLEVEL.items()}
    manifest["totals"] = {"text_chars": sum(p.get("text_chars", 0) for p in manifest["pages"]),
                          "keywords": totals, "daylevel": daylevel}
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 《大越史記全書》天象词频探针", "", f"生成（UTC）：{manifest['generated_utc']}", "",
             "| 卷 | 字数 | " + " | ".join(KEYWORDS) + " |", "|---|---:|" + "---:|" * len(KEYWORDS)]
    for p in manifest["pages"]:
        if p.get("missing"):
            lines.append(f"| {p['卷']} | 缺 | " + " | ".join("-" for _ in KEYWORDS) + " |")
        else:
            lines.append(f"| {p['卷']} | {p['text_chars']} | " + " | ".join(str(p['keywords'][k]) for k in KEYWORDS) + " |")
    lines.append(f"| **合计** | **{manifest['totals']['text_chars']}** | " + " | ".join(f"**{totals[k]}**" for k in KEYWORDS) + " |")
    lines += ["", "## 可定到日的记录数（全书）", "", "| 模式 | 条数 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in daylevel.items()]
    lines += ["", "## 日食 / 月食 记录示例（前 40 行）", ""]
    lines += [f"- 〔{s}〕{l}" for s, l in examples]
    anecdotes = [l.strip()[:220] for l in full.splitlines()
                 if re.search(r"裴時亨|不見食|救月|救日|密奏隐之", l)]
    lines += ["", "## 史料里现成的「核验者」片段（太史令误报 / 隐报月食，百官救月不见食）", ""]
    lines += [f"- {a}" for a in anecdotes[:8]]
    (HERE / "probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("TOTAL chars", manifest["totals"]["text_chars"], "keywords", totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
