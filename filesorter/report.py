"""Self-contained HTML report (no internet resources) showing where every file went."""

from __future__ import annotations

import html
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import APP_NAME, __version__
from .extract import KIND_FOLDERS
from .paths import REPORT_FILENAME, STATE_DIRNAME


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def build_data(state, dest: Path, stats: dict, models: dict, settings) -> dict:
    cats = {c["id"]: c for c in state.categories()}
    files = {f["id"]: f for f in state.files()}
    placements = state.placements()
    dirs: list[str] = []
    dir_index: dict[str, int] = {}

    def d(path: str) -> int:
        if path not in dir_index:
            dir_index[path] = len(dirs)
            dirs.append(path)
        return dir_index[path]

    kinds = list(KIND_FOLDERS)
    rows = []
    tree: dict = defaultdict(lambda: defaultdict(lambda: {"n": 0, "bytes": 0, "dups": 0}))
    totals = defaultdict(int)
    for p in placements:
        f = files.get(p["file_id"])
        if f is None:
            continue
        src_dir = str(Path(f["path"]).parent)
        dest_rel = p["dest"] or ""
        dest_dir = str(Path(dest_rel).parent).replace("\\", "/") if dest_rel else ""
        dest_name = Path(dest_rel).name if dest_rel else ""
        role = p["role"]
        status = p["status"] or "pending"
        cat = cats.get(p["category_id"]) if p["category_id"] else None
        if cat:
            area, topic = cat["area"], cat["name"]
        else:
            area, topic = "_Unsorted", KIND_FOLDERS.get(f["kind"], "Other files")
        node = tree[area][topic]
        if role == "duplicate":
            node["dups"] += 1
        else:
            node["n"] += 1
        node["bytes"] += f["size"] or 0
        totals[role] += 1
        totals["status:" + status] += 1
        rows.append([
            f["name"], d(src_dir), d(dest_dir), dest_name, role, status, f["size"] or 0,
            kinds.index(f["kind"]) if f["kind"] in kinds else len(kinds) - 1,
            p["reason"] or "", round(p["confidence"] or 0, 2), f["path"],
        ])
    tree_out = []
    for area in sorted(tree, key=lambda a: (a.startswith("_"), a.lower())):
        topics = [{"name": t, **v, "keywords": _keywords(cats, area, t)} for t, v in sorted(
            tree[area].items(), key=lambda kv: kv[0].lower())]
        tree_out.append({"name": area, "topics": topics, "n": sum(t["n"] for t in topics),
                         "dups": sum(t["dups"] for t in topics), "bytes": sum(t["bytes"] for t in topics)})
    return {
        "app": f"{APP_NAME} {__version__}",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "destination": str(dest),
        "sources": list(settings.sources),
        "stats": stats,
        "models": models,
        "totals": dict(totals),
        "tree": tree_out,
        "dirs": dirs,
        "kinds": [KIND_FOLDERS[k] for k in kinds],
        "files": rows,
        "destUrl": dest.resolve().as_uri(),
    }


def _keywords(cats, area, topic) -> list[str]:
    for c in cats.values():
        if c["area"] == area and c["name"] == topic:
            try:
                return json.loads(c["keywords"] or "[]")[:6]
            except ValueError:
                return []
    return []


def write_report(state, dest: Path, stats: dict, models: dict, settings) -> Path:
    data = build_data(state, dest, stats, models, settings)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__DATA__", payload).replace("__TITLE__", html.escape(f"Sorting report – {dest.name}"))
    out = dest / REPORT_FILENAME
    tmp = out.with_suffix(".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out)
    history = dest / STATE_DIRNAME / "reports"
    history.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, history / f"report-{time.strftime('%Y%m%d-%H%M%S')}.html")
    old = sorted(history.glob("report-*.html"))
    for f in old[:-20]:  # keep the 20 latest
        f.unlink(missing_ok=True)
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#5d6779;--line:#e3e6ec;--accent:#2f6fde;--accent-soft:#e8f0fd;
--ok:#1f8a5b;--warn:#b76e00;--dup:#7a4fd6;--bad:#c2410c;--bar:#d6e2fa}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--panel:#1c2027;--ink:#e6e9ef;--muted:#9aa3b2;--line:#2c323c;
--accent:#79a6ff;--accent-soft:#22304a;--ok:#4cc38a;--warn:#f0b54a;--dup:#b39cff;--bad:#ff8a5c;--bar:#2d3c5c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif}
header{padding:22px 28px 8px}h1{margin:0 0 4px;font-size:22px;font-weight:600}.sub{color:var(--muted)}
.sub a{color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;padding:12px 28px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}.card span{color:var(--muted);font-size:12.5px}
main{display:grid;grid-template-columns:minmax(260px,360px) 1fr;gap:16px;padding:8px 28px 28px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
.tree{max-height:calc(100vh - 120px);overflow:auto;padding:6px 0}
.area{border-bottom:1px solid var(--line)}.area:last-child{border-bottom:0}
.arow,.trow{display:flex;align-items:center;gap:8px;padding:6px 14px;cursor:pointer;user-select:none}
.arow{font-weight:600}.arow:hover,.trow:hover{background:var(--accent-soft)}
.trow{padding-left:30px;font-size:13.5px}.trow.sel,.arow.sel{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
.caret{width:12px;color:var(--muted);transition:transform .15s}.area.open .caret{transform:rotate(90deg)}
.area .topics{display:none}.area.open .topics{display:block}
.nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cnt{color:var(--muted);font-variant-numeric:tabular-nums;font-size:12.5px}
.kw{padding:0 14px 6px 30px;color:var(--muted);font-size:12px;margin-top:-4px}
.bar{height:4px;background:var(--bar);border-radius:2px;margin:0 14px 6px 30px}
.bar i{display:block;height:100%;background:var(--accent);border-radius:2px}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line)}
input[type=search]{flex:1;min-width:180px;padding:7px 10px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);font:inherit}
.chip{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:999px;padding:4px 11px;cursor:pointer;font:inherit;font-size:12.5px}
.chip.on{background:var(--accent);border-color:var(--accent);color:#fff}
button.link{border:0;background:none;color:var(--accent);cursor:pointer;font:inherit;padding:0}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel);cursor:pointer;white-space:nowrap}
td{font-size:13px}td.size{white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--muted)}
.path{color:var(--muted);font-size:12px;word-break:break-all}.fname{font-weight:500;word-break:break-word}
.pill{display:inline-block;border-radius:999px;padding:1px 8px;font-size:11.5px;font-weight:600;white-space:nowrap}
.p-sorted{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.p-duplicate{background:color-mix(in srgb,var(--dup) 15%,transparent);color:var(--dup)}
.p-unsorted{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.p-bad{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.tablewrap{max-height:calc(100vh - 170px);overflow:auto}
.pager{display:flex;gap:10px;align-items:center;justify-content:flex-end;padding:10px 14px;color:var(--muted)}
.pager button{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:6px;padding:4px 10px;cursor:pointer}
.why{color:var(--muted);font-size:12px}
.foot{padding:0 28px 28px;color:var(--muted);font-size:12.5px}.foot code{background:var(--panel);border:1px solid var(--line);padding:1px 5px;border-radius:4px}
a.open{color:var(--accent);text-decoration:none}a.open:hover{text-decoration:underline}
</style></head><body>
<header><h1>Sorting report</h1><div class="sub" id="sub"></div></header>
<section class="cards" id="cards"></section>
<main>
  <section class="panel"><h2>Folders created</h2><div class="tree" id="tree"></div></section>
  <section class="panel">
    <div class="tools">
      <input type="search" id="q" placeholder="Search file names, folders, reasons…">
      <button class="chip on" data-f="all">All</button><button class="chip" data-f="sorted">Sorted</button>
      <button class="chip" data-f="duplicate">Duplicates</button><button class="chip" data-f="unsorted">Unsorted</button>
      <button class="chip" data-f="problem">Problems</button>
      <button class="link" id="csv">Export CSV</button>
    </div>
    <div class="tablewrap"><table><thead><tr>
      <th data-k="0">File</th><th data-k="2">Copied to</th><th data-k="1">From</th><th data-k="6">Size</th><th data-k="4">Status</th>
    </tr></thead><tbody id="rows"></tbody></table></div>
    <div class="pager"><span id="pinfo"></span><button id="prev">‹ Prev</button><button id="next">Next ›</button></div>
  </section>
</main>
<div class="foot" id="foot"></div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const hb=n=>{const u=["B","KB","MB","GB","TB"];let i=0;while(n>=1024&&i<4){n/=1024;i++}return (i?n.toFixed(1):n)+" "+u[i]};
const S=D.stats||{},T=D.totals||{};
$("#sub").innerHTML=`${esc(D.generated)} · destination <a href="${esc(D.destUrl)}">${esc(D.destination)}</a> · sources: ${D.sources.map(esc).join(", ")}`;
const card=(v,l)=>`<div class="card"><b>${v}</b><span>${l}</span></div>`;
const nTopics=D.tree.filter(a=>!a.name.startsWith("_")).reduce((s,a)=>s+a.topics.length,0);
$("#cards").innerHTML=[card((T.sorted||0).toLocaleString(),"files sorted"),card(nTopics,"folders created"),
 card((T.duplicate||0).toLocaleString(),"duplicates (in “duplicate” folders)"),card((T.unsorted||0).toLocaleString(),"could not be sorted (in _Unsorted)"),
 card(hb(S["bytes copied"]||0),"copied this run"),card(((S.seconds||0)/60).toFixed(1)+" min","run time")].join("");
const m=D.models||{};
const skipped=Object.entries(S.skipped||{}).map(([k,v])=>`${v.toLocaleString()} ${esc(k)}`).join(", ");
$("#foot").innerHTML=`Analysed on this computer only — network access was ${m.offline?"<b>blocked</b>":"not blocked"}. `+
 `Text model: <code>${esc(m.text_model||"?")}</code> · Image model: <code>${esc(m.image_model||"?")}</code> · `+
 `OCR: <code>${m.ocr?"on":"off"}</code> · Folder naming: <code>${esc(m.naming_model||"keywords")}</code>`+
 (skipped?`<br>Not looked at: ${skipped}.`:"")+(m.problems&&m.problems.length?`<br>Notes: ${m.problems.map(esc).join("; ")}`:"")+
 `<br>Original files were not changed: everything was copied. ${esc(D.app)}`;

// ---- tree
let filterDir=null,filterArea=null;
const maxBytes=Math.max(1,...D.tree.flatMap(a=>a.topics.map(t=>t.n+t.dups)));
$("#tree").innerHTML=D.tree.map((a,ai)=>`<div class="area${ai<3?" open":""}" data-a="${ai}">
 <div class="arow" data-area="${esc(a.name)}"><span class="caret">▸</span><span class="nm">${esc(a.name)}</span><span class="cnt">${a.n.toLocaleString()}${a.dups?` +${a.dups} dup`:""}</span></div>
 <div class="topics">${a.topics.map(t=>`<div class="trow" data-dir="${esc(a.name+"/"+t.name)}"><span class="nm" title="${esc(t.name)}">${esc(t.name)}</span><span class="cnt">${t.n.toLocaleString()}${t.dups?` +${t.dups}`:""}</span></div>
 <div class="bar"><i style="width:${Math.max(2,100*(t.n+t.dups)/maxBytes)}%"></i></div>${t.keywords&&t.keywords.length?`<div class="kw">${t.keywords.map(esc).join(" · ")}</div>`:""}`).join("")}</div></div>`).join("");
document.querySelectorAll(".arow").forEach(el=>el.addEventListener("click",e=>{
 const area=el.parentElement;const wasSel=el.classList.contains("sel");
 if(e.target.classList.contains("caret")){area.classList.toggle("open");return}
 area.classList.add("open");clearSel();if(!wasSel){el.classList.add("sel");filterArea=el.dataset.area}apply()}));
document.querySelectorAll(".trow").forEach(el=>el.addEventListener("click",()=>{const was=el.classList.contains("sel");clearSel();
 if(!was){el.classList.add("sel");filterDir=el.dataset.dir}apply()}));
function clearSel(){document.querySelectorAll(".sel").forEach(x=>x.classList.remove("sel"));filterDir=null;filterArea=null}

// ---- table
let status="all",q="",page=0,sortK=null,sortDir=1,view=[];const PER=150;
const prob=r=>r[5]==="failed"||r[5]==="no space"||r[8].startsWith("could not be read");
function topDir(r){const d=D.dirs[r[2]]||"";return d}
function apply(){const qq=q.toLowerCase();
 view=D.files.filter(r=>{
  if(status==="problem"){if(!prob(r))return false}else if(status!=="all"&&r[4]!==status)return false;
  const dd=topDir(r);
  if(filterDir&&!(dd===filterDir||dd===filterDir+"/duplicate"))return false;
  if(filterArea&&!(dd===filterArea||dd.startsWith(filterArea+"/")))return false;
  if(qq&&!(r[0].toLowerCase().includes(qq)||dd.toLowerCase().includes(qq)||(D.dirs[r[1]]||"").toLowerCase().includes(qq)||r[8].toLowerCase().includes(qq)))return false;
  return true});
 if(sortK!==null)view.sort((a,b)=>{const x=sortK===1||sortK===2?D.dirs[a[sortK]]:a[sortK],y=sortK===1||sortK===2?D.dirs[b[sortK]]:b[sortK];return (x>y?1:x<y?-1:0)*sortDir});
 page=0;render()}
function render(){const s=page*PER,rows=view.slice(s,s+PER);
 $("#rows").innerHTML=rows.map(r=>{const dd=D.dirs[r[2]]||"";const link=dd?`${D.destUrl}/${dd.split("/").map(encodeURIComponent).join("/")}`:"";
  const pill=prob(r)?`<span class="pill p-bad">${r[5]==="no space"?"not copied: disk full":r[5]==="failed"?"copy failed":"unreadable"}</span>`:`<span class="pill p-${r[4]}">${r[4]}</span>`;
  return `<tr><td><div class="fname">${esc(r[0])}</div><div class="why">${esc(r[8])}</div></td>
  <td>${dd?`<a class="open" href="${esc(link)}">${esc(dd)}</a>`:"<span class='why'>(not copied yet)</span>"}${r[3]&&r[3]!==r[0]?`<div class="why">as ${esc(r[3])}</div>`:""}</td>
  <td class="path">${esc(D.dirs[r[1]]||"")}</td><td class="size">${hb(r[6])}</td><td>${pill}</td></tr>`}).join("")||`<tr><td colspan="5" class="why">No files match.</td></tr>`;
 const pages=Math.max(1,Math.ceil(view.length/PER));$("#pinfo").textContent=`${view.length.toLocaleString()} files · page ${page+1} of ${pages}`;
 $("#prev").disabled=page===0;$("#next").disabled=page>=pages-1}
$("#q").addEventListener("input",e=>{q=e.target.value;apply()});
document.querySelectorAll(".chip").forEach(c=>c.addEventListener("click",()=>{document.querySelectorAll(".chip").forEach(x=>x.classList.remove("on"));c.classList.add("on");status=c.dataset.f;apply()}));
document.querySelectorAll("th").forEach(th=>th.addEventListener("click",()=>{const k=+th.dataset.k;sortDir=sortK===k?-sortDir:1;sortK=k;apply()}));
$("#prev").onclick=()=>{page--;render()};$("#next").onclick=()=>{page++;render()};
$("#csv").onclick=()=>{const head=["file","copied_to","from","size_bytes","status","reason","source_path"];
 const lines=[head.join(",")].concat(view.map(r=>[r[0],(D.dirs[r[2]]||"")+"/"+(r[3]||""),D.dirs[r[1]],r[6],r[4],r[8],r[10]].map(v=>`"${String(v).replace(/"/g,'""')}"`).join(",")));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["﻿"+lines.join("\r\n")],{type:"text/csv"}));a.download="sorting-report.csv";a.click()};
apply();
</script></body></html>
"""
