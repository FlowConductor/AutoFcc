"""AutoFcc Gorev Paneli — istek/zincir analizi icin ayri UI.

Ne gosterir:
- Son gorev(ler): tamamlandi mi, hangi modeller kullanildi, bekleme oldu mu
- Her istegin zincir yuruyusu: hangi model ne ile dustu (403/402/429/404),
  hangi model cevap verdi, kac aday atlandi
- Kota durumu: saglayici bazli bugunku kullanim, limitler, ogrenilen RPM

Kullanim:  python tools/dashboard.py   ->  http://localhost:8890
Veri: ~/.fcc/logs/server.log + provider_usage.json (sadece okunur)
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

FCC_DIR = Path.home() / ".fcc"
LOG_PATH = FCC_DIR / "logs" / "server.log"
USAGE_PATH = FCC_DIR / "provider_usage.json"
LIMITS_PATH = Path(__file__).resolve().parent.parent / (
    "src/free_claude_code/core/data/provider_limits.json"
)

HOST_TO_PROVIDER = {
    "api.groq.com": "groq",
    "api.cerebras.ai": "cerebras",
    "openrouter.ai": "open_router",
    "api.cohere.ai": "cohere",
    "api.cloudflare.com": "cloudflare",
    "api.llm7.io": "llm7",
    "api.z.ai": "zai_api",
    "integrate.api.nvidia.com": "nvidia_nim",
    "router.huggingface.co": "huggingface",
    "api.kilo.ai": "kilo",
    "api.mistral.ai": "mistral",
    "generativelanguage.googleapis.com": "gemini",
    "oai.endpoints.kepler.ai.cloud.ovh.net": "ovhcloud",
}

RE_FALLBACK = re.compile(
    r"Model fallback: request_id=(\S+) from=(\S+) to=(\S+) "
    r"candidate=(\S+) failure_kind=(\S+) status_code=(\d+)"
)
RE_ERROR = re.compile(
    r"(\w+)_ERROR: request_id=(\S+) exc_type=\S+ http_status=(\d+)"
)
RE_POST = re.compile(r"HTTP Request: POST (\S+) \"HTTP/1\.1 (\d+)")
RE_QUOTA_FILTER = re.compile(r"Provider kota filtresi: \[([^\]]*)\] saglayici")
RE_CAND_SKIP = re.compile(
    r"Candidate atlandi \(provider kota\): provider=(\S+) model=(\S+) reason=(\S+)"
)
RE_RPM_WINDOW = re.compile(r"Provider RPM penceresi doldu: provider=(\S+)")
RE_MODEL_404 = re.compile(r"Model bulunamadi \(404\): provider=(\S+) model=(\S+)")


def parse_log() -> dict:
    requests: dict[str, dict] = {}
    info_events: list[dict] = []

    def req(rid: str, ts: float) -> dict:
        entry = requests.setdefault(
            rid,
            {
                "id": rid,
                "start": ts,
                "end": ts,
                "steps": [],
                "skips": 0,
                "skipped": [],
                "success": False,
                "final_model": None,
                "final_provider": None,
            },
        )
        entry["end"] = max(entry["end"], ts)
        return entry

    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        lines = []

    for line in lines:
        try:
            j = json.loads(line)
            rec = j.get("record", {})
        except Exception:
            continue
        msg = rec.get("message", "")
        ts = rec.get("time", {}).get("timestamp") or 0
        extra = rec.get("extra", {}) or {}
        rid = extra.get("request_id")

        fallback = RE_FALLBACK.search(msg)
        if fallback:
            r = req(fallback.group(1), ts)
            r["steps"].append(
                {
                    "t": ts,
                    "model": fallback.group(2),
                    "outcome": f"{fallback.group(5)}:{fallback.group(6)}",
                }
            )
            r["pending_to"] = fallback.group(3)
            continue

        error = RE_ERROR.search(msg)
        if error:
            r = req(error.group(2), ts)
            if r.get("pending_to"):
                r["steps"].append(
                    {"t": ts, "model": r["pending_to"], "outcome": f"err:{error.group(3)}"}
                )
                r["pending_to"] = None
            continue

        post = RE_POST.search(msg)
        if post:
            provider = HOST_TO_PROVIDER.get(urlparse(post.group(1)).hostname, "?")
            code = post.group(2)
            if rid:
                r = req(rid, ts)
                if code.startswith("2"):
                    if r.get("pending_to"):
                        r["steps"].append(
                            {"t": ts, "model": r["pending_to"], "outcome": "served"}
                        )
                        r["final_model"] = r["pending_to"]
                        r["pending_to"] = None
                    elif not r["final_model"]:
                        r["final_model"] = f"{provider}/(birincil)"
                    r["final_provider"] = provider
                    r["success"] = True
                elif r.get("pending_to"):
                    r["steps"].append(
                        {"t": ts, "model": r["pending_to"], "outcome": f"http:{code}"}
                    )
                    r["pending_to"] = None
            continue

        if RE_QUOTA_FILTER.search(msg):
            if rid:
                r = req(rid, ts)
                r["skips"] += (
                    msg.count(":") and len(re.findall(r"'([a-z_0-9]+)':", msg))
                ) or 1
            continue

        cand = RE_CAND_SKIP.search(msg)
        if cand and rid:
            r = req(rid, ts)
            r["skips"] += 1
            r["skipped"].append(f"{cand.group(1)}/{cand.group(2)} ({cand.group(3)})")
            continue

        for pattern, kind in ((RE_RPM_WINDOW, "rpm"), (RE_MODEL_404, "404")):
            m2 = pattern.search(msg)
            if m2:
                info_events.append(
                    {"t": ts, "kind": kind, "provider": m2.group(1), "msg": msg[:120]}
                )
                break

    for r in requests.values():
        r["duration_s"] = round(max(0.0, r["end"] - r["start"]), 1)
        r["walk"] = len([s for s in r["steps"] if s["outcome"] != "served"])
        r.pop("pending_to", None)

    ordered = sorted(requests.values(), key=lambda r: r["start"])

    # Gorev (session) kumeleme: 3 dk ustu bosluk = yeni gorev
    tasks: list[dict] = []
    for r in ordered:
        if tasks and r["start"] - tasks[-1]["end"] <= 180:
            t = tasks[-1]
            t["end"] = max(t["end"], r["end"])
            t["requests"].append(r["id"])
            t["success_count"] += 1 if r["success"] else 0
            t["walk"] += r["walk"]
            t["skips"] += r["skips"]
        else:
            tasks.append(
                {
                    "start": r["start"],
                    "end": r["end"],
                    "requests": [r["id"]],
                    "success_count": 1 if r["success"] else 0,
                    "walk": r["walk"],
                    "skips": r["skips"],
                }
            )
    for t in tasks:
        t["duration_s"] = round(max(0.0, t["end"] - t["start"]), 1)
        t["total"] = len(t["requests"])
        t["done"] = t["success_count"] == t["total"]

    return {
        "requests": {r["id"]: r for r in ordered[-120:]},
        "tasks": tasks[-12:],
        "info_events": info_events[-30:],
        "generated_at": time.time(),
    }


def load_quota() -> dict:
    usage = {}
    with contextlib.suppress(Exception):
        usage = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    limits = {}
    try:
        raw = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
        limits = raw.get("providers", {})
    except Exception:
        pass
    return {"usage": usage, "limits": limits}


def build_payload() -> dict:
    data = parse_log()
    reqs = list(data["requests"].values())
    today = time.strftime("%Y-%m-%d")
    today_reqs = [r for r in reqs if time.strftime("%Y-%m-%d", time.localtime(r["start"])) == today]
    served = [r for r in today_reqs if r["success"]]
    model_counts: dict[str, int] = {}
    for r in served:
        model_counts[r["final_model"] or "?"] = model_counts.get(r["final_model"] or "?", 0) + 1
    durations = [r["duration_s"] for r in today_reqs if r["duration_s"] > 0]
    summary = {
        "today_requests": len(today_reqs),
        "today_success": len(served),
        "success_rate": round(100 * len(served) / len(today_reqs), 1) if today_reqs else None,
        "avg_duration": round(sum(durations) / len(durations), 1) if durations else None,
        "total_walk": sum(r["walk"] for r in today_reqs),
        "total_skips": sum(r["skips"] for r in today_reqs),
        "top_models": sorted(model_counts.items(), key=lambda kv: -kv[1])[:8],
    }
    return {
        "summary": summary,
        "tasks": data["tasks"],
        "requests": data["requests"],
        "info_events": data["info_events"],
        "quota": load_quota(),
        "generated_at": data["generated_at"],
    }


HTML = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>AutoFcc Gorev Paneli</title>
<style>
:root{--bg:#0d1117;--panel:#161b23;--border:#2a2f3a;--ok:#3fb950;--err:#f85149;--warn:#d29922;--skip:#8957e5;--txt:#e6edf3;--dim:#8b949e}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.45 system-ui,sans-serif;padding:20px}
h1{font-size:19px;margin-bottom:4px} .sub{color:var(--dim);font-size:12px;margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px}
.card .v{font-size:21px;font-weight:700} .card .l{color:var(--dim);font-size:11px;margin-top:2px}
h2{font-size:15px;margin:18px 0 8px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase}
.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;margin:1px;border:1px solid var(--border);white-space:nowrap}
.ok{color:var(--ok);border-color:var(--ok)} .err{color:var(--err);border-color:var(--err)}
.warn{color:var(--warn);border-color:var(--warn)} .skip{color:var(--skip);border-color:var(--skip)}
.dim{color:var(--dim)} tr.clickable{cursor:pointer} tr.clickable:hover{background:#1c2128}
.hidden{display:none} .detail{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px;margin:8px 0}
.badge-ok{color:var(--ok);font-weight:700} .badge-err{color:var(--err);font-weight:700}
#refresh{position:fixed;top:14px;right:16px;color:var(--dim);font-size:11px}
</style></head><body>
<div id="refresh">her 5 sn yenilenir</div>
<h1>🎯 AutoFcc Gorev Paneli</h1>
<div class="sub">son gorev: tamamlandi mi · hangi modeller kullanildi · bekleme oldu mu</div>
<div class="cards" id="cards"></div>
<h2>Gorevler</h2>
<table><thead><tr><th>Baslangic</th><th>Sure</th><th>Istek</th><th>Basarili</th><th>Zincir yuruyusu</th><th>Atlanan</th><th>Durum</th></tr></thead><tbody id="tasks"></tbody></table>
<h2>Son Istekler</h2>
<table><thead><tr><th>Saat</th><th>Sure</th><th>Sonuc</th><th>Kullanilan modeller (zincir)</th><th>Detay</th></tr></thead><tbody id="reqs"></tbody></table>
<div id="details"></div>
<h2>Kota Durumu (bugun)</h2>
<table><thead><tr><th>Saglayici</th><th>Istek</th><th>Basari</th><th>Hata</th><th>Limit</th><th>Ogrenilen RPM</th><th>Durum</th></tr></thead><tbody id="quota"></tbody></table>
<h2>Sistem Olaylari</h2>
<div id="events" class="detail dim" style="font-size:12px"></div>
<script>
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = ts => new Date(ts*1000).toLocaleTimeString('tr-TR');
const chip = (txt, cls) => `<span class="chip ${cls}">${esc(txt)}</span>`;

function outcomeChip(o){
  if (o === 'served') return chip('✓ ' + 'cevap', 'ok');
  if (o.startsWith('permission:402')) return chip('402 kota', 'err');
  if (o.startsWith('authentication') || o.includes(':403')) return chip('403 erisim', 'err');
  if (o.includes('429')) return chip('429 limit', 'warn');
  if (o.includes('413')) return chip('413 cok buyuk', 'warn');
  if (o.includes('404')) return chip('404 olu', 'err');
  if (o.includes('400')) return chip('400', 'err');
  return chip(o, 'err');
}

let statik = (typeof GOMULU_VERI !== 'undefined') ? GOMULU_VERI : null;
async function tick(){
  let d;
  if (statik) { d = statik; }
  else {
    try { d = await (await fetch('/api/data')).json(); }
    catch (e) {
      document.getElementById('refresh').textContent = 'veri alinamadi, 5 sn sonra tekrar deneniyor...';
      return;
    }
  }
  document.getElementById('refresh').textContent = statik ? 'statik rapor (canli degil)' : 'her 5 sn yenilenir';
  const s = d.summary;
  document.getElementById('cards').innerHTML = `
    <div class="card"><div class="v">${s.today_requests}</div><div class="l">bugunku istek</div></div>
    <div class="card"><div class="v">${s.success_rate ?? '—'}%</div><div class="l">basari orani</div></div>
    <div class="card"><div class="v">${s.avg_duration ?? '—'}s</div><div class="l">ort. istek suresi</div></div>
    <div class="card"><div class="v">${s.total_walk}</div><div class="l">zincir hata adimi</div></div>
    <div class="card"><div class="v">${s.total_skips}</div><div class="l">atlanan aday</div></div>
    <div class="card"><div class="v">${esc((s.top_models[0]||['—'])[0])}</div><div class="l">en cok cevap veren</div></div>`;

  document.getElementById('tasks').innerHTML = d.tasks.slice(-8).reverse().map(t => `
    <tr><td>${fmt(t.start)}</td><td>${t.duration_s}s</td><td>${t.total}</td>
    <td>${t.success_count}/${t.total}</td><td>${t.walk}</td><td>${t.skips}</td>
    <td>${t.done ? '<span class="badge-ok">✓ tamamlandi</span>' : '<span class="badge-err">✗ eksik</span>'}</td></tr>`).join('');

  document.getElementById('reqs').innerHTML = Object.values(d.requests).slice(-40).reverse().map(r => {
    const chain = r.steps.slice(0, 10).map(st => `${esc(st.model.split('/').slice(0,2).join('/'))} ${outcomeChip(st.outcome)}`).join(' ');
    return `<tr class="clickable" onclick="toggle('${r.id}')">
      <td>${fmt(r.start)}</td><td>${r.duration_s}s</td>
      <td>${r.success ? '<span class="badge-ok">✓</span>' : '<span class="badge-err">✗</span>'}</td>
      <td style="max-width:480px">${chain || '<span class="dim">tek aday</span>'}</td>
      <td class="dim">${r.skips} atla</td></tr>
    <tr class="hidden" id="d-${r.id}"><td colspan="5">
      <div class="detail"><b>Zincir:</b><br>${r.steps.map(st =>
        `${fmt(st.t)} ${esc(st.model)} → ${outcomeChip(st.outcome)}`).join('<br>') || 'tek aday, fallback yok'}
        ${r.final_model ? `<br><b>Cevap veren:</b> <span class="badge-ok">${esc(r.final_model)}</span>` : ''}
        ${r.skipped.length ? `<br><b>Atlanan (${r.skipped}):</b> ${r.skipped.slice(0,12).map(s2=>chip(s2,'skip')).join(' ')}` : ''}
      </div></td></tr>`;
  }).join('');

  const q = d.quota, rows = [];
  const all = new Set([...Object.keys(q.usage||{}), ...Object.keys(q.limits||{})]);
  for (const p of [...all].sort()) {
    const u = (q.usage||{})[p]||{}, lim = (q.limits||{})[p]||{};
    const reason = u.exhausted_reason ? ({'quota_402':'402 kota bitti','access_denied:403':'403 erisim yok','access_denied:401':'401 erisim yok'}[u.exhausted_reason]||u.exhausted_reason)
      : (u.learned_rpm ? `yerel olcum (RPM=${u.learned_rpm})` : '✓ acik');
    const limTxt = [lim.rpm?lim.rpm+' RPM':'', lim.rpd?lim.rpd+' RPD':''].filter(Boolean).join(' / ') || '—';
    rows.push(`<tr><td>${esc(p)}</td><td>${u.attempts??0}</td><td>${u.successes??0}</td><td>${u.failures??0}</td>
      <td class="dim">${limTxt}</td><td>${u.learned_rpm??'—'}</td>
      <td>${u.exhausted_reason?'<span class="badge-err">'+esc(reason)+'</span>':(u.learned_rpm?'<span style="color:var(--warn)">'+esc(reason)+'</span>':'<span class="badge-ok">'+reason+'</span>')}</td></tr>`);
  }
  document.getElementById('quota').innerHTML = rows.join('') || '<tr><td colspan="7" class="dim">henuz veri yok</td></tr>';

  document.getElementById('events').innerHTML = d.info_events.slice().reverse().map(e =>
    `${fmt(e.t)} [${e.kind}] ${esc(e.msg)}`).join('<br>') || 'olay yok';
}
function toggle(id){ document.getElementById('d-'+id).classList.toggle('hidden'); }
tick(); setInterval(tick, 5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            body = json.dumps(build_payload(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/" or self.path.startswith("/?"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def build_static_html() -> str:
    """Verisi HTML'in icine gomulu, ag gerektirmeyen rapor (file:// ile acilir)."""
    payload = json.dumps(build_payload(), ensure_ascii=False)
    inject = "const GOMULU_VERI = " + payload + ";"
    return HTML.replace(
        "let statik = (typeof GOMULU_VERI", inject + "\nlet statik = (typeof GOMULU_VERI"
    )


def main() -> None:
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--static":
        out = Path(sys.argv[2])
        out.write_text(build_static_html(), encoding="utf-8")
        print(f"Statik rapor -> {out}")
        return
    server = ThreadingHTTPServer(("127.0.0.1", 8890), Handler)
    print("AutoFcc Gorev Paneli -> http://localhost:8890")
    server.serve_forever()


if __name__ == "__main__":
    main()
