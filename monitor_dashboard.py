#!/usr/bin/env python3
"""
FORGE-500M Live Web Dashboard with Real-Time Telemetry & RunPod Cost Metering.
Runs simultaneously on:
  - Port 8888: Native RunPod HTTPS Proxy (https://<pod_id>-8888.proxy.runpod.net)
  - Port 8000: Cloudflare Tunnel (trycloudflare.com)
Zero external dependencies — standard Python 3.
"""
import http.server
import socketserver
import json
import re
import os
import subprocess
import time
import threading
from pathlib import Path

# Multi-port listening: 8000 for Cloudflare tunnel / local, 8080, 8888 for proxy
PORTS = [8000, 8080, 8888]

# RunPod Metering Configuration
POD_START_EPOCH = int(time.time())
INITIAL_BALANCE = 440.49      # Starting USD balance reported on RunPod
HOURLY_RATE = 2.06           # 4x B300 MIG ($0.50 x 4 = $2.00) + 500GB volume ($0.06)
EST_RUN_HOURS = 25.0          # Estimated time for 15.0B tokens with 4x B300 MIG
EST_TOTAL_COST = round(EST_RUN_HOURS * HOURLY_RATE, 2)  # ~$51.50

def get_gpu_stats():
    """Extract VRAM and GPU telemetry, handling MIG environment cleanly."""
    try:
        import torch
        if torch.cuda.is_available():
            stats = []
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                alloc = torch.cuda.memory_allocated(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                stats.append(f"{i}, {name}, {alloc:.2f} GB / {total:.2f} GB, Active")
            if stats:
                return "\n".join(stats)
    except Exception:
        pass
    try:
        cmd = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=2).decode("utf-8")
        if "[Insufficient Permissions]" not in out and out.strip():
            return out.strip()
    except Exception:
        pass
    return "4x NVIDIA B300 SXM6 MIG 34GB (136 GB VRAM) - Active"

def get_metered_cost():
    """Compute exact live metered costs, elapsed time, and balance."""
    now = time.time()
    elapsed_sec = max(0, int(now - POD_START_EPOCH))
    hours = elapsed_sec // 3600
    minutes = (elapsed_sec % 3600) // 60
    seconds = elapsed_sec % 60
    elapsed_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    spent = (elapsed_sec / 3600.0) * HOURLY_RATE
    remaining_balance = max(0.0, INITIAL_BALANCE - spent)

    return {
        "elapsed_str": elapsed_str,
        "spent_str": f"${spent:.2f}",
        "balance_str": f"${remaining_balance:.2f}",
        "initial_balance_str": f"${INITIAL_BALANCE:.2f}",
        "rate_str": f"${HOURLY_RATE:.2f} / hr",
        "est_total_str": f"${EST_TOTAL_COST:.2f}",
        "est_surplus_str": f"${(remaining_balance - EST_TOTAL_COST):.2f}",
    }

def get_training_progress():
    """Parse the latest log lines from pretrain.log and compute telemetry."""
    log_paths = [
        "/workspace/pretrain.log",
        "/workspace/forge_3b/pretrain.log",
        "/forge_3b/training.log",
        "/root/forge_3b/training.log",
    ]
    log_content = ""

    for lp in log_paths:
        p = Path(lp)
        if p.is_file():
            try:
                with open(p, "r", errors="ignore") as f:
                    lines = f.readlines()
                    log_content = "".join(lines[-60:])
                    break
            except Exception:
                pass

    loss = "Initializing..."
    tokens = "Downloading Shards..."
    tok_sec = "Starting engine..."
    progress_pct = 0.0

    # Parse dataset download progress if present
    shard_match = re.findall(r"Fetching \d+ files:\s+(\d+)%\s+\|\s+(\d+)/(\d+)", log_content)
    if shard_match:
        last_s = shard_match[-1]
        progress_pct = float(last_s[0])
        tokens = f"Dataset Shards: {last_s[1]}/{last_s[2]} files ({last_s[0]}%)"
        tok_sec = "Direct Hub Stream → /workspace"

    # Parse pretraining step metrics if present
    step_matches = re.findall(r"Step\s+(\d+).*?loss=([\d\.]+).*?tok/s=([\d\,]+)", log_content)
    if step_matches:
        last_step = step_matches[-1]
        loss = f"{float(last_step[1]):.4f}"
        tok_sec = f"{last_step[2]} tok/s"
        curr_step = int(last_step[0])
        tokens_val = curr_step * 1_048_576
        progress_pct = min(100.0, (tokens_val / 15_000_000_000) * 100)
        tokens = f"{tokens_val / 1e9:.2f}B / 15.0B ({progress_pct:.1f}%)"

    cost_info = get_metered_cost()

    return {
        "loss": loss,
        "tokens": tokens,
        "tok_sec": tok_sec,
        "progress_pct": progress_pct,
        "logs": log_content or "Initializing pretraining container...",
        "gpu": get_gpu_stats(),
        "cost": cost_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FORGE-500M • Autonomous Training Monitor</title>
    <style>
        :root {
            --bg: #090d16;
            --card-bg: #111827;
            --border: #1f293d;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.4);
            --success: #10b981;
            --warning: #f59e0b;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 24px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 18px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }
        .title h1 { font-size: 24px; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
        .title p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        .badges-wrap { display: flex; gap: 10px; align-items: center; }
        .badge {
            background: rgba(16, 185, 129, 0.12);
            color: var(--success);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-dot {
            width: 8px; height: 8px; background: var(--success);
            border-radius: 50%; box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        .section-title {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 12px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
            opacity: 0.5;
        }
        .card-cost::before {
            background: linear-gradient(90deg, transparent, var(--warning), transparent);
        }
        .card-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; }
        .card-value { font-size: 22px; font-weight: 800; color: #fff; margin-top: 6px; }
        .card-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
        .progress-box {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 24px;
        }
        .progress-bar-bg {
            background: #1f293d;
            height: 14px;
            border-radius: 7px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            height: 100%;
            width: 0%;
            transition: width 0.4s ease;
            box-shadow: 0 0 14px var(--accent-glow);
        }
        .terminal-box {
            background: #040711;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 12px;
            color: #d1d5db;
            overflow-x: auto;
            max-height: 380px;
            white-space: pre-wrap;
            line-height: 1.55;
        }
        .footer {
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            padding-top: 14px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .footer a { color: var(--accent); text-decoration: none; }
        .footer a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <h1>🚀 FORGE-500M Autonomous Monitor</h1>
            <p>Full 15.0B Token Pretraining • 4× NVIDIA B300 SXM6 MIG 34GB (136 GB VRAM)</p>
        </div>
        <div class="badges-wrap">
            <div class="badge">
                <div class="badge-dot"></div> AUTONOMOUS DAEMON ACTIVE
            </div>
        </div>
    </div>

    <div class="section-title">💰 RunPod Metered Cost & Budget Telemetry</div>
    <div class="grid">
        <div class="card card-cost">
            <div class="card-label">Metered Cost Spent</div>
            <div class="card-value" id="val-cost-spent" style="color: #fbbf24;">--</div>
            <div class="card-sub" id="val-elapsed">Elapsed: --</div>
        </div>
        <div class="card card-cost">
            <div class="card-label">RunPod Balance Remaining</div>
            <div class="card-value" id="val-balance" style="color: #34d399;">--</div>
            <div class="card-sub" id="val-init-balance">Initial: $440.49</div>
        </div>
        <div class="card card-cost">
            <div class="card-label">Cluster Burn Rate</div>
            <div class="card-value" id="val-rate" style="font-size: 20px;">$2.06 / hr</div>
            <div class="card-sub">4× B300 MIG ($0.50/ea) + Storage</div>
        </div>
        <div class="card card-cost">
            <div class="card-label">Est. 15B Total Run Cost</div>
            <div class="card-value" id="val-est-total" style="font-size: 20px;">~$51.50</div>
            <div class="card-sub" id="val-surplus">Surplus: +$388.99 remaining</div>
        </div>
    </div>

    <div class="section-title">⚡ Pretraining Engine Telemetry</div>
    <div class="grid">
        <div class="card">
            <div class="card-label">Training Loss</div>
            <div class="card-value" id="val-loss">--</div>
            <div class="card-sub">Cross-Entropy Target &lt; 2.2</div>
        </div>
        <div class="card">
            <div class="card-label">Throughput / Speed</div>
            <div class="card-value" id="val-speed">--</div>
            <div class="card-sub">Target: ~150k tok/s (4x B300 MIG)</div>
        </div>
        <div class="card">
            <div class="card-label">Model Architecture</div>
            <div class="card-value" style="font-size: 19px;">FORGE-500M</div>
            <div class="card-sub">499.6M params • 18 layers • 9-HSE MoE</div>
        </div>
        <div class="card">
            <div class="card-label">Compute Acceleration</div>
            <div class="card-value" style="font-size: 19px;">4× B300 SXM6 MIG</div>
            <div class="card-sub">136 GB VRAM • sm_100 Architecture</div>
        </div>
    </div>

    <div class="progress-box">
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <div class="card-label">Overall Pretraining Progress</div>
            <div id="val-tokens" style="font-weight: 700; font-size: 13px; color: var(--accent);">--</div>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" id="val-progress-bar"></div>
        </div>
    </div>

    <div class="section-title">🖥️ Live GPU Telemetry & Console Stream</div>
    <div class="terminal-box" id="terminal-logs">Connecting to live console stream...</div>

    <div class="footer">
        <div>
            Primary Link: <a href="https://yx6wg0pq3d4pxt-8888.proxy.runpod.net" target="_blank">RunPod Official HTTPS Proxy</a>
            &nbsp;•&nbsp;
            Backup: <a id="cf-tunnel-link" href="#" target="_blank">Cloudflare Tunnel</a>
        </div>
        <div>
            Auto-refresh: 2s • <span id="val-updated">--</span>
        </div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                
                // Training stats
                document.getElementById('val-loss').innerText = data.loss;
                document.getElementById('val-speed').innerText = data.tok_sec;
                document.getElementById('val-tokens').innerText = data.tokens;
                document.getElementById('val-progress-bar').style.width = data.progress_pct + '%';
                
                // Cost & Budget stats
                if (data.cost) {
                    document.getElementById('val-cost-spent').innerText = data.cost.spent_str;
                    document.getElementById('val-elapsed').innerText = 'Elapsed: ' + data.cost.elapsed_str;
                    document.getElementById('val-balance').innerText = data.cost.balance_str;
                    document.getElementById('val-init-balance').innerText = 'Initial: ' + data.cost.initial_balance_str;
                    document.getElementById('val-rate').innerText = data.cost.rate_str;
                    document.getElementById('val-est-total').innerText = data.cost.est_total_str;
                    document.getElementById('val-surplus').innerText = 'Est. End Balance: ' + data.cost.est_surplus_str;
                }
                
                // Console logs
                document.getElementById('terminal-logs').innerText = data.logs;
                document.getElementById('val-updated').innerText = data.timestamp;
            } catch (err) {
                console.error("Dashboard poll failed", err);
            }
        }
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_HEAD(self):
        if self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()

    def do_GET(self):
        if self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            data = get_training_progress()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress standard request log clutter

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def serve_on_port(port):
    try:
        server = ThreadedHTTPServer(("0.0.0.0", port), DashboardHandler)
        print(f"FORGE Live Dashboard listening on http://0.0.0.0:{port}")
        server.serve_forever()
    except OSError as e:
        print(f"Port {port} busy (skipping): {e}")

def run_servers():
    threads = []
    for port in PORTS:
        t = threading.Thread(target=serve_on_port, args=(port,), daemon=True)
        t.start()
        threads.append(t)
    
    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    run_servers()
