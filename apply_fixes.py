#!/usr/bin/env python3
"""apply_fixes.py — Applies all 5 frontend fixes to frontend/index.html"""
from pathlib import Path
import re, sys

path = Path("frontend/index.html")
if not path.exists():
    print(f"X {path} not found. Run from project root.")
    sys.exit(1)

html = path.read_text(encoding="utf-8")
orig_len = len(html)
log = []

# ============ FIX 1: Readability CSS ============
readability_css = '''<style id="readability-lift-v1">
.sec-lead{font-size:14.5px!important;line-height:1.72!important}
.step-tag,.dossier-label,.insight-overline,.viewport-overline,.lbl{font-size:11px!important;letter-spacing:.11em!important}
.hero-desc{font-size:15px!important;line-height:1.75!important}
.hero-trust-note,.hero-data-item small,.hero-data-rail-note b{font-size:11.5px!important}
.hero-data-item span{font-size:10px!important}
.overview-card p{font-size:12.5px!important;line-height:1.6!important}
.overview-card-top,.overview-card-link,.overview-method-note p{font-size:11.5px!important}
.kpi-k{font-size:10.5px!important}
.kpi-d,.verdict-plain,.pearl,.lab-footnote,.validation-method-note span,.crew-model-note span{font-size:12.5px!important;line-height:1.6!important}
.triage-box>div:first-child>div:nth-child(2){font-size:17px!important}
.gas-head{font-size:12.5px!important}
.tw th{font-size:11px!important}
.tw td{font-size:12.5px!important}
.row{font-size:12.5px!important}
.fm{font-size:12.5px!important;line-height:1.8!important}
.ai,.research-answer,.insight-output{font-size:13.5px!important;line-height:1.7!important}
.research-dossier-card p,.research-corpus-card p,.research-ask-head p,.architecture-card p,.readme-metric-row>span,.readme-caveat,.readme-list,.source-register-grid p,.research-footer-note{font-size:13px!important;line-height:1.65!important}
.architecture-card h4{font-size:15px!important}
.scene-text{font-size:13.5px!important;line-height:1.7!important}
.scene-data{font-size:11.5px!important}
footer{font-size:11.5px!important}
</style>
'''
if 'id="readability-lift-v1"' not in html:
    html = html.replace('</head>', readability_css + '</head>', 1)
    log.append("OK 1: Readability CSS added")

# ============ FIX 2: Hero data rail ============
old = '<div class="hero-data-item"><span>CORE VALIDATION</span><b>2.83<small>% MAPE</small></b><small>In-sample · 5 core cases</small></div>'
new = '<div class="hero-data-item"><span>NESTED LOO ERROR</span><b>10.4–12.3<small>% MAPE</small></b><small>5.90% in-sample · 16 constants refit</small></div>'
if old in html:
    html = html.replace(old, new, 1)
    log.append("OK 2: Hero rail updated (2.83 -> 10.4-12.3)")

# ============ FIX 3: Workbench tab + validation note + 4th KPI ============
old = 'data-tab="parity">NASA Flight Parity (2.83% MAPE)</button>'
new = 'data-tab="parity">NASA Flight Parity (5.90% In-Sample · 10.4% Nested LOO)</button>'
if old in html:
    html = html.replace(old, new, 1)
    log.append("OK 3a: Workbench tab updated")

old = '2.83% is the core five-case in-sample MAPE. The 13-point 6.68% LOO refits four selected parameters per fold; other empirical parameters previously fit on the full dataset remain fixed, so this is not fully leakage-free and may be optimistic.'
new = 'Core in-sample MAPE is 2.83% (5 runs), Combined 13-run in-sample MAPE is 5.90% (R² = 0.993). Under strict fully nested leave-one-out validation (refitting all 16 empirical constants from scratch per fold across 3 seeds), MAPE is 10.4–12.3%.'
if old in html:
    html = html.replace(old, new, 1)
    log.append("OK 3b: Validation note updated")

old = '<div class="kpi"><div class="kpi-k">13-Point LOO · Partial Refit</div><div class="kpi-v">6.68<small>% MAPE</small></div><div class="kpi-d">Four selected parameters refit per fold; other full-set empirical fits remain fixed.</div></div>'
new = '<div class="kpi"><div class="kpi-k">Fully Nested LOO (3 Seeds)</div><div class="kpi-v">10.4–12.3<small>% MAPE</small></div><div class="kpi-d">All 16 empirical constants refit from scratch per fold (median APE 6.8–11.2%).</div></div>'
if old in html:
    html = html.replace(old, new, 1)
    log.append("OK 3c: 4th KPI card updated (6.68 -> 10.4-12.3)")

# ============ FIX 4a: <audio> tag ============
if 'id="storyAudio"' not in html:
    anchor = '<div class="story-theater" id="storyTheater" role="dialog" aria-modal="true">'
    if anchor in html:
        html = html.replace(anchor, anchor + '\n  <audio id="storyAudio" src="/static/audio/story.mp3" preload="auto"></audio>', 1)
        log.append("OK 4a: Audio tag added (/static/audio/story.mp3)")

# ============ FIX 4b: Mute button ============
old = '<button class="btn" id="closeStoryBtn" type="button" style="padding:6px 14px;font-size:11px">✕ Close</button>'
new = '<div style="display:flex;gap:8px;align-items:center">\n        <button class="btn" id="storyMuteBtn" type="button" style="padding:6px 12px;font-size:11px">🔊 Audio On</button>\n        <button class="btn" id="closeStoryBtn" type="button" style="padding:6px 14px;font-size:11px">✕ Close</button>\n      </div>'
if 'id="storyMuteBtn"' not in html and old in html:
    html = html.replace(old, new, 1)
    log.append("OK 4b: Mute button added")

# ============ FIX 4c: syncStoryAudio function + replace 4 functions ============
old_open = '''function openStoryTheater(){
  const modal=$('storyTheater');if(!modal)return;modal.classList.add('open');modal.setAttribute('aria-hidden','false');
  sceneIdx=0;storyPaused=false;storyRemaining=6000;storyPauseStarted=0;setScene(0);storyStartT=performance.now();
  const loop=now=>{if(!modal.classList.contains('open'))return;if(!storyPaused)drawStoryScene(now-storyStartT);const remain=storyPaused?storyRemaining:Math.max(0,storyDeadline-Date.now());const elapsed=sceneIdx*6+(6-remain/1000);const clock=$('storyClock');if(clock){const ss=String(Math.floor(elapsed)).padStart(2,'0');clock.textContent=`00:${ss} / 00:30`;}storyAnimId=requestAnimationFrame(loop);};
  cancelAnimationFrame(storyAnimId);storyAnimId=requestAnimationFrame(loop);startStoryClock();
}
function pauseStory(){
  const pause=$('storyPauseBtn');
  if(!storyPaused){storyRemaining=Math.max(0,storyDeadline-Date.now());storyPaused=true;storyPauseStarted=performance.now();clearTimeout(storyAutoTimer);if($('storyProgress'))$('storyProgress').style.width=(100-storyRemaining/60)+'%';if(pause)pause.textContent='▶ Resume';}
  else{storyStartT+=performance.now()-storyPauseStarted;storyPauseStarted=0;storyPaused=false;if(pause)pause.textContent='Ⅱ Pause';startStoryClock();}
}
function moveStoryScene(delta){
  if(delta>0&&sceneIdx>=SCENES.length-1){closeStoryTheater();return;}
  clearTimeout(storyAutoTimer);storyPaused=false;storyPauseStarted=0;storyRemaining=6000;storyStartT=performance.now();setScene(sceneIdx+delta);startStoryClock();
}
function closeStoryTheater(){
  const modal=$('storyTheater');if(!modal)return;modal.classList.remove('open');modal.setAttribute('aria-hidden','true');
  cancelAnimationFrame(storyAnimId);clearTimeout(storyAutoTimer);storyPaused=false;storyPauseStarted=0;
}'''

new_open = '''function syncStoryAudio(seekSec){
  const au=$('storyAudio'),mb=$('storyMuteBtn');if(!au)return;
  if(typeof seekSec==='number'&&!Number.isNaN(seekSec)){try{au.currentTime=Math.max(0,seekSec);}catch(e){}}
  if(storyPaused){au.pause();}
  else{au.play().then(()=>{if(mb)mb.textContent=au.muted?'🔇 Unmute':'🔊 Audio On';}).catch(()=>{if(mb)mb.textContent='🔇 Tap for Audio';toast('Click "Tap for Audio" to enable narration');});}
}
function openStoryTheater(){
  const modal=$('storyTheater');if(!modal)return;modal.classList.add('open');modal.setAttribute('aria-hidden','false');
  sceneIdx=0;storyPaused=false;storyRemaining=6000;storyPauseStarted=0;setScene(0);storyStartT=performance.now();
  syncStoryAudio(0);
  const loop=now=>{if(!modal.classList.contains('open'))return;if(!storyPaused)drawStoryScene(now-storyStartT);const remain=storyPaused?storyRemaining:Math.max(0,storyDeadline-Date.now());const elapsed=sceneIdx*6+(6-remain/1000);const clock=$('storyClock');if(clock){const ss=String(Math.floor(elapsed)).padStart(2,'0');clock.textContent=`00:${ss} / 00:30`;}storyAnimId=requestAnimationFrame(loop);};
  cancelAnimationFrame(storyAnimId);storyAnimId=requestAnimationFrame(loop);startStoryClock();
}
function pauseStory(){
  const pause=$('storyPauseBtn');
  if(!storyPaused){storyRemaining=Math.max(0,storyDeadline-Date.now());storyPaused=true;storyPauseStarted=performance.now();clearTimeout(storyAutoTimer);if($('storyProgress'))$('storyProgress').style.width=(100-storyRemaining/60)+'%';if(pause)pause.textContent='▶ Resume';}
  else{storyStartT+=performance.now()-storyPauseStarted;storyPauseStarted=0;storyPaused=false;if(pause)pause.textContent='Ⅱ Pause';startStoryClock();}
  syncStoryAudio();
}
function moveStoryScene(delta){
  if(delta>0&&sceneIdx>=SCENES.length-1){closeStoryTheater();return;}
  clearTimeout(storyAutoTimer);storyPaused=false;storyPauseStarted=0;storyRemaining=6000;storyStartT=performance.now();setScene(sceneIdx+delta);startStoryClock();
  syncStoryAudio(sceneIdx*6);
}
function closeStoryTheater(){
  const modal=$('storyTheater');if(!modal)return;modal.classList.remove('open');modal.setAttribute('aria-hidden','true');
  cancelAnimationFrame(storyAnimId);clearTimeout(storyAutoTimer);storyPaused=false;storyPauseStarted=0;
  const au=$('storyAudio');if(au){au.pause();try{au.currentTime=0;}catch(e){}}
}'''

if old_open in html:
    html = html.replace(old_open, new_open, 1)
    log.append("OK 4c: syncStoryAudio + 4 functions replaced (audio sync)")

# ============ FIX 4d: Mute button handler in wire() ============
anchor = "  $('storyPauseBtn').onclick=pauseStory;"
if anchor in html and "$('storyMuteBtn').onclick" not in html:
    handler = anchor + '''
  $('storyMuteBtn').onclick=()=>{const au=$('storyAudio'),mb=$('storyMuteBtn');if(!au)return;if(au.paused&&!storyPaused){au.play().catch(()=>{});au.muted=false;if(mb)mb.textContent='🔊 Audio On';}else{au.muted=!au.muted;if(mb)mb.textContent=au.muted?'🔇 Unmute':'🔊 Audio On';}};'''
    html = html.replace(anchor, handler, 1)
    log.append("OK 4d: Mute button onclick handler added")

# ============ FIX 5: Download handler ============
if "a.download='Ashes_of_Microgravity_README.md'" not in html:
    anchor = "async function boot(){"
    if anchor in html:
        dl = '''const dlBtn=$('dlReadmeBtn');
if(dlBtn)dlBtn.onclick=e=>{e.preventDefault();const md=`# Ashes of Microgravity\\n\\n- Core In-Sample MAPE: 2.83% (5 runs)\\n- Combined In-Sample MAPE: 5.90% (13 runs)\\n- Fully Nested LOO MAPE: 10.4-12.3% (16 constants, 3 seeds)\\n- Materials: 118 (3 flight-calibrated)\\n- Toxicology: NASA JSC 20584 + ISO 13571\\n\\nTeam: Satyam Sharma (B.Tech CSE) & Shivam Sharma (MBBS)\\nLicense: Apache 2.0`;const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([md],{type:'text/markdown'}));a.download='Ashes_of_Microgravity_README.md';a.click();};

'''
        html = html.replace(anchor, dl + anchor, 1)
        log.append("OK 5: Download handler added")

# ============ FIX 6: Replace entire research section ============
research_start = '<section class="wrap research-page" id="research" data-page="research">'
research_end_marker = '</section>\n\n<footer data-page="shared">'

if research_start in html and research_end_marker in html:
    start_idx = html.find(research_start)
    end_idx = html.find(research_end_marker, start_idx)
    if end_idx > start_idx:
        new_research = '''<section class="wrap research-page" id="research" data-page="research">
  <div class="section-heading-row">
    <div>
      <div class="step-tag">04 / PROJECT DOCUMENTATION &amp; NASA LITERATURE</div>
      <h2 class="sec-title">Project Dossier &amp; Research Library</h2>
      <p class="sec-lead">Complete technical specification, flight-calibration audit, crew toxicology methodology, and citation-grounded Q&amp;A over 3,426 indexed NASA report chunks.</p>
    </div>
    <div class="section-index"><span>DOCUMENTATION</span><b>04</b><i>README / RAG</i></div>
  </div>
  <div class="research-overview-grid">
    <article class="research-dossier-card">
      <div class="dossier-label">NASA SPACE APPS 2026 · FLAME IN FREEFALL</div>
      <h3>Fire does not rise in orbit. Without buoyancy, it creeps spherically and fills cabins with invisible toxicants.</h3>
      <p>Full 3D reacting CFD takes hours per run and rarely models downstream crew health. <b>Ashes of Microgravity</b> couples reduced-order flame-spread physics (de Ris / Quintiere), a 2D thermal-pyrolysis PDE solver, a 41-feature PyTorch ensemble, and a dynamic cabin toxicology ODE (ISO 13571 FED &amp; CFK COHb%) to screen both material flammability and crew survival in milliseconds.</p>
      <div class="dossier-tags"><span>13 NASA FLIGHT RUNS</span><span>2D THERMAL PDE</span><span>41-D PYTORCH ENSEMBLE</span><span>NASA JSC 20584 SMACs</span><span>ISO 13571 FED</span></div>
      <a class="dossier-download" href="#download-readme" id="dlReadmeBtn">Download Technical Dossier (.md) <b>↓</b></a>
    </article>
    <article class="research-corpus-card">
      <div class="corpus-top"><span class="corpus-dot"></span> GROUNDED LITERATURE RAG</div>
      <div class="corpus-count">3,426 <small>chunks</small></div>
      <p>ChromaDB vector index over NASA Saffire I–V, BASS-II, and SoFIE combustion reports with cosine-distance anti-hallucination filtering (≤ 1.4) and offline extractive fallback.</p>
      <div class="corpus-sources"><span>SAFFIRE I–V</span><span>BASS-II</span><span>SoFIE</span><span>NTRS</span></div>
      <small class="corpus-note">Research &amp; educational screening tool only — not a substitute for NASA-STD-6001B testing or clinical medical advice.</small>
    </article>
  </div>
  <section class="research-ask-card">
    <div class="research-ask-head">
      <div><div class="dossier-label">ASK THE NASA CORPUS</div><h3>Query Microgravity Combustion Reports</h3><p>Ask about limiting oxygen index in 0g, Saffire fabric vs. PMMA slab behavior, forced ventilation thresholds, or cabin toxicants.</p></div>
      <div class="ask-mark">RAG</div>
    </div>
    <div class="q" style="margin-top:12px">
      <button class="btn" type="button" data-q="How does microgravity change the Limiting Oxygen Index (LOI) compared to 1g?">0g vs 1g LOI shift</button>
      <button class="btn" type="button" data-q="Why does concurrent flame spread behave differently from opposed flow in Saffire experiments?">Concurrent vs Opposed in Saffire</button>
      <button class="btn" type="button" data-q="What are the primary toxic combustion products of PVC and Kapton in spacecraft fires?">PVC &amp; Kapton toxic gases</button>
    </div>
    <div class="search-row research-search-row">
      <input type="text" id="aq" placeholder="Ask about flame spread, extinction, oxygen transport, smoke species…" aria-label="Ask NASA research question">
      <button class="btn pri" id="ag" type="button">Search reports <span>↗</span></button>
    </div>
    <div class="ai research-answer" id="ao" aria-live="polite"></div>
  </section>
  <div class="research-section-heading"><div><div class="dossier-label">SYSTEM ARCHITECTURE</div><h3>Four-Engine Hybrid Pipeline</h3></div><span>FASTAPI v2.1.0 · 11 ENDPOINTS</span></div>
  <div class="architecture-grid">
    <article class="architecture-card"><span class="architecture-index">01</span><div class="architecture-icon">∿</div><h4>1D &amp; Quasi-2D Physics</h4><p>Coupled boundary-layer heat transfer (de Ris / Quintiere), Stefan-Boltzmann radiation, char insulation, lateral O₂ entrainment (BASS-II), and clamped-edge quenching.</p><small>BACKEND/FIRE_MODEL.PY</small></article>
    <article class="architecture-card"><span class="architecture-index">02</span><div class="architecture-icon">▦</div><h4>2D Transient PDE Solver</h4><p>NumPy Cartesian thermal-pyrolysis grid with automatic Fourier CFL sub-stepping (Fo ≤ 0.20), resolving parabolic (clamped) vs. concave (free-edge) flame fronts.</p><small>SIMULATE_2D()</small></article>
    <article class="architecture-card"><span class="architecture-index">03</span><div class="architecture-icon">⌁</div><h4>41-D Neural Surrogate</h4><p>5-member SiLU PyTorch deep ensemble trained on 35,000 stratified multi-gravity points (~2 ms). Includes a 30% divergence guardrail where physics remains authoritative.</p><small>AI/SURROGATE.PY</small></article>
    <article class="architecture-card"><span class="architecture-index">04</span><div class="architecture-icon">◌</div><h4>Crew Toxicology ODE</h4><p>1-minute cabin mass-balance with ECLSS decay (1.8 ACH), SFPE/Tewarson yields, ISO 13571 Purser FED, CFK COHb% uptake, and 100% O₂ PBA mask washout.</p><small>BACKEND/TOXICOLOGY.PY</small></article>
  </div>
  <div class="research-readme-grid">
    <article class="readme-panel">
      <div class="dossier-label">FLIGHT VALIDATION &amp; SCIENTIFIC RIGOR</div>
      <h3>Honest Error Reporting (13 NASA Flight Runs)</h3>
      <div class="readme-metric-row"><b>2.83%</b><span><b>Core In-Sample MAPE</b> (R² = 1.000) across 5 wide-plate 101.3 kPa Saffire-I, II, and V runs.</span></div>
      <div class="readme-metric-row"><b>5.90%</b><span><b>Combined In-Sample MAPE</b> (R² = 0.993) across all 13 Saffire I–V &amp; BASS-II runs (incl. rods, low-P, narrow sheets).</span></div>
      <div class="readme-metric-row"><b>10.4–12.3%</b><span><b>Fully Nested Leave-One-Out MAPE</b> across 3 seeds — all 16 empirical constants refit from scratch per fold (median APE 6.8–11.2%).</span></div>
      <p class="readme-caveat"><b>Key Physical Insight — Extinct ≠ Safe:</b> Even when flame spread extinguishes, an assumed 0.012 g/s smolder floor releases corrosive HCl from halogenated polymers (f-PVC), triggering CRITICAL crew triage while PMMA smolders benignly.</p>
    </article>
    <article class="readme-panel">
      <div class="dossier-label">KNOWN LIMITATIONS &amp; NASA-STD-7009A ALIGNMENT</div>
      <h3>Scope &amp; Credibility Boundaries</h3>
      <ul class="readme-list">
        <li><b>3 Flight-Validated Fuels of 118:</b> Direct microgravity telemetry exists for <b>SIBAL</b>, <b>PMMA</b>, and <b>Nomex 450</b> (used as proxy for Saffire-IV Nomex HT9040). Other 115 materials are thermophysical extrapolations with explicit applicability scores (0–100).</li>
        <li><b>Single-Point Regimes:</b> Low-pressure radiative feedback (55 kPa, Saffire-V S4) and near-LOI damping (26% O₂, Saffire-IV S2) rely on single runs and carry wider uncertainty.</li>
        <li><b>Surrogate Is an Emulator:</b> It reproduces the physics engine (~2.2% avg gap on plates/fabrics, 16–26% on low-flow PMMA rods) and is not independently validated against flight data.</li>
        <li><b>Well-Mixed Cabin Assumption:</b> Toxicology ODE models uniform cabin mixing (ISS 106 m³, Orion 9.3 m³, Gateway 60 m³, Mars 250 m³) with an assumed 2.1× 0g CO/soot multiplier.</li>
        <li><b>Disclaimer:</b> Research and educational tool only — not clinical guidance or NASA-STD-6001B hardware certification.</li>
      </ul>
    </article>
  </div>
  <section class="source-register">
    <div class="research-section-heading"><div><div class="dossier-label">PRIMARY LITERATURE &amp; STANDARDS</div><h3>Calibration &amp; Physiological Basis</h3></div><span>CONSTANTS PROVENANCE</span></div>
    <div class="source-register-grid">
      <div><b>Combustion Theory &amp; Flight Data</b><p>de Ris (1969), Quintiere (2006), Olson &amp; Ferkul (2019) Saffire-I/III, Ruff et al. (2021) Saffire-V, Urban et al. (ICES-2021-125) Saffire-IV, and Ferkul et al. BASS-II.</p></div>
      <div><b>Toxicology &amp; Clinical Kinetics</b><p>NASA JSC 20584 Spacecraft Maximum Allowable Concentrations (SMACs), ISO 13571 Purser FED, Tewarson SFPE Handbook yields, and Coburn-Forster-Kane (CFK) COHb kinetics.</p></div>
      <div><b>Uncertainty &amp; Credibility</b><p>Kline &amp; McClintock (1953) first-order Taylor sensor tolerance propagation combined with heuristic applicability scoring inspired by NASA-STD-7009A.</p></div>
    </div>
  </section>
  <div class="research-footer-note">
    <span>TEAM &amp; ATTRIBUTION</span>
    <b>Satyam Sharma</b> (B.Tech CSE — Data Science · Combustion physics, 2D PDE, PyTorch surrogate, FastAPI &amp; frontend) &nbsp;·&nbsp;
    <b>Shivam Sharma</b> (MBBS — Clinical Medicine · Spacecraft crew toxicology, SMACs, ISO 13571 FED &amp; CFK COHb%)
    <span class="license-note">Apache License 2.0 · Not affiliated with NASA</span>
  </div>
</section>'''
        html = html[:start_idx] + new_research + html[end_idx:]
        log.append("OK 6: Research page completely rebuilt (README dossier + download)")

# ============ Save ============
path.write_text(html, encoding="utf-8")
print("\n" + "="*60)
print(f"File: {path}")
print(f"Size: {orig_len} -> {len(html)} bytes ({len(html)-orig_len:+d})")
print("="*60)
for l in log: print(l)
print("="*60)