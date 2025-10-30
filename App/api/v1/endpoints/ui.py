from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


router = APIRouter()


@router.get("/ui", response_class=HTMLResponse)
def ui_page(request: Request) -> str:
	# Simple inline HTML to avoid template dependency
	return (
		"""
		<!doctype html>
		<html>
		<head>
			<meta charset=\"utf-8\">
			<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
			<title>SW-AI-6 Inference</title>
			<style>
				:root {
					--bg: #0a1224;
					--bg-accent: radial-gradient(1200px 600px at 20% -10%, rgba(56,189,248,0.08), transparent), radial-gradient(1000px 400px at 120% 10%, rgba(99,102,241,0.06), transparent);
					--card: #0f172a;
					--text: #e5e7eb;
					--muted: #9ca3af;
					--primary: #22d3ee; /* cyan-400 */
					--primary-600: #06b6d4; /* cyan-500 */
					--success: #10b981;
					--warning: #f59e0b;
					--danger: #ef4444;
					--border: #1f2937;
				}
				* { box-sizing: border-box; }
				body { margin: 0; background: var(--bg); background-image: var(--bg-accent); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
				.container { max-width: 980px; margin: 40px auto; padding: 0 20px; }
				.header { display:flex; align-items:center; justify-content:space-between; margin-bottom: 20px; }
				.title { font-size: 22px; font-weight: 700; letter-spacing: 0.3px; }
				.hero { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }
				.hero h1 { margin: 0 0 8px 0; font-size: 20px; }
				.hero p { margin: 0; color: var(--muted); }
				.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid var(--border); border-radius: 14px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
				.form-grid { display:grid; grid-template-columns: 1fr; gap:16px; }
				label { font-size: 14px; color: var(--muted); }
				input[type="file"] { width: 100%; padding: 10px; border: 1px dashed var(--border); border-radius: 10px; background: rgba(255,255,255,0.02); color: var(--text); }
				.checkbox { display:flex; align-items:center; gap:8px; }
				.btn { appearance:none; border:none; background: var(--primary); color: white; padding: 10px 16px; border-radius: 10px; font-weight: 600; cursor:pointer; transition: background .15s ease; }
				.btn:hover { background: var(--primary-600); }
				.actions { display:flex; gap:10px; align-items:center; }
				.status { display:flex; align-items:center; gap:8px; color: var(--muted); font-size: 14px; }
				.spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.25); border-top-color: white; border-radius: 50%; animation: spin 1s linear infinite; display:none; }
				@keyframes spin { to { transform: rotate(360deg); } }
				.section { margin-top: 18px; }
				.section h3 { margin: 0 0 10px 0; font-size: 16px; color: var(--muted); font-weight: 600; }
				pre#result { padding:14px; border:1px solid var(--border); background: #0b1220; border-radius:12px; overflow:auto; max-height: 420px; }
				.badge { display:inline-block; padding:4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; border:1px solid var(--border); }
				.badge.success { background: rgba(16,185,129,.15); color: #34d399; }
				.badge.warn { background: rgba(245,158,11,.15); color: #fbbf24; }
				.badge.danger { background: rgba(239,68,68,.15); color: #f87171; }
				.badge.info { background: rgba(37,99,235,.15); color: #60a5fa; }
				#pdf-section a { color: #93c5fd; text-decoration: none; font-weight: 600; }
				.readable { padding:14px; border:1px solid var(--border); background: rgba(255,255,255,0.02); border-radius:12px; }
				.kv { display:grid; grid-template-columns: 180px 1fr; gap:10px; }
				.kv div { padding:6px 0; border-bottom: 1px dashed var(--border); }
				.kv div strong { color: var(--muted); font-weight: 600; }
				ul.compact { margin: 6px 0 0 18px; padding: 0; }
				ul.compact li { margin: 4px 0; }
				#pdf-section a:hover { text-decoration: underline; }
				/* Responsive adjustments */
				@media (max-width: 640px) {
					.container { margin: 20px auto; padding: 0 12px; }
					.title { font-size: 18px; }
					.hero { padding: 14px; }
					.hero h1 { font-size: 18px; }
					.card { padding: 16px; }
					.actions { flex-direction: column; align-items: stretch; gap: 8px; }
					.btn { width: 100%; padding: 12px 14px; }
					.status { justify-content: center; }
					.kv { grid-template-columns: 1fr; }
					label { font-size: 13px; }
					input[type="file"] { padding: 12px; font-size: 14px; }
					pre#result { max-height: 300px; font-size: 12px; }
				}
			</style>
		</head>
		<body>
			<div class=\"container\">
				<div class=\"header\">
					<div class=\"title\">SW-AI-6 • Knowledge-Guided Autism Screening</div>
				</div>
				<div class="hero">
					<h1>Early Autism Screening Assistant</h1>
					<p>Upload a short video to receive a preliminary, knowledge-guided screening summary with severity, confidence, and domain highlights. This is an assistive tool and not a medical diagnosis.</p>
				</div>
				<div class="card">
					<form id="infer-form" enctype="multipart/form-data">
						<div class="form-grid">
							<div>
								<label>Upload video (max 100MB)</label>
								<input type="file" name="video" accept="video/*" required>
							</div>
							<div class="actions">
								<button class="btn" type="submit">Run Inference</button>
								<div class="status"><span class="spinner" id="spin"></span><span id="status-text">Idle</span></div>
							</div>
						</div>
					</form>

					<div class="section">
						<h3>Severity & Report</h3>
						<div id="severity-row"></div>
						<div id="pdf-section" style="margin-top:8px;"></div>
					</div>

					<div class="section"> 
						<h3>Results</h3>
						<div id="readable" class="readable"></div>
					</div>
				</div>

				<div class="section" style="margin-top:16px">
					
				</div>
			</div>

			<script>
			const form = document.getElementById('infer-form');
				const readable = document.getElementById('readable');
				const pdfSec = document.getElementById('pdf-section');
			const spin = document.getElementById('spin');
			const st = document.getElementById('status-text');
			const sevRow = document.getElementById('severity-row');

			function severityBadge(sev) {
			  const span = document.createElement('span');
			  span.className = 'badge info';
			  const s = String(sev || '').toLowerCase();
			  if (s.includes('minimal')) span.className = 'badge';
			  if (s.includes('low')) span.className = 'badge success';
			  if (s.includes('moderate')) span.className = 'badge warn';
			  if (s.includes('high')) span.className = 'badge danger';
			  span.textContent = sev || 'Unknown';
			  return span;
			}

				function renderReadable(summary) {
				  const scores = summary.scores || {};
				  const tsn = typeof scores.tsn === 'number' ? scores.tsn.toFixed(3) : (scores.tsn ?? 'N/A');
				  const sgcn = typeof scores.sgcn === 'number' ? scores.sgcn.toFixed(3) : (scores.sgcn ?? 'N/A');
				  const stgcn = typeof scores.stgcn === 'number' ? scores.stgcn.toFixed(3) : (scores.stgcn ?? 'N/A');
				  const fused = typeof summary.fused_score === 'number' ? summary.fused_score.toFixed(3) : (summary.fused_score ?? 'N/A');
				  const severity = scores.severity || summary.severity || 'Unknown';
				  const kg = summary.knowledge_guidance || {};
				  const base = kg.base_explanation || '';
				  const domains = kg.domains || {};
				  const domainItems = Object.entries(domains).slice(0, 6).map(([name, info]) => {
				    const desc = (info && info.description) ? info.description : '';
				    return `<li><strong>${name}</strong>: ${desc}</li>`;
				  }).join('');
				  return `
				    <div class="kv">
				      <div><strong>Severity</strong></div><div>${severity}</div>
				      <div><strong>Fused score</strong></div><div>${fused}</div>
				      <div><strong>TSN score</strong></div><div>${tsn}</div>
				      <div><strong>SGCN score</strong></div><div>${sgcn}</div>
				      <div><strong>ST-GCN score</strong></div><div>${stgcn}</div>
				    </div>
				    ${base ? `<div style="margin-top:10px;"><strong style="color: var(--muted);">Summary</strong><div style="margin-top:6px;">${base}</div></div>` : ''}
				    ${domainItems ? `<div style="margin-top:10px;"><strong style="color: var(--muted);">Domains</strong><ul class="compact">${domainItems}</ul></div>` : ''}
				  `;
				}

				form.addEventListener('submit', async (e) => {
			  e.preventDefault();
				  readable.innerHTML = '';
			  pdfSec.innerHTML = '';
			  sevRow.innerHTML = '';
			  spin.style.display = 'inline-block';
			  st.textContent = 'Running inference...';
			  const fd = new FormData(form);
			  try {
			    const resp = await fetch('/api/v1/infer', { method: 'POST', body: fd });
			    const json = await resp.json();
			    if (!resp.ok) throw new Error(json.detail || 'Inference failed');
				    const summary = json.summary || {};
				    readable.innerHTML = renderReadable(summary);

			    // Severity badge
			    const sev = (summary.scores && summary.scores.severity) || summary.severity;
			    const label = document.createElement('div');
			    label.style.marginBottom = '8px';
			    label.textContent = 'Severity: ';
			    label.appendChild(severityBadge(sev));
			    sevRow.appendChild(label);

			    // Generate PDF
			    st.textContent = 'Generating PDF report...';
			    const reportId = 'report_' + Date.now();
			    const pdfResp = await fetch(`/api/v1/report/${reportId}`, {
			      method: 'POST',
			      headers: { 'Content-Type': 'application/json' },
			      body: JSON.stringify(summary)
			    });
			    const pdfJson = await pdfResp.json();
			    if (pdfResp.ok) {
			      const link = document.createElement('a');
			      link.href = `/api/v1/report/${reportId}`;
			      link.textContent = 'Download PDF report';
			      link.setAttribute('download', `${reportId}.pdf`);
			      pdfSec.appendChild(link);
			      st.textContent = 'Completed';
			    } else {
			      pdfSec.textContent = 'PDF generation failed: ' + (pdfJson.detail || 'Unknown error');
			      st.textContent = 'Completed (PDF failed)';
			    }
			  } catch (err) {
			    pdfSec.textContent = 'Error: ' + err.message;
			    st.textContent = 'Error';
			  } finally {
			    spin.style.display = 'none';
			  }
			});
			</script>
		</body>
		</html>
		"""
	)


