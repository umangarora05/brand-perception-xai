import { useState, useCallback, useRef } from 'react';
import './index.css';

/* ─────────── helpers ─────────── */
function ShapChart({ features, impacts }) {
  const maxImpact = Math.max(...impacts.map(Math.abs), 0.0001);
  return (
    <div className="shap-chart">
      {features.map((f, i) => {
        const val = impacts[i];
        const isNeg = val < 0;
        const pct = (Math.abs(val) / maxImpact) * 46;
        const color = isNeg ? 'var(--danger)' : 'var(--success)';
        return (
          <div className="shap-row" key={f}>
            <div className="shap-label">{f.replace(/_/g, ' ')}</div>
            <div className="shap-track">
              <div className="shap-zero" />
              <div
                className="shap-fill"
                style={{
                  background: color,
                  width: `${pct}%`,
                  left: isNeg ? `calc(50% - ${pct}%)` : '50%',
                  opacity: 0.85,
                }}
              />
            </div>
            <div className="shap-num">{val.toFixed(4)}</div>
          </div>
        );
      })}
    </div>
  );
}

function ChannelBar({ name, value255, color }) {
  const pct = (value255 / 255) * 100;
  return (
    <div className="channel-row">
      <div className="channel-name">{name}</div>
      <div className="channel-track">
        <div className="channel-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="channel-val">{value255}</div>
    </div>
  );
}

function deriveShapInsights(features, impacts) {
  let maxPos = { feature: null, impact: -Infinity };
  let maxNeg = { feature: null, impact: Infinity };
  let totalImpact = 0;

  features.forEach((f, i) => {
    const val = impacts[i];
    totalImpact += val;
    if (val > maxPos.impact) maxPos = { feature: f.replace(/_/g, ' '), impact: val };
    if (val < maxNeg.impact) maxNeg = { feature: f.replace(/_/g, ' '), impact: val };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', justifyContent: 'center', height: '100%' }}>
      {maxPos.impact > 0 && (
        <div style={{ padding: '12px', background: 'rgba(13,123,87,0.06)', borderLeft: '3px solid var(--success)', borderRadius: '0 8px 8px 0', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--success)', display: 'block', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '4px' }}>Primary Trust Driver</strong>
          The <strong>{maxPos.feature}</strong> attribute provides the strongest positive signal, increasing the trust coefficient by <strong>{(maxPos.impact).toFixed(4)}</strong>.
        </div>
      )}
      {maxNeg.impact < 0 && (
        <div style={{ padding: '12px', background: 'rgba(192,48,43,0.04)', borderLeft: '3px solid var(--danger)', borderRadius: '0 8px 8px 0', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--danger)', display: 'block', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '4px' }}>Primary Friction Point</strong>
          The <strong>{maxNeg.feature}</strong> attribute exerts the most downward pressure, reducing the trust score by <strong>{Math.abs(maxNeg.impact).toFixed(4)}</strong>.
        </div>
      )}
      <div style={{ padding: '12px', background: 'var(--bg-raised)', borderLeft: '3px solid var(--accent-mid)', borderRadius: '0 8px 8px 0', fontSize: '12.5px', color: 'var(--text-secondary)' }}>
        <strong style={{ color: 'var(--accent)', display: 'block', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '4px' }}>Net Calibration</strong>
        The net calibrated attribution is <strong>{totalImpact > 0 ? '+' : ''}{totalImpact.toFixed(4)}</strong>. Signals are routed through the Dynamic Signal Contextualiser to prevent false flags.
      </div>
    </div>
  );
}

/* ─────────── main app ─────────── */
export default function App() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef(null);

  const analyze = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, SVG, WEBP).');
      return;
    }
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/api/analyze', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Server error: ${res.statusText}`);
      setResults(await res.json());
    } catch (e) {
      setError(e.message || 'Failed to reach the analysis server.');
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    analyze(e.dataTransfer.files[0]);
  }, [analyze]);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onFileChange = (e) => analyze(e.target.files[0]);

  return (
    <>
      {/* ── NAV ── */}
      <nav className="nav">
        <div className="nav-brand">
          <div className="nav-brand-dot" />
          Differential Trust Engine
        </div>
        <div className="nav-badge">
          🔒 IN202641096863 A1
        </div>
      </nav>

      <div className="app-container">

        {/* ── HERO ── */}
        <section className="hero">

          <h1>
            Brand Logo<br />
            <span>Trust-Propensity Inference</span>
          </h1>
          <p className="hero-sub">
            An AI-powered explainability engine that computes a chromatic-condition-adaptive
            trust coefficient for brand logos and generates quantitative design prescriptions —
            grounded in emotion-manifold mapping and SHAP attribution.
          </p>

          <div className="patent-banner">
            <div className="patent-icon">📜</div>
            <div>
              <strong>Indian Patent Application No. 202641096863 A1</strong>
              <br />
              Filed 11 August 2026 · Vellore Institute of Technology · Published 14 August 2026
            </div>
          </div>
        </section>

        {/* ── STATS ── */}
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-number">13</div>
            <div className="stat-label">Multi-domain Features</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">2</div>
            <div className="stat-label">Ensemble Models</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">Logo-2K+</div>
            <div className="stat-label">Corroborating Corpus</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">SHAP</div>
            <div className="stat-label">Explainability Engine</div>
          </div>
        </div>

        {/* ── DIVIDER ── */}
        <div className="section-divider" />

        {/* ── PIPELINE ── */}
        <div className="section-head">
          <div className="section-title">Processing Pipeline</div>
          <div className="section-sub">Seven-stage patented architecture for brand trust inference</div>
        </div>
        <div className="pipeline-grid">
          {[
            ['01', 'Image Validation', 'File-type, dimensions, transparency and integrity checks before processing.'],
            ['02', 'Multi-Domain Feature Extraction', 'Chromatic, geometric, typographic and deep-embedding features extracted across 13 dimensions.'],
            ['03', 'Emotion-Manifold Mapping', 'Colour vector mapped to nearest neighbour emotion-labelled palette via KD-tree search.'],
            ['04', 'Trust-Inference Engine', 'Gradient-boosted decision-tree model generates normalised trust-propensity coefficient (0–1).'],
            ['05', 'Dynamic Signal Calibration', 'Achromatic Signal Gate mutes colour channels for monochromatic logos. Dominant Signal Contextualiser re-routes intentional brand colours.'],
            ['06', 'Quantitative Design Prescription', 'Modifiable features with adverse calibrated contributions generate bounded actionable redesign directives.'],
          ].map(([step, title, desc]) => (
            <div className="pipeline-card" key={step}>
              <div className="pipeline-step">Step {step}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>

        {/* ── UPLOAD ── */}
        <div className="upload-section">
          <div className="section-head">
            <div className="section-title">Audit Your Brand Logo</div>
            <div className="section-sub">Upload a logo image to generate a full trust-propensity report</div>
          </div>

          {!results && (
            <div
              className={`upload-zone ${dragging ? 'drag-over' : ''} ${loading ? 'uploading' : ''}`}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => !loading && fileRef.current?.click()}
            >
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFileChange} />
              {loading ? (
                <div className="loading-wrap">
                  <div className="spinner" />
                  <div className="loading-text">Calibrating trust-propensity signals…</div>
                </div>
              ) : (
                <>
                  <div className="upload-icon-wrap">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                      stroke="var(--accent)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </div>
                  <div className="upload-title">Drag &amp; Drop your brand asset</div>
                  <div className="upload-subtitle">or use the buttons below to upload a file</div>
                  <div className="upload-actions">
                    <button
                      className="btn-primary"
                      onClick={(e) => { e.stopPropagation(); fileRef.current?.click(); }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                      Browse Files
                    </button>
                  </div>
                  <div className="upload-formats">Supported formats: PNG · JPG · SVG · WEBP · BMP</div>
                </>
              )}
            </div>
          )}

          {error && (
            <div className="error-banner">
              <span>⚠</span> {error}
            </div>
          )}
        </div>

        {/* ── RESULTS DASHBOARD ── */}
        {results && (
          <>
            <div className="identity-row">
              <div className="identity-dot" />
              <div className="identity-file">{results.identity}</div>
              <div className="identity-status">Analysis Complete</div>
              <button
                className="btn-primary"
                style={{ marginLeft: '16px', padding: '6px 12px', fontSize: '11px', height: 'auto' }}
                onClick={() => setResults(null)}
              >
                ← Audit Another Logo
              </button>
            </div>

            <div className="dashboard">
              {/* Top Left: Trust Metrics */}
              <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="panel-title">Trust-Propensity Scores</div>
                <div className="panel-sub">Ensemble-corroborated coefficient</div>
                <div className="mode-chip" style={{ alignSelf: 'flex-start' }}>⚙ {results.engine_status}</div>
                <div className="score-item">
                  <div className="score-label">Ensemble Trust Coefficient</div>
                  <div className="score-value">{results.scores.ensemble_trust_coefficient.toFixed(4)}</div>
                </div>
                <div className="score-item">
                  <div className="score-label">Differential Model Score</div>
                  <div className="score-value secondary">{results.scores.differential.toFixed(4)}</div>
                </div>
                {results.scores.corroborating_167k !== null && (
                  <div className="score-item" style={{ borderBottom: 'none' }}>
                    <div className="score-label">Logo-2K+ Corroborating Score</div>
                    <div className="score-value secondary">{results.scores.corroborating_167k.toFixed(4)}</div>
                  </div>
                )}
              </div>

              {/* Top Right: Colour Composition */}
              <div className="glass-panel">
                <div className="panel-title">Chromatic Composition</div>
                <div className="panel-sub">Extracted ink colour from primary logo mark</div>
                <div
                  className="color-swatch"
                  style={{ background: `rgb(${results.ink_color.R},${results.ink_color.G},${results.ink_color.B})` }}
                >
                  RGB ({results.ink_color.R}, {results.ink_color.G}, {results.ink_color.B})
                  &nbsp;·&nbsp; {results.dominant_channel} dominant
                </div>
                <ChannelBar name="RED" value255={results.ink_color.R} color="#ef4444" />
                <ChannelBar name="GREEN" value255={results.ink_color.G} color="#10b981" />
                <ChannelBar name="BLUE" value255={results.ink_color.B} color="#3b82f6" />
              </div>

              {/* Bottom Row: SHAP Attribution (Full Width) */}
              <div className="glass-panel" style={{ gridColumn: '1 / -1' }}>
                <div className="panel-title">SHAP Feature Attribution</div>
                <div className="panel-sub">Context-calibrated impact on perceptual trust</div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', alignItems: 'center' }}>
                  <ShapChart
                    features={results.shap.features}
                    impacts={results.shap.impacts}
                  />
                  {deriveShapInsights(results.shap.features, results.shap.impacts)}
                </div>
              </div>
            </div>

            {/* Prescriptive Strategy (Full Width) */}
            <div className="glass-panel" style={{ marginBottom: 32 }}>
              <div className="panel-title">Prescriptive Strategy</div>
              <div className="panel-sub">Design friction signals with friction pathways</div>
              {results.prescriptions.length === 0 ? (
                <p style={{ color: 'var(--success)', fontSize: 14 }}>
                  ✓ No significant perceptual friction detected — brand asset is well-optimised.
                </p>
              ) : (
                <div className="quant-grid">
                  {results.prescriptions.map((rec, i) => (
                    <div className="prescription-card" key={i} style={{ margin: 0, height: '100%' }}>
                      <div className="prescription-tag">⚡ Friction Detected</div>
                      <div className="prescription-title">{rec.title}</div>
                      <div className="prescription-desc">{rec.description}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Quantitative Design Prescription Report */}
            {results.quantitative_report && results.quantitative_report.length > 0 && (
              <div className="glass-panel" style={{ marginBottom: 32 }}>
                <div className="panel-title">Quantitative Design Prescription Report</div>
                <div className="panel-sub">
                  Bounded, actionable redesign directives generated from context-calibrated attribution vectors
                </div>
                <div className="quant-grid">
                  {results.quantitative_report.map((rec, i) => (
                    <div className="quant-card" key={i}>
                      <div className="quant-header">
                        <div className="quant-pillar">{rec.pillar.replace(/_/g, ' ')}</div>
                        <div className="quant-score">Δ {Math.abs(rec.impact).toFixed(4)}</div>
                      </div>
                      <div className="quant-title">{rec.title}</div>
                      <div className="quant-desc">{rec.description}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Bottom Action */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 64 }}>
              <button className="btn-primary" onClick={() => setResults(null)}>
                ← Audit Another Logo
              </button>
            </div>
          </>
        )}

        {/* ── PATENT SECTION ── */}
        <div className="patent-section">
          <div className="patent-grid">
            <div>
              <div className="hero-label" style={{ marginBottom: 16, display: 'inline-flex' }}>
                📜 Patent Details
              </div>
              <div className="section-title">
                Context-Adaptive XAI System for Brand-Logo Trust-Propensity Inference
              </div>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7, marginTop: 12 }}>
                This platform implements the patented Differential Trust Engine — a context-adaptive
                explainable AI system that infers a trust-propensity coefficient from brand logos and
                generates quantitative design prescriptions by adapting explanation logic to chromatic
                operating conditions. The invention separates intentional dominant-colour brand strategies
                from modifiable design friction, preventing misleading recommendations for achromatic
                or near-monochromatic logos.
              </p>
              <div className="patent-detail-row">
                <div className="patent-detail-item">
                  <div className="patent-detail-label">Application No.</div>
                  <div className="patent-detail-value"><strong>IN 202641096863 A1</strong></div>
                </div>
                <div className="patent-detail-item">
                  <div className="patent-detail-label">Filing Date</div>
                  <div className="patent-detail-value">11 August 2026</div>
                </div>
                <div className="patent-detail-item">
                  <div className="patent-detail-label">Publication</div>
                  <div className="patent-detail-value">14 August 2026 · Journal No. 33/2026</div>
                </div>
                <div className="patent-detail-item">
                  <div className="patent-detail-label">Applicant</div>
                  <div className="patent-detail-value">Vellore Institute of Technology, Vellore, Tamil Nadu, India</div>
                </div>
                <div className="patent-detail-item">
                  <div className="patent-detail-label">Inventors</div>
                  <div className="patent-detail-value">Shynu P G · Umang Arora · Khushi</div>
                </div>
                <div className="patent-detail-item">
                  <div className="patent-detail-label">IPC Classes</div>
                  <div className="patent-detail-value">G06N 5/04 · G06K 9/62 · G06N 20/00 · G06N 5/02 · G06N 3/04</div>
                </div>
              </div>
            </div>
            <div>
              <div className="section-title" style={{ marginBottom: 8 }}>Novel Contributions</div>
              <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
                Key architectural innovations not found in prior art:
              </p>
              <ul className="claims-list">
                <li>Dynamic Signal Calibration Module with Achromatic Signal Gate — mutes colour attribution channels when saturation is below configurable threshold, eliminating irrelevant palette advice for monochromatic logos.</li>
                <li>Dominant Signal Contextualiser — re-routes intentional dominant brand colours away from the prescription pathway, preventing them from being flagged as design defects.</li>
                <li>Matched-Pair Differential Encoder — computes embedding displacement between baseline and revised logos to isolate the effect of deliberate design interventions on trust-propensity.</li>
                <li>Emotion-Manifold Mapping Module — maps colour vectors to nearest-neighbour emotion-labelled palette vectors via KD-tree indexing, grounding predictions in psychographic data.</li>
                <li>Ensemble Corroboration Module — combines Differential Trust Engine (60%) with Logo-2K+ general-logo model (40%) for robust cross-model trust coefficient computation.</li>
                <li>Quantitative Design Prescription Generator — converts calibrated attributions into bounded, auditable numerical redesign directives rather than general feature rankings.</li>
              </ul>
            </div>
          </div>
        </div>

      </div>

      {/* ── FOOTER ── */}
      <footer className="footer">
        <div></div>
        <div>Indian Patent Application IN202641096863 A1 · Vellore Institute of Technology · 2026</div>
        <div></div>
      </footer>
    </>
  );
}
