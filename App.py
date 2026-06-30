from __future__ import annotations
import os, io, uuid, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — must be before plt import
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── data_model import ──────────────────────────────────────────────────────────
try:
    from data_model import (
        initial_analysis, clean_data, numeric_stats, categorical_stats,
        distribution_plots, correlation_analysis, time_series_analysis,
        build_ai_prompt, detect_problematic_columns, sanitize_for_display,
    )
    DATA_MODEL_OK = True
except ImportError as _e:
    DATA_MODEL_OK = False
    _import_error = str(_e)

try:
    import openai as _openai_mod
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False
    _openai_mod = None

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# ── app setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# In-memory session store: session_id -> {'df': DataFrame, 'clean_df': DataFrame}
_sessions: dict = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    """Save a matplotlib Figure to a base64-encoded PNG string, then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def _clean(v):
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    if isinstance(v, float):
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return str(v)
    if isinstance(v, pd.Series):
        return {str(k): _clean(vv) for k, vv in v.items()}
    if isinstance(v, pd.DataFrame):
        return _clean(v.to_dict(orient='records'))
    if isinstance(v, dict):
        return {str(k): _clean(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def df_preview(df: pd.DataFrame, n: int = 50) -> dict:
    """Return the first n rows as JSON-safe columns + row data."""
    sample = df.head(n).copy()
    for col in sample.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
        sample[col] = sample[col].astype(str)
    sample = sample.where(pd.notna(sample), None)
    return {
        'columns': [str(c) for c in sample.columns],
        'data': _clean(sample.values.tolist()),
    }


def call_openai(prompt: str, model: str = 'gpt-4o-mini',
                max_tokens: int = 700, temperature: float = 0.2) -> str:
    if not HAS_OPENAI:
        return '❌ openai package not installed. Add it to requirements.txt.'
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        return '❌ OPENAI_API_KEY environment variable is not set.'
    try:
        client = _openai_mod.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content':
                    'You are an expert data analyst. Provide concise, actionable insights.'},
                {'role': 'user', 'content': prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f'❌ OpenAI call failed: {e}'


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo-data')
def demo_data():
    """Serve the bundled demo CSV for the 'Try Demo' button."""
    demo_path = os.path.join(app.root_path, 'static')
    return send_from_directory(demo_path, 'demo_data.csv',
                               mimetype='text/csv',
                               as_attachment=False)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not DATA_MODEL_OK:
        return jsonify({'error': 'data_model.py not found or failed to import.'}), 500

    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file uploaded.'}), 400

    # Parse CSV
    try:
        df = pd.read_csv(f)
    except Exception:
        try:
            f.seek(0)
            df = pd.read_csv(io.StringIO(f.read().decode('utf-8', errors='ignore')))
        except Exception as e:
            return jsonify({'error': f'Could not parse CSV: {e}'}), 400

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {'df': df.copy()}

    # ── run pipeline ──
    ia       = initial_analysis(df)
    ns       = numeric_stats(df)
    cats     = categorical_stats(df, top_n=10)
    corr_res = correlation_analysis(df)
    ts_res   = time_series_analysis(df)

    # ── distribution charts (capped for response size) ──
    charts, cat_charts = {}, {}
    try:
        plots, c_plots = distribution_plots(df)
        charts     = {c: fig_to_b64(fig) for c, fig in list(plots.items())[:12]}
        cat_charts = {c: fig_to_b64(fig) for c, fig in list(c_plots.items())[:6]}
    except Exception as e:
        print(f'Chart generation error: {e}')

    # ── correlation heatmap ──
    corr_heatmap = None
    if corr_res.get('corr') is not None:
        try:
            cmat = corr_res['corr']
            sz = max(5, min(12, len(cmat.columns)))
            fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
            if HAS_SNS:
                sns.heatmap(cmat, annot=len(cmat.columns) <= 10, fmt='.2f',
                            cmap='RdBu_r', ax=ax, vmin=-1, vmax=1, center=0,
                            linewidths=0.3)
            else:
                im = ax.imshow(cmat.values, cmap='RdBu_r', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax)
                ticks = range(len(cmat.columns))
                ax.set_xticks(ticks); ax.set_yticks(ticks)
                ax.set_xticklabels(cmat.columns, rotation=45, ha='right')
                ax.set_yticklabels(cmat.columns)
            ax.set_title('Correlation Heatmap', fontsize=12, pad=10)
            plt.tight_layout()
            corr_heatmap = fig_to_b64(fig)
        except Exception as e:
            print(f'Heatmap error: {e}')

    # ── serialise results ──
    missing_dict = {}
    if isinstance(ia.get('missing'), pd.Series):
        missing_dict = {str(k): int(v) for k, v in ia['missing'].items() if v > 0}

    dtypes_dict = {}
    if isinstance(ia.get('dtypes'), pd.Series):
        dtypes_dict = {str(k): str(v) for k, v in ia['dtypes'].items()}

    ns_rows = []
    if isinstance(ns, pd.DataFrame) and not ns.empty:
        for col_name, row in ns.iterrows():
            d = {'column': str(col_name)}
            for k, v in row.items():
                d[str(k)] = _clean(v)
            ns_rows.append(d)

    top_pairs = []
    for p in (corr_res.get('top_pairs') or [])[:15]:
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            try:
                top_pairs.append({
                    'col_a': str(p[0]), 'col_b': str(p[1]),
                    'corr': round(float(p[2]), 4),
                })
            except Exception:
                pass

    multicollinear = []
    for p in (corr_res.get('multicollinear') or []):
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            try:
                multicollinear.append({
                    'col_a': str(p[0]), 'col_b': str(p[1]),
                    'corr': round(float(p[2]), 4),
                })
            except Exception:
                pass

    sanitized_df = sanitize_for_display(df)

    return jsonify({
        'session_id':        session_id,
        'shape':             [int(df.shape[0]), int(df.shape[1])],
        'preview':           df_preview(sanitized_df),
        'dtypes':            dtypes_dict,
        'missing':           missing_dict,
        'fully_empty':       [str(c) for c in ia.get('fully_empty', [])],
        'more50':            [str(c) for c in ia.get('more50', [])],
        'duplicates':        int(ia.get('duplicates', 0)),
        'inconsistent':      _clean(ia.get('inconsistent', {})),
        'numeric_stats':     ns_rows,
        'categorical_stats': _clean(cats),
        'top_pairs':         top_pairs,
        'multicollinear':    multicollinear,
        'corr_heatmap':      corr_heatmap,
        'time_series':       _clean({'datetime_column': ts_res.get('datetime_column') if ts_res else None}),
        'charts':            charts,
        'cat_charts':        cat_charts,
    })


@app.route('/api/clean', methods=['POST'])
def clean():
    data = request.get_json() or {}
    sid = data.get('session_id')
    if not sid or sid not in _sessions:
        return jsonify({'error': 'Session not found — please re-upload your file.'}), 400
    df = _sessions[sid]['df']
    try:
        clean_df, summary = clean_data(df.copy())
        _sessions[sid]['clean_df'] = clean_df
        return jsonify({
            'summary': _clean(summary),
            'shape':   [int(clean_df.shape[0]), int(clean_df.shape[1])],
            'preview': df_preview(sanitize_for_display(clean_df)),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-insights', methods=['POST'])
def ai_insights_route():
    data = request.get_json() or {}
    sid         = data.get('session_id')
    model       = data.get('model', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
    temperature = float(data.get('temperature', 0.2))

    if not sid or sid not in _sessions:
        return jsonify({'error': 'Session not found — please re-upload your file.'}), 400

    sess    = _sessions[sid]
    working = sess.get('clean_df', sess.get('df'))

    try:
        ns       = numeric_stats(working)
        cats     = categorical_stats(working, top_n=5)
        corr_res = correlation_analysis(working)

        top_variances, ns_head = [], {}
        if isinstance(ns, pd.DataFrame) and not ns.empty:
            top_variances = list(ns.head(10).index)
            ns_head       = ns.head().to_dict()

        summary = {
            'shape':             working.shape,
            'missing':           working.isna().sum().sort_values(ascending=False),
            'correlation_top':   [list(x) for x in (corr_res.get('top_pairs') or [])[:10]],
            'top_variances':     top_variances,
            'categorical_sample':{k: list(v['top_categories'].items()) for k, v in cats.items()},
            'numeric_stats_head': ns_head,
        }
        prompt = build_ai_prompt(summary)
        return jsonify({'insights': call_openai(prompt, model=model, temperature=temperature)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
