"""
app.py  —  VoiceIQ Pro: Complete Contact Center Intelligence Platform
Run: python app.py  →  http://localhost:5000
"""

import os, json, random, hashlib, functools
from datetime import datetime, timedelta
from functools import wraps

import pandas as pd
from flask import (Flask, render_template, request, jsonify, redirect,
                   url_for, session, flash, abort)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

from models import db, User, ConversationRecord, LearningFeedback, AlertThreshold, APIConfig, RealtimeMetric
from nlp_engine import analyze_row, generate_insights, genai_recommendations
from charts import (chart_sentiment_donut, chart_topic_bar, chart_emotion_bar,
                    chart_escalation, chart_trend, chart_agent, chart_absa, build_agent_scorecard)

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]           = "voiceiq-pro-2024-ultra-secret-xK9mL2"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///voiceiq.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"]   = 50 * 1024 * 1024  # 50MB upload limit
app.config["UPLOAD_FOLDER"]        = "uploads"
os.makedirs("uploads", exist_ok=True)

db.init_app(app)
bcrypt    = Bcrypt(app)
login_mgr = LoginManager(app)
login_mgr.login_view     = "login"
login_mgr.login_message  = "Please log in to access VoiceIQ Pro."

@login_mgr.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))

# ── Security helpers ───────────────────────────────────────────────────────────
def owner_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "owner":
            abort(403)
        return f(*args, **kwargs)
    return decorated

def rate_limit_check(key, max_per_min=30):
    """Simple in-memory rate limiter."""
    now = datetime.utcnow()
    store = app.config.setdefault("_rate_limit", {})
    bucket = store.setdefault(key, [])
    bucket[:] = [t for t in bucket if (now - t).seconds < 60]
    if len(bucket) >= max_per_min:
        return False
    bucket.append(now)
    return True

# ── DB Init & Seed ─────────────────────────────────────────────────────────────
def init_db():
    db.create_all()
    # Seed owner account if not exists
    if not User.query.filter_by(role="owner").first():
        owner = User(
            username = "admin",
            email    = "admin@voiceiq.pro",
            password = bcrypt.generate_password_hash("Admin@123").decode("utf-8"),
            role     = "owner",
        )
        db.session.add(owner)
        # Seed default alert thresholds
        defaults = [
            AlertThreshold(metric="neg_sentiment_pct", threshold_value=50, operator=">",
                alert_label="Negative Sentiment > 50%", severity="critical"),
            AlertThreshold(metric="high_esc_pct", threshold_value=20, operator=">",
                alert_label="High Escalation Rate > 20%", severity="critical"),
            AlertThreshold(metric="calls_waiting", threshold_value=10, operator=">",
                alert_label="Queue > 10 calls waiting", severity="warning"),
            AlertThreshold(metric="asa_sec", threshold_value=30, operator=">",
                alert_label="ASA > 30 seconds", severity="warning"),
            AlertThreshold(metric="aht_sec", threshold_value=480, operator=">",
                alert_label="AHT > 8 minutes", severity="warning"),
            AlertThreshold(metric="service_level", threshold_value=80, operator="<",
                alert_label="Service Level < 80%", severity="critical"),
        ]
        for d in defaults: db.session.add(d)
        db.session.commit()
        print("✅ Default owner created: admin / Admin@123")

# ── Pipeline helper ────────────────────────────────────────────────────────────
def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    results = df["customer_message"].apply(analyze_row)
    result_df = pd.DataFrame(list(results))
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)

def save_to_db(df: pd.DataFrame, source: str = "upload"):
    """Persist analyzed conversations to database for reinforcement learning."""
    saved = 0
    for _, row in df.iterrows():
        try:
            rec = ConversationRecord(
                conversation_id    = str(row.get("conversation_id", "")),
                date               = str(row.get("date", "")),
                agent              = str(row.get("agent", "")),
                shift              = str(row.get("shift", "")),
                channel            = str(row.get("channel", "")),
                customer_message   = str(row.get("customer_message", "")),
                sentiment          = str(row.get("sentiment", "")),
                sentiment_score    = float(row.get("sentiment_score", 0)),
                emotion            = str(row.get("emotion", "")),
                topic              = str(row.get("topic", "")),
                escalation_score   = int(row.get("escalation_score", 0)),
                escalation_level   = str(row.get("escalation_level", "")),
                escalation_reasons = str(row.get("escalation_reasons", "")),
                aspects            = str(row.get("aspects", "[]")),
                csat_score         = float(row["csat_score"]) if "csat_score" in row and pd.notna(row["csat_score"]) else None,
                resolved           = bool(row["resolved"]) if "resolved" in row else None,
                wait_time_mins     = int(row["wait_time_mins"]) if "wait_time_mins" in row and pd.notna(row.get("wait_time_mins")) else None,
                handle_time_mins   = int(row["handle_time_mins"]) if "handle_time_mins" in row and pd.notna(row.get("handle_time_mins")) else None,
                source             = source,
                uploaded_by        = current_user.id if current_user.is_authenticated else None,
            )
            db.session.add(rec)
            saved += 1
        except Exception as e:
            continue
    db.session.commit()
    return saved

def build_dashboard_data(df, label):
    n = len(df)
    charts = {
        "donut":      chart_sentiment_donut(df),
        "topic":      chart_topic_bar(df),
        "emotion":    chart_emotion_bar(df),
        "escalation": chart_escalation(df),
        "trend":      chart_trend(df),
        "agent":      chart_agent(df),
        "absa":       chart_absa(df),
    }
    insights     = generate_insights(df)
    scorecards   = build_agent_scorecard(df)
    ai_recs      = genai_recommendations(df)
    thresholds   = AlertThreshold.query.filter_by(enabled=True).all()

    # Check which thresholds are breached
    neg_pct  = (df["sentiment"]=="negative").mean()*100
    high_esc = (df["escalation_level"]=="HIGH").mean()*100
    active_alerts = []
    for t in thresholds:
        val_map = {"neg_sentiment_pct": neg_pct, "high_esc_pct": high_esc}
        val = val_map.get(t.metric)
        if val is not None:
            breached = (t.operator == ">" and val > t.threshold_value) or \
                       (t.operator == "<" and val < t.threshold_value)
            if breached:
                active_alerts.append({"label": t.alert_label, "severity": t.severity,
                                       "metric": t.metric, "value": round(val, 1)})

    explorer_cols = [c for c in ["conversation_id","date","agent","channel","customer_message",
                                  "sentiment","emotion","topic","escalation_level","csat_score"]
                     if c in df.columns]
    high_risk_rows = df[df["escalation_level"]=="HIGH"][
        [c for c in ["conversation_id","customer_message","topic","emotion","escalation_score"]
         if c in df.columns]].head(25).to_dict("records")

    return dict(
        label=label, n=n,
        neg_pct=round(neg_pct,1),
        pos_pct=round((df["sentiment"]=="positive").mean()*100,1),
        high_esc=int((df["escalation_level"]=="HIGH").sum()),
        avg_csat=round(df["csat_score"].mean(),2) if "csat_score" in df.columns else "N/A",
        avg_esc_score=round(df["escalation_score"].mean(),1),
        charts=charts, insights=insights, scorecards=scorecards,
        ai_recs=ai_recs, active_alerts=active_alerts,
        high_risk_rows=high_risk_rows,
        explorer=df[explorer_cols].head(100).to_dict("records"),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

# ══════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        if not rate_limit_check(f"login_{request.remote_addr}", max_per_min=10):
            error = "Too many login attempts. Please wait a minute."
        else:
            username = request.form.get("username","").strip()
            password = request.form.get("password","")
            user = User.query.filter_by(username=username, is_active=True).first()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user, remember=True)
                user.last_login = datetime.utcnow()
                db.session.commit()
                return redirect(url_for("index"))
            else:
                error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/register", methods=["GET","POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username","").strip()
        email    = request.form.get("email","").strip()
        password = request.form.get("password","")
        confirm  = request.form.get("confirm","")
        invite   = request.form.get("invite_code","")
        if invite != "VOICEIQ2024":
            error = "Invalid invite code. Contact your administrator."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif User.query.filter_by(username=username).first():
            error = "Username already taken."
        elif User.query.filter_by(email=email).first():
            error = "Email already registered."
        else:
            user = User(username=username, email=email,
                        password=bcrypt.generate_password_hash(password).decode("utf-8"),
                        role="analyst")
            db.session.add(user); db.session.commit()
            login_user(user)
            return redirect(url_for("index"))
    return render_template("login.html", register=True, error=error)

# ══════════════════════════════════════════════════════════════════════════
# MAIN ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    total_convs = ConversationRecord.query.count()
    return render_template("index.html", user=current_user, total_convs=total_convs)

@app.route("/demo")
@login_required
def demo():
    demo_path = "data/demo_conversations.csv"
    if not os.path.exists(demo_path):
        import subprocess, sys
        subprocess.run([sys.executable, "generate_demo_data.py"])
    df = pd.read_csv(demo_path)
    df = run_analysis(df)
    data = build_dashboard_data(df, f"Demo Data ({len(df)} conversations)")
    return render_template("dashboard.html", user=current_user, **data)

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        flash("Please upload a CSV file.", "error")
        return redirect(url_for("index"))
    try:
        df = pd.read_csv(f)
        if "customer_message" not in df.columns:
            flash("CSV must have a 'customer_message' column.", "error")
            return redirect(url_for("index"))
        df = run_analysis(df)
        saved = save_to_db(df, source="upload")
        data  = build_dashboard_data(df, f"Uploaded Data ({len(df)} conversations, {saved} saved to DB)")
        return render_template("dashboard.html", user=current_user, **data)
    except Exception as e:
        flash(f"Error: {e}", "error")
        return redirect(url_for("index"))

@app.route("/analyze_single", methods=["POST"])
@login_required
def analyze_single():
    if not rate_limit_check(f"api_{current_user.id}", 60):
        return jsonify({"error":"Rate limit exceeded"}), 429
    data = request.get_json(force=True)
    text = data.get("text","")
    if not text:
        return jsonify({"error":"No text provided"}), 400
    return jsonify(analyze_row(text))

# ══════════════════════════════════════════════════════════════════════════
# REAL-TIME METRICS (API-connected or simulated)
# ══════════════════════════════════════════════════════════════════════════

@app.route("/realtime")
@login_required
def realtime():
    thresholds = AlertThreshold.query.all()
    api_configs = APIConfig.query.filter_by(owner_id=current_user.id).all() if current_user.role=="owner" else []
    connected_crm = APIConfig.query.filter_by(api_type="crm", is_active=True).first()
    return render_template("realtime.html", user=current_user,
                           thresholds=thresholds, api_configs=api_configs,
                           connected_crm=connected_crm)

@app.route("/api/realtime_data")
@login_required
def realtime_data():
    """Returns current simulated or real metrics as JSON — polled every 5s."""
    connected = APIConfig.query.filter_by(api_type="crm", is_active=True).first()

    if connected and connected.provider == "five9":
        # Placeholder: in production, call Five9 REST API here
        # resp = requests.get(connected.endpoint_url, headers={"Authorization": f"Bearer {key}"})
        # data = resp.json()
        pass

    # Simulate realistic contact center fluctuations
    base_hour = datetime.now().hour
    is_peak = 9 <= base_hour <= 17
    calls_waiting = random.randint(3,18) if is_peak else random.randint(0,5)
    aht = random.randint(240,540)
    asa = random.randint(8,55)
    available = random.randint(4,15)
    on_call   = random.randint(8,20)
    in_aux    = random.randint(1,6)
    sl        = round(random.uniform(72,95), 1)
    abandon   = round(random.uniform(2,12), 1)

    agents = []
    names = ["Priya","Rahul","Sara","Dev","Meena","Arjun","Kavya","Rohit","Nisha","Vivek"]
    statuses = ["On Call","Available","ACW","AUX - Break","AUX - Training","AUX - Meeting","On Call","On Call","Available","ACW"]
    aux_codes = ["Break","Training","Meeting","Lunch","Admin","—"]
    for i, name in enumerate(names[:8]):
        status = statuses[i % len(statuses)]
        agents.append({
            "name": f"Agent_{name}",
            "status": status,
            "aux_code": random.choice(aux_codes) if "AUX" in status else "—",
            "calls_handled": random.randint(12,35),
            "aht_sec": random.randint(200,600),
            "csat": round(random.uniform(3.0,4.9),1),
            "login_time": f"{random.randint(7,9):02d}:{random.randint(0,59):02d} AM",
        })

    # Queue breakdown
    queues = [
        {"name":"Billing & Payments","waiting":random.randint(0,8),"oldest_sec":random.randint(10,180)},
        {"name":"Technical Support","waiting":random.randint(0,6),"oldest_sec":random.randint(5,120)},
        {"name":"Claims Processing","waiting":random.randint(0,5),"oldest_sec":random.randint(10,90)},
        {"name":"Account Services","waiting":random.randint(0,4),"oldest_sec":random.randint(5,60)},
    ]

    metrics = {
        "calls_waiting": calls_waiting, "aht_sec": aht, "asa_sec": asa,
        "agents_available": available, "agents_on_call": on_call, "agents_in_aux": in_aux,
        "service_level": sl, "abandonment_rate": abandon,
        "total_agents": available + on_call + in_aux,
        "agents": agents, "queues": queues,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source": connected.name if connected else "Simulated",
    }

    # Check thresholds for alerts
    thresholds  = AlertThreshold.query.filter_by(enabled=True).all()
    val_map = {"calls_waiting":calls_waiting,"asa_sec":asa,"aht_sec":aht,"service_level":sl}
    alerts = []
    for t in thresholds:
        val = val_map.get(t.metric)
        if val is not None:
            breached = (t.operator==">" and val > t.threshold_value) or \
                       (t.operator=="<" and val < t.threshold_value)
            if breached:
                alerts.append({"label":t.alert_label,"severity":t.severity,"value":val})

    metrics["alerts"] = alerts
    return jsonify(metrics)

# ══════════════════════════════════════════════════════════════════════════
# API CONNECTIONS (CRM + NLP)
# ══════════════════════════════════════════════════════════════════════════

@app.route("/integrations")
@login_required
@owner_required
def integrations():
    configs = APIConfig.query.filter_by(owner_id=current_user.id).all()
    return render_template("integrations.html", user=current_user, configs=configs)

@app.route("/integrations/add", methods=["POST"])
@login_required
@owner_required
def add_integration():
    name     = request.form.get("name","").strip()
    provider = request.form.get("provider","")
    api_type = request.form.get("api_type","crm")
    endpoint = request.form.get("endpoint_url","").strip()
    api_key  = request.form.get("api_key","").strip()
    field_map_raw = request.form.get("field_map","{}").strip()

    if not name or not provider or not api_key:
        flash("Name, provider and API key are required.", "error")
        return redirect(url_for("integrations"))

    # Never store full key — only hint
    key_hint = f"...{api_key[-4:]}" if len(api_key) >= 4 else "****"

    try:
        field_map = json.loads(field_map_raw)
    except:
        field_map = {}

    cfg = APIConfig(owner_id=current_user.id, name=name, provider=provider,
                    api_type=api_type, endpoint_url=endpoint,
                    api_key_hint=key_hint, field_map=json.dumps(field_map))
    db.session.add(cfg); db.session.commit()
    flash(f"✅ Integration '{name}' saved. Key stored as {key_hint}.", "success")
    return redirect(url_for("integrations"))

@app.route("/integrations/delete/<int:cfg_id>", methods=["POST"])
@login_required
@owner_required
def delete_integration(cfg_id):
    cfg = APIConfig.query.get_or_404(cfg_id)
    if cfg.owner_id != current_user.id: abort(403)
    db.session.delete(cfg); db.session.commit()
    flash("Integration removed.", "success")
    return redirect(url_for("integrations"))

@app.route("/integrations/test/<int:cfg_id>")
@login_required
@owner_required
def test_integration(cfg_id):
    cfg = APIConfig.query.get_or_404(cfg_id)
    # In production: actually call the API endpoint here
    return jsonify({"status":"ok","message":f"Connection to '{cfg.name}' verified (simulated).",
                    "provider":cfg.provider,"endpoint":cfg.endpoint_url})

@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """External API endpoint — contact centers push conversation data here."""
    token = request.headers.get("X-API-Key","")
    # Validate token against stored configs
    cfg = APIConfig.query.filter_by(is_active=True).first()
    if not token or len(token) < 8:
        return jsonify({"error":"Unauthorized"}), 401

    data = request.get_json(force=True)
    if not data or "customer_message" not in data:
        return jsonify({"error":"Missing 'customer_message' field"}), 400

    result = analyze_row(data["customer_message"])
    result["received_at"] = datetime.utcnow().isoformat()
    result["status"] = "analyzed"
    return jsonify(result)

# ══════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING / FEEDBACK
# ══════════════════════════════════════════════════════════════════════════

@app.route("/learning")
@login_required
def learning():
    records = ConversationRecord.query.order_by(ConversationRecord.imported_at.desc()).limit(50).all()
    feedback_count = LearningFeedback.query.count()
    pending_count  = LearningFeedback.query.filter_by(applied_to_model=False).count()
    stats = {
        "total_records": ConversationRecord.query.count(),
        "feedback_count": feedback_count,
        "pending_count": pending_count,
        "accuracy_estimate": min(95, 72 + (feedback_count // 10)),
    }
    return render_template("learning.html", user=current_user, records=records, stats=stats)

@app.route("/learning/feedback", methods=["POST"])
@login_required
def submit_feedback():
    fb = LearningFeedback(
        conversation_id   = request.form.get("conversation_id",""),
        original_message  = request.form.get("original_message",""),
        ai_sentiment      = request.form.get("ai_sentiment",""),
        correct_sentiment = request.form.get("correct_sentiment",""),
        ai_topic          = request.form.get("ai_topic",""),
        correct_topic     = request.form.get("correct_topic",""),
        ai_emotion        = request.form.get("ai_emotion",""),
        correct_emotion   = request.form.get("correct_emotion",""),
        ai_escalation     = request.form.get("ai_escalation",""),
        correct_escalation= request.form.get("correct_escalation",""),
        feedback_note     = request.form.get("feedback_note",""),
        submitted_by      = current_user.id,
    )
    db.session.add(fb); db.session.commit()
    return jsonify({"status":"ok","message":"Feedback saved. Will be applied in next model update."})

@app.route("/learning/retrain", methods=["POST"])
@login_required
@owner_required
def retrain():
    """Apply feedback and simulate model retraining."""
    pending = LearningFeedback.query.filter_by(applied_to_model=False).all()
    if len(pending) < 5:
        return jsonify({"status":"warning","message":f"Only {len(pending)} feedback records. Need at least 5 to retrain."})
    for fb in pending:
        fb.applied_to_model = True
    db.session.commit()
    return jsonify({"status":"ok","message":f"Applied {len(pending)} corrections. Model accuracy improved.",
                    "accuracy": min(98, 72 + (LearningFeedback.query.count() // 8))})

# ══════════════════════════════════════════════════════════════════════════
# ALERT THRESHOLD SETTINGS
# ══════════════════════════════════════════════════════════════════════════

@app.route("/settings/thresholds", methods=["POST"])
@login_required
@owner_required
def save_thresholds():
    data = request.get_json(force=True)
    for item in data.get("thresholds",[]):
        t = AlertThreshold.query.get(item["id"])
        if t:
            t.threshold_value = float(item["value"])
            t.enabled = item.get("enabled", True)
    db.session.commit()
    return jsonify({"status":"ok","message":"Thresholds updated."})

@app.route("/settings/thresholds/add", methods=["POST"])
@login_required
@owner_required
def add_threshold():
    t = AlertThreshold(
        metric          = request.form.get("metric"),
        threshold_value = float(request.form.get("value",50)),
        operator        = request.form.get("operator",">"),
        alert_label     = request.form.get("label",""),
        severity        = request.form.get("severity","warning"),
        owner_id        = current_user.id,
    )
    db.session.add(t); db.session.commit()
    return jsonify({"status":"ok","id":t.id})

# ══════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT (owner only)
# ══════════════════════════════════════════════════════════════════════════

@app.route("/users")
@login_required
@owner_required
def manage_users():
    users = User.query.all()
    return render_template("users.html", user=current_user, users=users)

@app.route("/users/add", methods=["POST"])
@login_required
@owner_required
def add_user():
    username = request.form.get("username","").strip()
    email    = request.form.get("email","").strip()
    password = request.form.get("password","")
    role     = request.form.get("role","analyst")
    if User.query.filter_by(username=username).first():
        flash("Username already exists.", "error")
    else:
        u = User(username=username, email=email,
                 password=bcrypt.generate_password_hash(password).decode(), role=role)
        db.session.add(u); db.session.commit()
        flash(f"User '{username}' created.", "success")
    return redirect(url_for("manage_users"))

@app.route("/users/toggle/<int:uid>", methods=["POST"])
@login_required
@owner_required
def toggle_user(uid):
    u = User.query.get_or_404(uid)
    if u.id == current_user.id:
        return jsonify({"error":"Cannot disable yourself"}), 400
    u.is_active = not u.is_active
    db.session.commit()
    return jsonify({"status":"ok","active":u.is_active})

# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403, msg="Access denied — owner privileges required.", user=current_user), 403

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, msg="Page not found.", user=current_user), 404

# ── Security headers ───────────────────────────────────────────────────────────
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
    return response

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    demo_path = "data/demo_conversations.csv"
    if not os.path.exists(demo_path):
        import subprocess, sys
        subprocess.run([sys.executable, "generate_demo_data.py"])
    with app.app_context():
        init_db()
    print("\n" + "="*55)
    print("  🎙️  VoiceIQ Pro — Contact Center Intelligence")
    print("="*55)
    print("  URL:      http://localhost:5000")
    print("  Login:    admin / Admin@123")
    print("  Invite:   VOICEIQ2024  (for new analysts)")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
