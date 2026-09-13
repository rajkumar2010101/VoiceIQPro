"""
start.py — VoiceIQ Pro one-click launcher
Run: python start.py
"""
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"--break-system-packages","--quiet"])

packages = {"flask":"Flask","flask_login":"flask-login","flask_sqlalchemy":"flask-sqlalchemy",
            "flask_bcrypt":"flask-bcrypt","pandas":"pandas","numpy":"numpy",
            "matplotlib":"matplotlib","sklearn":"scikit-learn","plotly":"plotly","requests":"requests"}
print("🔍 Checking packages...")
for imp, pkg in packages.items():
    try: __import__(imp)
    except ImportError: print(f"  Installing {pkg}..."); install(pkg)

os.makedirs("data", exist_ok=True)
if not os.path.exists("data/demo_conversations.csv"):
    print("📝 Generating demo data...")
    subprocess.run([sys.executable,"generate_demo_data.py"])

from app import app
from models import db, User, AlertThreshold
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt(app)
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        owner = User(username="admin",email="admin@voiceiq.pro",
                     password=bcrypt.generate_password_hash("Admin@123").decode(),role="owner")
        db.session.add(owner)
        for m,v,op,lbl,sev in [
            ("neg_sentiment_pct",50,">","Negative Sentiment > 50%","critical"),
            ("high_esc_pct",20,">","High Escalation Rate > 20%","critical"),
            ("calls_waiting",10,">","Queue > 10 calls","warning"),
            ("asa_sec",30,">","ASA > 30 seconds","warning"),
            ("aht_sec",480,">","AHT > 8 minutes","warning"),
            ("service_level",80,"<","Service Level < 80%","critical"),
        ]:
            db.session.add(AlertThreshold(metric=m,threshold_value=v,operator=op,alert_label=lbl,severity=sev))
        db.session.commit()
        print("✅ Admin account created")

print("\n" + "="*55)
print("  🎙️  VoiceIQ Pro — Contact Center Intelligence")
print("="*55)
print("  URL:      http://localhost:5000")
print("  Login:    admin  /  Admin@123")
print("  Invite:   VOICEIQ2024  (for new analysts)")
print("="*55 + "\n")
app.run(debug=True, port=5000)
