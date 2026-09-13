"""
nlp_engine.py  —  VoiceIQ Pro: NLP core
Includes: sentiment, emotion, topic, escalation, aspect-based sentiment analysis (ABSA)
Zero external NLP dependencies. Pure Python + scikit-learn only.
"""

import re, math, json
from collections import Counter, defaultdict

# ─── Lexicons ──────────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "amazing","awesome","excellent","fantastic","great","good","happy","love",
    "wonderful","perfect","brilliant","outstanding","satisfied","impressed","helpful",
    "thank","thanks","appreciate","grateful","resolved","fixed","quick","fast",
    "efficient","professional","kind","friendly","pleased","delighted","superb",
    "best","top","nice","smooth","easy","clear","reliable","responsive","effective",
    "thorough","beautiful","enjoy","enjoyed","works","working","smooth","seamless",
}
NEGATIVE_WORDS = {
    "bad","terrible","horrible","awful","worst","hate","angry","furious","upset",
    "disappointed","frustrated","annoyed","disgusting","unacceptable","pathetic",
    "useless","incompetent","ridiculous","outraged","broken","failed","error",
    "problem","issue","wrong","missing","delayed","waiting","ignored","rude",
    "slow","confusing","complicated","unfair","scam","fraud","unreliable",
    "unavailable","disconnected","lost","stuck","blocked","denied","rejected",
    "overcharged","duplicate","mistake","incorrect","inaccurate","gone down",
    "deteriorated","declined","worsened","worthless","waste","overpriced",
}
INTENSIFIERS = {"very","extremely","absolutely","totally","completely","utterly","really","so","too"}
NEGATORS     = {"not","no","never","cannot","cant","wont","dont","isnt","wasnt","didnt","doesnt"}

EMOTION_PATTERNS = {
    "angry":     ["unacceptable","furious","outraged","lawsuit","fed up","last straw","gone too far","demand","sick of this","i am done","enough is enough","this is insane","absolutely ridiculous"],
    "frustrated":["frustrated","annoyed","again","still not","not resolved","keep calling","no response","ignored","third time","multiple times","keep happening","inconvenient","been waiting","why cant","why wont"],
    "happy":     ["thank you","thanks","amazing","excellent","love","great job","fantastic","impressed","happy","wonderful","perfect","brilliant","awesome","superb","delighted","appreciate","grateful","best experience"],
    "satisfied": ["resolved","sorted","fixed","done","completed","processed","received","approved","confirmed","good service","helpful","efficient","quick response","no issues","all good","working fine","pleased with"],
    "confused":  ["confused","unclear","don't understand","not sure","can you explain","what does this mean","why was i charged","how does","clarify","confusing","strange charge","help me understand"],
    "anxious":   ["urgent","asap","immediately","critical","emergency","time sensitive","need this now","worried","concern","important deadline","scared","nervous"],
}

TOPIC_TAXONOMY = {
    "Billing Problem":   ["charge","bill","invoice","payment","amount","fee","deduct","overcharged","duplicate","wrong amount","billed","statement","price","cost","charged twice","subscription fee"],
    "Refund Delay":      ["refund","money back","reimbursement","return","credit back","not received","still waiting","refund not","where is my refund","reimburs","get my money"],
    "Login / Access":    ["login","password","access","locked","account locked","reset password","cannot log","sign in","authentication","otp","two factor","forgot password","cant login"],
    "Claim Status":      ["claim","insurance claim","pending claim","claim status","filed","submitted","processing","approved","rejected","documents required","claim number"],
    "Technical Issue":   ["error","bug","not working","crashes","loading","portal","app","website","slow","broken","glitch","technical","page not","wont open","keeps crashing"],
    "Service Quality":   ["wait time","hold","queue","response time","slow service","nobody answers","no response","rude","unprofessional","45 minutes","long wait","waited","on hold","supervisor","real person","chatbot","speak to a","called multiple"],
    "Churn Risk":        ["cancel","terminate","close account","switching","competitor","leaving","unsubscribe","last time","done with","cancel subscription","discontinue","move to another","service quality has gone"],
    "Product Feedback":  ["feature","update","improve","suggestion","feedback","better","interface","design","new version","app update","functionality","user experience","ui","ux","report looks","love using"],
    "Account Update":    ["update","change address","name change","email update","phone number","profile","personal details","modify account","change my"],
    "General Praise":    ["thank you","great service","amazing team","wonderful","really helpful","best service","love your","keep it up","well done","impressed with","excellent service","empathetic","working perfectly","great job"],
}

ESCALATION_SIGNALS = [
    "supervisor","manager","lawyer","lawsuit","report you","regulatory","ombudsman",
    "i will post","social media","bbb","review","escalate","legal action",
    "last chance","final warning","done with you","cancel everything","file a complaint",
]

# ─── Aspect-Based Sentiment Analysis ──────────────────────────────────────────

ASPECT_LEXICON = {
    "agent_behavior": {
        "keywords": ["agent","representative","staff","team member","support person","operator","employee","spoke to","talked to"],
        "positive":  ["helpful","kind","patient","professional","empathetic","polite","friendly","understanding","excellent","great","knowledgeable","efficient","quick"],
        "negative":  ["rude","dismissive","unhelpful","ignorant","impatient","slow","unprofessional","clueless","useless","incompetent","arrogant","transferred","repeated"],
    },
    "wait_time": {
        "keywords": ["wait","hold","queue","minutes","hours","on hold","waiting time","response time","long time"],
        "positive":  ["quick","fast","immediate","short wait","no wait","right away","instantly","promptly","within minutes"],
        "negative":  ["long","forever","too long","45 minutes","hours","whole day","never","kept waiting","eternity","ages"],
    },
    "resolution": {
        "keywords": ["resolved","fixed","solved","issue","problem","done","completed","answer","solution","help"],
        "positive":  ["resolved","fixed","solved","done","completed","answered","worked","success","helped","finally","great solution"],
        "negative":  ["not resolved","still broken","unresolved","no solution","didnt help","same problem","came back","recurring","failed"],
    },
    "billing_accuracy": {
        "keywords": ["bill","charge","invoice","payment","amount","fee","price","cost","statement"],
        "positive":  ["correct","accurate","fair","reasonable","expected","right amount","as agreed","no issues"],
        "negative":  ["wrong","incorrect","overcharged","duplicate","unexpected","unauthorized","extra","more than","inflated","error"],
    },
    "product_quality": {
        "keywords": ["app","website","platform","portal","system","tool","feature","interface","dashboard","product"],
        "positive":  ["great","easy","smooth","intuitive","fast","reliable","works well","love","excellent","improved","clear"],
        "negative":  ["broken","slow","confusing","crashed","error","bug","not working","terrible","outdated","clunky","down"],
    },
    "communication": {
        "keywords": ["email","notification","update","informed","communication","message","response","reply","contact"],
        "positive":  ["timely","clear","informative","proactive","helpful","detailed","regular","transparent","good updates"],
        "negative":  ["no response","ignored","no update","silent","unclear","confusing","late","never replied","no communication"],
    },
}


def analyze_aspects(text: str) -> list:
    """Aspect-Based Sentiment Analysis — returns which aspect is causing the sentiment."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    results = []
    for aspect, config in ASPECT_LEXICON.items():
        # Check if this aspect is mentioned
        if not any(kw in text_lower for kw in config["keywords"]):
            continue
        pos = sum(1 for w in config["positive"] if w in text_lower)
        neg = sum(1 for w in config["negative"] if w in text_lower)
        if pos == 0 and neg == 0:
            continue
        if pos > neg:
            sentiment = "positive"
        elif neg > pos:
            sentiment = "negative"
        else:
            sentiment = "mixed"
        confidence = min(95, 40 + max(pos, neg) * 20)
        results.append({
            "aspect": aspect.replace("_", " ").title(),
            "sentiment": sentiment,
            "confidence": int(confidence),
        })
    return results


# ─── Core functions ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.findall(r'\b[a-z]+\b', text.lower())

def analyze_sentiment(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"label": "neutral", "score": 0.0, "confidence": 50}
    tokens = _tokenize(text)
    pos = sum(1 for w in tokens if w in POSITIVE_WORDS)
    neg = sum(1 for w in tokens if w in NEGATIVE_WORDS)
    for i, t in enumerate(tokens):
        if t in INTENSIFIERS and i + 1 < len(tokens):
            nw = tokens[i + 1]
            if nw in POSITIVE_WORDS: pos += 0.5
            elif nw in NEGATIVE_WORDS: neg += 0.5
    for i, t in enumerate(tokens):
        if t in NEGATORS:
            for j in range(i + 1, min(i + 4, len(tokens))):
                if tokens[j] in POSITIVE_WORDS: pos -= 1; neg += 1
                elif tokens[j] in NEGATIVE_WORDS: neg -= 1; pos += 1
    total = pos + neg
    if total == 0:
        return {"label": "neutral", "score": 0.0, "confidence": 50}
    score = (pos - neg) / (pos + neg + 1)
    if score > 0.1:
        label = "positive"; confidence = min(95, int(50 + score * 50))
    elif score < -0.1:
        label = "negative"; confidence = min(95, int(50 + abs(score) * 50))
    else:
        label = "neutral"; confidence = 50
    return {"label": label, "score": round(score, 3), "confidence": confidence}

def detect_emotion(text: str) -> dict:
    if not isinstance(text, str):
        return {"label": "neutral", "confidence": 0}
    text_lower = text.lower()
    scores = {emo: sum(1 for p in pats if p in text_lower) for emo, pats in EMOTION_PATTERNS.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return {"label": "neutral", "confidence": 40}
    best = max(scores, key=scores.get)
    return {"label": best, "confidence": min(95, 40 + scores[best] * 18)}

def detect_topic(text: str) -> dict:
    if not isinstance(text, str):
        return {"label": "General", "confidence": 30}
    text_lower = text.lower()
    scores = {t: sum(1 for kw in kws if kw in text_lower) for t, kws in TOPIC_TAXONOMY.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return {"label": "General", "confidence": 30}
    best = max(scores, key=scores.get)
    return {"label": best, "confidence": min(95, 40 + scores[best] * 15)}

def escalation_score(text: str, sentiment: dict, emotion: dict) -> dict:
    score = 0; reasons = []
    text_lower = (text or "").lower()
    if sentiment["label"] == "negative":
        score += max(10, int(abs(sentiment["score"]) * 40)); reasons.append("negative sentiment")
    emo_map = {"angry": 32, "frustrated": 20, "anxious": 12}
    if emotion["label"] in emo_map:
        score += emo_map[emotion["label"]]; reasons.append(f"{emotion['label']} emotion")
    for sig in ESCALATION_SIGNALS:
        if sig in text_lower: score += 20; reasons.append("escalation language"); break
    for kw in ["cancel","leaving","done with","last time","switch","discontinue","close account"]:
        if kw in text_lower: score += 18; reasons.append("churn signal"); break
    for word in ["again","third time","multiple","keep calling","still not","same issue"]:
        if word in text_lower: score += 12; reasons.append("repeat complaint"); break
    for ph in ["absolutely unacceptable","extremely disappointed","fed up","gone too far","last straw","am done","sick of this"]:
        if ph in text_lower: score += 15; reasons.append("strong negative"); break
    for kw in ["urgent","asap","immediately","right now","today","deadline"]:
        if kw in text_lower: score += 8; reasons.append("time pressure"); break
    score = min(score, 100)
    level = "HIGH" if score >= 55 else "MEDIUM" if score >= 18 else "LOW"
    return {"score": score, "level": level, "reasons": list(set(reasons))}

def analyze_row(text: str) -> dict:
    sent  = analyze_sentiment(text)
    emo   = detect_emotion(text)
    topic = detect_topic(text)
    esc   = escalation_score(text, sent, emo)
    aspects = analyze_aspects(text)
    return {
        "sentiment": sent["label"],
        "sentiment_score": sent["score"],
        "sentiment_confidence": sent["confidence"],
        "emotion": emo["label"],
        "emotion_confidence": emo["confidence"],
        "topic": topic["label"],
        "topic_confidence": topic["confidence"],
        "escalation_score": esc["score"],
        "escalation_level": esc["level"],
        "escalation_reasons": ", ".join(esc["reasons"]),
        "aspects": json.dumps(aspects),
    }

def generate_insights(df) -> list:
    import pandas as pd
    insights = []
    n = len(df)
    if n == 0: return []
    neg_pct      = (df["sentiment"] == "negative").mean() * 100
    pos_pct      = (df["sentiment"] == "positive").mean() * 100
    high_esc_pct = (df["escalation_level"] == "HIGH").mean() * 100

    if neg_pct > 50:
        insights.append({"level":"critical","icon":"🚨","title":"High Negative Sentiment",
            "body":f"{neg_pct:.0f}% of {n} conversations are negative. Immediate review recommended."})
    elif pos_pct > 55:
        insights.append({"level":"good","icon":"✅","title":"Healthy Customer Satisfaction",
            "body":f"{pos_pct:.0f}% positive sentiment — customers are generally satisfied."})

    top_topic = df["topic"].value_counts().idxmax()
    top_pct   = df["topic"].value_counts().max() / n * 100
    insights.append({"level":"high","icon":"📌","title":f"Top Issue: {top_topic}",
        "body":f"{top_topic} drives {top_pct:.0f}% of conversations. Consider proactive resolution."})

    if high_esc_pct > 18:
        insights.append({"level":"critical","icon":"⚠️","title":"Elevated Escalation Risk",
            "body":f"{high_esc_pct:.0f}% of interactions are HIGH risk. Increase supervisor availability."})

    top_emotion = df["emotion"].value_counts().idxmax()
    emo_pct     = df["emotion"].value_counts().max() / n * 100
    insights.append({"level":"medium","icon":"😤","title":f"Dominant Emotion: {top_emotion.title()}",
        "body":f"{top_emotion.title()} is most frequent at {emo_pct:.0f}% of interactions."})

    if "agent" in df.columns and df["agent"].nunique() > 1 and "csat_score" in df.columns:
        best = df.groupby("agent")["csat_score"].mean().idxmax()
        best_score = df.groupby("agent")["csat_score"].mean().max()
        insights.append({"level":"good","icon":"⭐","title":f"Top Agent: {best}",
            "body":f"{best} leads with avg CSAT {best_score:.1f}/5.0. Use as a coaching model."})

    churn_count = (df["topic"] == "Churn Risk").sum()
    if churn_count > 0:
        insights.append({"level":"critical","icon":"🔴","title":"Churn Risk Signals",
            "body":f"{churn_count} customers ({churn_count/n*100:.1f}%) expressed cancellation intent. Trigger retention workflows."})

    if "channel" in df.columns and df["channel"].nunique() > 1:
        ch_neg    = df.groupby("channel").apply(lambda x: (x["sentiment"]=="negative").mean()*100)
        worst_ch  = ch_neg.idxmax()
        insights.append({"level":"medium","icon":"📡","title":f"Weakest Channel: {worst_ch}",
            "body":f"{worst_ch} has the highest negative rate at {ch_neg.max():.0f}%. Review support quality."})

    return insights


def genai_recommendations(df, threshold_esc=55, threshold_neg=50) -> list:
    """
    GenAI-style real-time recommendations for management to balance Service Level.
    Returns actionable recommendation strings with urgency levels.
    """
    recs = []
    n = len(df)
    if n == 0: return recs

    neg_pct = (df["sentiment"]=="negative").mean()*100
    high_esc = (df["escalation_level"]=="HIGH").mean()*100
    churn    = (df["topic"]=="Churn Risk").sum()
    top_topic = df["topic"].value_counts().idxmax()

    if high_esc > 25:
        recs.append({"urgency":"critical","rec":"🚨 Deploy 2+ additional supervisors immediately — HIGH escalation rate exceeds safe threshold.",
            "impact":"Reduces escalation-to-supervisor wait time by ~40%","action":"Staff Reallocation"})
    if neg_pct > threshold_neg:
        recs.append({"urgency":"high","rec":f"📣 Trigger proactive callback campaign for {top_topic} cases — negative sentiment is at {neg_pct:.0f}%.",
            "impact":"Expected 15–20% reduction in repeat contacts","action":"Proactive Outreach"})
    if churn > 5:
        recs.append({"urgency":"critical","rec":f"🔴 Route {churn} churn-risk customers to senior retention specialists immediately.",
            "impact":"Retention uplift of ~30% with specialist handling","action":"Retention Workflow"})
    if "shift" in df.columns:
        shift_neg = df.groupby("shift").apply(lambda x:(x["sentiment"]=="negative").mean()*100)
        if shift_neg.max() > 60:
            worst = shift_neg.idxmax()
            recs.append({"urgency":"high","rec":f"🌙 Add {worst} shift capacity — sentiment is {shift_neg.max():.0f}% negative during this window.",
                "impact":"SL improvement of 8–12% by matching demand","action":"Schedule Optimization"})
    if "channel" in df.columns:
        ch_vol = df["channel"].value_counts()
        dominant = ch_vol.idxmax()
        if ch_vol.max()/n > 0.5:
            recs.append({"urgency":"medium","rec":f"📞 Redistribute load from {dominant} — it carries {ch_vol.max()/n*100:.0f}% of all volume. Enable omnichannel deflection.",
                "impact":"Reduces {dominant} queue by ~25%","action":"Channel Deflection"})

    if not recs:
        recs.append({"urgency":"low","rec":"✅ All metrics within acceptable ranges. Continue monitoring.",
            "impact":"Maintain current SL","action":"Monitor"})
    return recs
