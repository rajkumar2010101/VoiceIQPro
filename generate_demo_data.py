import csv, random, os
from datetime import datetime, timedelta
random.seed(42)
CONVS=[
    ("I've been waiting 3 weeks for my refund! Nobody answers calls.","negative","angry","Refund Delay"),
    ("Thank you so much! Issue resolved in minutes. Really impressed.","positive","happy","General Praise"),
    ("My account got locked and I can't login. Need this fixed urgently.","negative","frustrated","Login / Access"),
    ("Billing amount is wrong again. This is the third time this month!","negative","angry","Billing Problem"),
    ("Just checking when my claim will be processed. No urgency.","neutral","neutral","Claim Status"),
    ("Your agent was incredibly empathetic and helpful. Excellent service.","positive","satisfied","General Praise"),
    ("The portal keeps loading but nothing happens. Very frustrating.","negative","frustrated","Technical Issue"),
    ("Confused about the charges on my last invoice. Can someone explain?","neutral","confused","Billing Problem"),
    ("Everything is working perfectly! I love using your platform.","positive","happy","General Praise"),
    ("Called 5 times about the same issue. I am done waiting.","negative","angry","Service Quality"),
    ("New app update is great. So much easier to navigate now.","positive","satisfied","Product Feedback"),
    ("My claim has been pending 2 months. I need answers or I will escalate.","negative","angry","Claim Status"),
    ("I want to cancel my subscription. Service quality has gone down.","negative","frustrated","Churn Risk"),
    ("Your chatbot is useless. I need a real person right now.","negative","angry","Technical Issue"),
    ("Submitted claim 3 days ago and already got it resolved. Super fast!","positive","happy","Claim Status"),
    ("Charged twice. Please refund the duplicate charge immediately.","negative","frustrated","Billing Problem"),
    ("Can you help me update my address? No rush at all.","neutral","neutral","Account Update"),
    ("Refund process is confusing. Don't know what documents to submit.","neutral","confused","Refund Delay"),
    ("I need to speak to a supervisor right now. This has gone too far.","negative","angry","Service Quality"),
    ("Refund was processed quickly. Very satisfied with how it was handled.","positive","happy","Refund Delay"),
    ("Password reset email never arrived. Been trying for hours.","negative","frustrated","Login / Access"),
    ("Reasonable service. Nothing outstanding but got the job done.","neutral","neutral","General Praise"),
    ("I'm extremely disappointed. This is my last time using your services.","negative","angry","Churn Risk"),
    ("Your team helped me file a complex claim with ease. Really grateful.","positive","satisfied","Claim Status"),
    ("Waiting time on calls is too long. Waited 45 minutes today.","negative","frustrated","Service Quality"),
]
AGENTS=["Agent_Priya","Agent_Rahul","Agent_Sara","Agent_Dev","Agent_Meena"]
SHIFTS=["Morning","Afternoon","Night"]
CHANNELS=["Phone","Chat","Email"]
rows=[]
base=datetime(2024,1,1)
for i in range(300):
    msg,sent,emo,topic=random.choice(CONVS)
    date=base+timedelta(days=random.randint(0,89))
    agent=random.choice(AGENTS); shift=random.choice(SHIFTS); channel=random.choice(CHANNELS)
    csat=random.randint(4,5) if sent=="positive" else (random.randint(2,3) if sent=="neutral" else random.randint(1,2))
    resolved=True if sent=="positive" else random.random()>0.4
    rows.append([f"CONV_{1000+i}",date.strftime("%Y-%m-%d"),agent,shift,channel,msg,csat,resolved,random.randint(2,60),random.randint(3,30)])
os.makedirs("data",exist_ok=True)
with open("data/demo_conversations.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["conversation_id","date","agent","shift","channel","customer_message","csat_score","resolved","wait_time_mins","handle_time_mins"])
    w.writerows(rows)
print(f"Generated {len(rows)} demo rows")
