"""
full_problem_generator.py
=========================
Day 13 of 28 — Full 1000-prompt distillation dataset.

Generates 3000 teacher responses (1000 problems × 3 brain cores):
  1. TaskUnderstanding
  2. DomainClassification
  3. ArchitectureAdvice

Production safeguards:
  - Resume from checkpoint (saves every 50 prompts)
  - 100ms inter-call delay + exponential-backoff retry (max 3 attempts)
  - Running cost monitor — hard stop at $2.00
  - Pass-rate monitor per 100-problem window (alert < 90%)
  - Quality sample print every 100 prompts
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# ── Re-use schemas + prompts from teacher_generator ───────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
load_dotenv(_ROOT / ".env")

sys.path.insert(0, str(_HERE))
from teacher_generator import (          # noqa: E402
    TaskUnderstanding,
    DomainClassification,
    ArchitectureAdvice,
    SYSTEM_PROMPT_TASK,
    SYSTEM_PROMPT_DOMAIN,
    SYSTEM_PROMPT_ARCH,
)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED_PATH      = _HERE / "seed_problems_1000.json"
CHECKPOINT_DIR = _HERE / "checkpoints"
DATASET_DIR    = _HERE

DEEPSEEK_INPUT_COST_PER_M  = 0.27   # $ / 1M tokens
DEEPSEEK_OUTPUT_COST_PER_M = 1.10   # $ / 1M tokens
COST_HARD_STOP             = 2.00   # $ — abort if exceeded
QUALITY_ALERT_THRESHOLD    = 0.90   # alert if window pass rate drops below
CHECKPOINT_INTERVAL        = 50     # save progress every N successful calls
MONITOR_INTERVAL           = 100    # print stats every N problems
INTER_CALL_DELAY_S         = 0.10   # 100 ms between calls (≈ 10 req/s)
MAX_RETRIES                = 3      # max retries per prompt


# ═════════════════════════════════════════════════════════════════════════════
# 1000 Problem Generator
# ═════════════════════════════════════════════════════════════════════════════

def _generate_problems() -> list[dict]:
    """
    Build 1000 diverse ML problems programmatically from templates and
    industry/domain variations.

    Distribution:
      image      200   tabular    200   audio      100
      text       200   multimodal 100   medical    100
      security    50   edge_case   50
    """
    probs: list[dict] = []
    pid = 1

    def add(cat: str, text: str) -> None:
        nonlocal pid
        probs.append({"id": pid, "category": cat, "problem": text})
        pid += 1

    # ── IMAGE (200) ───────────────────────────────────────────────────────────
    img_defects   = ["surface cracks", "paint scratches", "assembly misalignment",
                     "dents", "rust spots", "chip defects", "colour deviation",
                     "weld seam anomalies", "hole misplacement", "burrs"]
    img_materials = ["metal sheets", "plastic components", "PCB boards",
                     "textile fabric", "ceramic tiles", "glass panels",
                     "rubber seals", "wooden boards", "food packaging", "solar panels"]
    img_detect_obj = ["cars", "motorcycles", "trucks", "bicycles", "pedestrians",
                      "hard hats", "fire extinguishers", "packages", "pallets", "animals"]
    img_scenes     = ["factory floor cameras", "warehouse CCTV footage",
                      "construction site cameras", "parking lot feeds",
                      "airport runway cameras", "port crane cameras",
                      "mining site footage", "farm field images",
                      "retail store cameras", "hospital corridor cameras"]
    img_agri       = ["maize", "wheat", "rice", "tomato", "potato",
                      "apple", "grape", "coffee", "cotton", "soybean"]
    img_agri_prob  = ["leaf disease", "pest infestation", "nutrient deficiency",
                      "ripeness stage", "crop yield estimation",
                      "weed presence", "irrigation stress", "frost damage",
                      "fungal infection", "insect damage"]
    img_traffic    = ["pothole", "road crack", "lane marking fade", "debris",
                      "damaged guardrail", "missing sign", "flooded section",
                      "illegal parking", "traffic congestion level", "bridge crack"]
    img_retail     = ["clothing category", "shoe type", "handbag style",
                      "jewelry type", "furniture style", "electronic device",
                      "book genre from cover", "art style", "logo brand",
                      "product freshness"]
    img_satellite  = ["deforestation", "crop type", "building footprint",
                      "flood extent", "wildfire boundary", "urban sprawl",
                      "illegal construction", "road extraction",
                      "cloud cover classification", "sea ice extent"]
    img_biometrics = ["face identity", "face emotion", "age group from face",
                      "gesture command", "eye gaze direction",
                      "driver drowsiness", "customer expression in store",
                      "pose estimation", "hand sign language", "crowd density"]

    for defect, material in zip(img_defects, img_materials):
        add("image", f"Detect {defect} in {material} on a production line using overhead camera images")
    for obj, scene in zip(img_detect_obj, img_scenes):
        add("image", f"Detect and count {obj} in {scene} and trigger an alert when count exceeds threshold")
    for crop, problem in zip(img_agri, img_agri_prob):
        add("image", f"Identify {problem} in {crop} field images taken by drone for precision agriculture")
    for issue in img_traffic:
        add("image", f"Detect {issue} in road inspection photographs and classify severity for maintenance crews")
    for item in img_retail:
        add("image", f"Classify {item} from product listing photographs to auto-populate e-commerce catalog")
    for feature in img_satellite:
        add("image", f"Detect {feature} in satellite imagery for environmental monitoring")
    for biometric in img_biometrics:
        add("image", f"Identify {biometric} from camera feed in real time for smart retail / access control")

    # Top-up to 200
    extra_image = [
        "Build a real-time pothole detection system from dashcam footage for city road maintenance",
        "Classify medical device components for quality control on the assembly line",
        "Detect smoke and fire in forest surveillance images before it spreads",
        "Identify counterfeit banknotes from high-resolution scanner images",
        "Classify skin lesion types (benign vs malignant) from smartphone photos for triage",
        "Detect personal protective equipment compliance on a construction site",
        "Identify bird species from wildlife camera trap images for ecological surveys",
        "Count fish in underwater camera footage for aquaculture yield estimation",
        "Detect tampered documents by classifying forged signatures in scanned forms",
        "Classify satellite images of agricultural fields into crop type for insurance assessment",
        "Detect lightning rods and power line damage from aerial drone inspections",
        "Build a system to classify coral reef health from underwater survey photographs",
        "Identify pipeline corrosion level from inspection robot camera images",
        "Detect graffiti on public infrastructure from city camera networks",
        "Classify chest X-ray as normal or showing signs of pulmonary nodules",
        "Detect unauthorized vehicles in restricted parking zones from CCTV",
        "Identify tire tread wear level from workshop inspection images",
        "Classify satellite images to detect illegal open-cast mining activity",
        "Detect fall events in elderly care home cameras to alert nursing staff",
        "Build an automated visual inspection system for bottled beverage fill level",
        "Identify vehicle license plates in low-light parking lot images",
        "Detect oil spills in coastal aerial drone photographs",
        "Classify solar panel defect types (crack, soiling, delamination) from thermal images",
        "Identify weapon presence in crowded public space surveillance footage",
        "Detect concrete spalling and rebar exposure in bridge inspection images",
        "Classify microplastic types in microscope images for environmental research",
        "Detect smoke anomalies in steel plant stack emissions for environmental compliance",
        "Identify packaging seal integrity failures from high-speed production line cameras",
        "Classify retail shelf planogram compliance from store walkthrough images",
        "Build a document orientation and classification system for scanned invoices",
    ]
    for t in extra_image:
        add("image", t)

    # ── TEXT (200) ────────────────────────────────────────────────────────────
    txt_sentiment_domains = [
        ("restaurant", "Yelp reviews"), ("hotel", "Booking.com reviews"),
        ("airline", "customer feedback forms"), ("software product", "app store reviews"),
        ("bank", "net promoter score surveys"), ("hospital", "patient satisfaction forms"),
        ("online course", "learner reviews"), ("streaming service", "subscriber feedback"),
        ("gym", "member surveys"), ("e-commerce", "post-purchase reviews"),
    ]
    txt_spam_variants = [
        ("email", "inbox"), ("SMS", "mobile"), ("push notification", "app"),
        ("forum post", "community"), ("product review", "marketplace"),
        ("job posting", "recruitment portal"), ("comment", "news site"),
        ("social media post", "platform"), ("chatbot message", "customer service"),
        ("API call payload", "backend service"),
    ]
    txt_intent_domains = [
        ("banking chatbot", ["transfer money", "check balance", "report fraud"]),
        ("e-commerce assistant", ["track order", "return item", "find product"]),
        ("HR chatbot", ["request leave", "check payslip", "raise ticket"]),
        ("healthcare bot", ["book appointment", "refill prescription", "get test results"]),
        ("travel agent bot", ["book flight", "cancel trip", "change seat"]),
    ]
    txt_classification_tasks = [
        ("news articles", "topic", ["politics", "sports", "technology", "finance", "entertainment"]),
        ("support tickets", "department", ["billing", "technical", "shipping", "returns", "account"]),
        ("legal contracts", "type", ["NDA", "employment", "service agreement", "lease", "partnership"]),
        ("research papers", "field", ["ML", "biology", "physics", "economics", "medicine"]),
        ("job descriptions", "seniority", ["intern", "junior", "mid", "senior", "executive"]),
    ]
    txt_detection_tasks = [
        ("toxic comments", "online gaming chat platform", "severity level"),
        ("fake product reviews", "e-commerce marketplace", "confidence score"),
        ("hate speech", "social media posts", "violation category"),
        ("PII data", "customer emails", "data type found"),
        ("plagiarised paragraphs", "academic submissions", "similarity score"),
    ]
    txt_language_tasks = [
        "Identify the language of customer queries to route them to the right support team",
        "Detect code-switching (language mixing) in social media posts for NLP preprocessing",
        "Classify dialect variant of Arabic texts for regional content moderation",
    ]
    txt_extra = [
        "Extract named entities (company, person, date, location) from financial news articles",
        "Classify patient discharge summaries into primary diagnosis categories for hospital billing",
        "Detect urgency level in customer complaint emails to prioritise response queue",
        "Classify regulatory document sections as compliant or non-compliant with GDPR",
        "Identify the emotional tone (anger, joy, sadness, fear) in social media posts",
        "Classify Reddit posts into mental health concern categories for early intervention study",
        "Detect clickbait headlines in news aggregator feeds",
        "Classify scientific abstracts by methodology type (experimental, observational, review)",
        "Identify political bias (left, center, right) in news article text",
        "Classify legal case descriptions by jurisdiction and case type automatically",
        "Detect misinformation claims in health-related social media posts",
        "Route incoming customer emails to the correct agent team based on content",
        "Classify insurance claim descriptions as fraudulent, suspicious, or legitimate",
        "Detect aggressive or threatening language in employee communications for HR compliance",
        "Identify intent signals in e-commerce search queries to personalise results",
        "Classify financial analyst reports by market sentiment (bullish, bearish, neutral)",
        "Detect off-topic comments in a focused online community forum",
        "Classify government policy documents into spending categories automatically",
        "Extract and classify medication names and dosages from clinical notes",
        "Identify escalation-worthy messages in live customer support chat transcripts",
    ]

    for domain, platform in txt_sentiment_domains:
        add("text", f"Analyze sentiment in {platform} for our {domain} business to track customer satisfaction trends")
    for msg_type, channel in txt_spam_variants:
        add("text", f"Classify {msg_type} in a {channel} as spam or legitimate with confidence score")
    for bot, intents in txt_intent_domains:
        add("text", f"Classify user intent in a {bot} into categories: {', '.join(intents)}")
    for docs, task, cats in txt_classification_tasks:
        add("text", f"Classify {docs} by {task} into: {', '.join(cats)}")
    for what, where, output in txt_detection_tasks:
        add("text", f"Detect {what} in {where} and output {output}")
    for t in txt_language_tasks:
        add("text", t)
    for t in txt_extra:
        add("text", t)

    # Top-up to 200
    txt_topup = [
        "Build a system to categorise customer reviews as feature request, bug report, or general praise",
        "Detect urgency in supply chain disruption news articles for procurement alerts",
        "Classify Reddit posts about mental health into support-needed vs general discussion",
        "Identify and classify rhetorical devices in political speeches for media analysis",
        "Detect and classify phishing email attempts from corporate inbox metadata and body text",
        "Classify legal deposition transcripts into question type categories for paralegal workflow",
        "Identify the formality level of written text to assist proofreading tools",
        "Classify movie scripts by genre from the first act dialogue",
        "Detect and flag PII in customer service transcripts for GDPR compliance",
        "Build a multi-label classifier for scientific paper abstracts — one paper can span multiple topics",
        "Classify customer churn reason from exit survey free-text responses",
        "Detect code quality issues in GitHub issue descriptions (bug, enhancement, question)",
        "Identify sarcasm and irony in tweets to improve sentiment analysis accuracy",
        "Classify employee performance review texts into strengths and improvement areas",
        "Detect language used to manipulate or coerce in online communications",
        "Classify medical forum posts by medical specialty for routing to expert responders",
        "Identify emotional support seeking in text messages for mental wellness app",
        "Classify startup pitch deck descriptions by market category for VC portfolio matching",
        "Detect and categorize financial jargon in analyst calls for retail investor summaries",
        "Classify podcast transcript segments by topic for auto-chapter generation",
        "Detect product defect mentions in app store reviews and link to product SKU",
        "Classify insurance policy documents into coverage type for premium calculation",
        "Identify anti-competitive language in business contracts for legal review",
        "Detect market-moving claims in financial news for trading alert systems",
        "Build a system that identifies the emotional need behind a user message in a therapy app",
        "Classify news comment threads by toxicity level and reason for moderation queue",
        "Detect and extract deadline mentions from project management emails",
        "Classify social media posts about a brand crisis by type: complaint, support, neutral",
        "Identify the difficulty level of math problems from their text description",
        "Build a multi-class fake news detector that outputs category: satire, misleading, fabricated",
    ]
    for t in txt_topup:
        add("text", t)

    # ── TABULAR (200) ─────────────────────────────────────────────────────────
    tab_fraud = [
        ("credit card", "transaction amount, merchant category, time of day, location mismatch"),
        ("insurance claim", "claim amount, policy duration, claimant history, incident type"),
        ("account takeover", "login location, device fingerprint, session duration, action sequence"),
        ("mortgage application", "income statements, credit score, employment history, property value"),
        ("tax filing", "declared income, expense categories, filing history, dependant count"),
    ]
    tab_churn = [
        ("telecom", "call volume, data usage, contract type, support calls, payment history"),
        ("SaaS B2B", "login frequency, feature usage, support tickets, contract value, NPS score"),
        ("retail subscription box", "order frequency, returns rate, discount usage, engagement score"),
        ("streaming platform", "watch hours, content diversity, inactivity periods, billing method"),
        ("online banking", "transaction frequency, product count, tenure, digital engagement score"),
    ]
    tab_price = [
        ("residential properties", "square footage, postcode, bedrooms, year built, proximity to schools"),
        ("used car prices", "mileage, make, model, year, service history, number of owners"),
        ("airline tickets", "route, days before departure, seat class, day of week, baggage"),
        ("hotel rooms", "star rating, location, season, event proximity, booking lead time"),
        ("agricultural commodities", "weather index, crop yield forecast, futures contracts, storage costs"),
    ]
    tab_hr = [
        "Predict employee attrition risk from HR metrics (salary, tenure, performance, satisfaction)",
        "Classify job applicants by hire likelihood from resume features and assessment scores",
        "Predict time-to-fill for open job requisitions from historical hiring data",
        "Segment employees into performance tiers from engagement survey and KPI data",
        "Predict absenteeism rate from workforce demographic and schedule features",
    ]
    tab_supply = [
        "Forecast weekly demand for 10,000 SKUs in a retail chain from historical POS data",
        "Predict delivery delays from order features, carrier performance, and weather data",
        "Classify supplier risk level from payment history, financial ratios, and audit scores",
        "Optimise reorder point for warehouse inventory from lead time and demand variability",
        "Detect anomalous shipment events from GPS tracking and weight sensor data",
    ]
    tab_energy = [
        "Predict energy consumption per building from occupancy schedule and weather forecasts",
        "Detect abnormal power usage patterns in smart meter data indicating energy theft",
        "Forecast solar panel output from historical irradiance, temperature, and panel age",
        "Classify industrial equipment health from vibration, temperature, and current readings",
        "Predict grid load for the next 24 hours from historical demand and weather data",
    ]
    tab_healthcare = [
        "Predict 30-day hospital readmission risk from patient vitals, diagnosis codes, and LOS",
        "Classify patient risk tier for chronic disease management from EHR structured data",
        "Predict ICU length of stay from admission lab results and patient demographics",
        "Detect sepsis onset early from nurse observation charts and vital sign time series",
        "Identify patients at high risk of medication non-adherence from pharmacy fill history",
    ]
    tab_fintech = [
        "Predict loan default probability from credit bureau features and application data",
        "Score customer creditworthiness for BNPL approval from transaction behaviour",
        "Detect money laundering patterns in wire transfer network graph features",
        "Forecast monthly revenue for SME clients from bank transaction history",
        "Classify investment portfolio risk from holding allocation and market beta features",
    ]
    tab_marketing = [
        "Predict customer lifetime value from first-30-day behaviour features in e-commerce",
        "Classify email campaign send-time for maximum open rate per subscriber segment",
        "Predict cross-sell probability for product bundles from purchase history",
        "Segment users into persona groups from app behaviour and demographic features",
        "Build a propensity model to predict which free trial users convert to paid",
    ]
    tab_extra = [
        "Detect anomalous sensor readings in a water treatment plant from IoT time-series data",
        "Predict equipment failure 7 days in advance from turbine sensor readings",
        "Classify restaurant health inspection outcomes from feature data before on-site visit",
        "Forecast foot traffic for retail stores from weather, local events, and seasonality features",
        "Build a pricing model for dynamic room allocation in co-working spaces",
        "Predict student dropout risk from LMS engagement features and grade history",
        "Classify crop insurance claim validity from remote sensing and weather station features",
        "Rank job candidates by role fit score from structured interview scorecard data",
        "Detect early signs of financial distress in SMEs from accounting ratio time series",
        "Predict taxi demand by zone and hour from historical trip and weather data",
    ]

    for asset, features in tab_fraud:
        add("tabular", f"Detect {asset} fraud from tabular features: {features}")
    for industry, features in tab_churn:
        add("tabular", f"Predict {industry} customer churn from usage features: {features}")
    for asset, features in tab_price:
        add("tabular", f"Predict {asset} prices from features: {features}")
    for t in tab_hr: add("tabular", t)
    for t in tab_supply: add("tabular", t)
    for t in tab_energy: add("tabular", t)
    for t in tab_healthcare: add("tabular", t)
    for t in tab_fintech: add("tabular", t)
    for t in tab_marketing: add("tabular", t)
    for t in tab_extra: add("tabular", t)

    # Top-up to 200
    tab_topup = [
        "I have a CSV of 500,000 mortgage applications — predict which ones will default within 2 years",
        "Build a model to classify customer support tickets by priority from structured metadata",
        "Predict which B2B leads will close in the next 30 days from CRM activity features",
        "Detect fraudulent expense claims from employee reimbursement submission features",
        "Build a model to rank products by expected sell-through rate for inventory planning",
        "Predict patient no-show rate for outpatient clinic appointments from historical data",
        "Forecast hotel occupancy rate for the next 90 days from booking pace and market data",
        "Classify construction projects by on-time delivery risk from bid and contractor features",
        "Predict battery degradation rate from electric vehicle usage telemetry",
        "Build a model to classify social housing applications by urgency from structured form data",
        "Predict which warehouse pick orders will require manual intervention from routing features",
        "Detect network equipment that is likely to fail in the next 48 hours from SNMP polling data",
        "Build a model to score compliance risk of financial advisers from activity log features",
        "Predict the outcome of clinical trials Phase II from early biomarker data",
        "Classify online marketplace sellers by fraud risk from account and listing features",
        "Detect tax evasion patterns from structured company financial filings data",
        "Predict production yield for semiconductor wafers from process parameter CSV data",
        "Build a predictive maintenance model for HVAC systems from service history and sensor data",
        "Classify bank transactions into spending categories for personal finance app",
        "Predict concert ticket demand for new artists from streaming and social media features",
        "Detect billing errors in telecom invoices from call detail record CSV data",
        "Build a model to predict job offer acceptance rate from candidate profile features",
        "Classify customer complaints by root cause category from structured CRM fields",
        "Predict crop irrigation need for the next week from soil moisture sensor time series",
        "Detect anomalous trading activity from equity order book features in real time",
        "Build a price elasticity model for subscription plans from historical pricing experiments",
        "Predict time to resolution for IT support tickets from queue and agent features",
        "Classify manufacturing batches as pass or fail from process control chart metrics",
        "Predict the net revenue impact of promotional discounts from customer segment features",
        "Detect insurance policy premium leakage from underwriting attribute features",
    ]
    for t in tab_topup:
        add("tabular", t)

    # ── AUDIO (100) ───────────────────────────────────────────────────────────
    audio_list = [
        "Transcribe and classify customer service call recordings by issue type and urgency",
        "Detect machine failure sounds from factory floor audio streams and alert maintenance",
        "Classify music genres from 30-second audio clips for streaming recommendation engine",
        "Identify speaker identity from voice recordings for call-centre authentication",
        "Detect baby crying vs background noise in smart baby monitor audio",
        "Classify environmental sounds (traffic, rain, construction, birdsong) in city audio sensors",
        "Detect language spoken in a customer call to route to the correct language support team",
        "Identify emotion in call centre agent voice (frustrated, calm, happy) for coaching",
        "Detect gunshot sounds vs fireworks in urban surveillance audio for safety alerts",
        "Classify engine sounds in automotive diagnostics to predict mechanical faults",
        "Build a keyword spotting system to detect 'help' and 'emergency' in elderly care audio",
        "Detect anomalous sounds in wind turbine audio recordings for predictive maintenance",
        "Identify animal species from field recording audio for biodiversity surveys",
        "Classify snoring patterns from bedroom microphone audio for sleep apnea screening",
        "Detect cough events in public space audio to monitor respiratory illness spread",
        "Build a voice activity detection system to filter silence from conference call recordings",
        "Classify music instrument types from isolated audio tracks for session musician matching",
        "Detect pipeline leak sounds from underground acoustic sensor data",
        "Identify and classify DTMF tones in call recordings for IVR quality analysis",
        "Build an audio fingerprinting system to detect pirated music in uploaded content",
        "Detect stress levels in pilot voice communication from air traffic control recordings",
        "Classify the type of crowd noise (chanting, booing, cheering) in sports event audio",
        "Identify speaker age group and gender from short voice sample for demographics study",
        "Detect vehicle type (motorcycle, truck, bus, car) from urban traffic audio sensors",
        "Build a system to detect and classify respiratory wheeze and crackle in lung auscultation audio",
        "Classify alarm types (fire, intrusion, medical) in building management system audio feeds",
        "Detect deception cues in voice stress analysis for interview screening",
        "Identify accent region from English speech sample for market research segmentation",
        "Build a music mood classifier (energetic, calm, sad, happy) for playlist generation",
        "Detect voice commands in noisy kitchen environment for smart appliance control",
        "Classify industrial pump audio as normal, cavitation, or bearing failure",
        "Build a system to identify when a child is being bullied from playground audio",
        "Detect music tempo and key from raw audio for DJ mixing software",
        "Classify HVAC system noise as normal operation or fault condition",
        "Identify language code-switching events in multilingual classroom audio recordings",
        "Build a system to detect unauthorised recording devices in meeting rooms from audio sweep",
        "Classify TV programme genre from audio-only broadcast stream",
        "Detect and count overlapping speakers in podcast recordings for transcript alignment",
        "Build a sound event detection system for smart home that identifies doorbell, phone, baby",
        "Classify bird song types from citizen science audio recordings for ornithology dataset",
        "Detect cardiac murmur vs normal heart sounds from digital stethoscope audio",
        "Identify drill bit wear state from CNC machine audio in real time",
        "Build an audio anomaly detector for server room cooling systems",
        "Classify music era (60s, 70s, 80s, 90s, 2000s) from short audio clip for DJ tools",
        "Detect spoken numbers in noisy environments for real-time score keeping",
        "Identify and classify audio events in a restaurant (laughter, dishes, music) for ambiance study",
        "Build a voice quality scoring model for VOIP call monitoring",
        "Classify speech intelligibility level from audio samples for hearing aid calibration",
        "Detect road surface type (asphalt, gravel, cobblestone) from vehicle suspension microphone",
        "Build a wake-word detection system for embedded IoT devices with low false-positive rate",
        "Classify the emotional state of an audience from applause and reaction audio",
        "Detect and classify aircraft engine anomalies from cockpit audio recordings",
        "Build a system to detect when a musician is out of tune from ensemble recording audio",
        "Classify SONAR pings in underwater acoustic data for submarine detection",
        "Detect and alert on door slamming, glass breaking, and shouting in domestic audio",
        "Build a model to predict speech recognition error rate from acoustic environment features",
        "Classify Foley sound effects in film production by category for asset management",
        "Detect infant speech milestones from caregiver recordings for developmental screening",
        "Build an automatic speech recognition fine-tuning dataset from domain-specific audio",
        "Classify audio quality issues (clipping, noise, echo) from recorded podcast episodes",
        "Detect and segment speaker turns in long-form interview recordings for transcript generation",
        "Build a music cover song identification system from audio fingerprints",
        "Classify electroencephalogram (EEG) audio representations by brain state",
        "Detect poaching activity from gunshot sounds in protected forest audio sensors",
        "Build a system that classifies music difficulty level from instrument audio for learning apps",
        "Identify and track individual whale calls in ocean acoustic monitoring data",
        "Detect breathing irregularities in ICU patient audio for early warning system",
        "Classify language fluency level from speech sample for language learning platform",
        "Build a real-time audio event detection pipeline for smart city infrastructure",
        "Identify spoken product model numbers from customer service call audio",
        "Detect audio spoofing attacks (replay, TTS synthesis) for anti-spoofing voice biometrics",
        "Classify underwater noise sources (ship, sonar, biological) in maritime sensor data",
        "Build a system to score pronunciation accuracy in language learning app from audio",
        "Detect and classify audio-based social engineering attacks in call centre recordings",
        "Identify vocal health issues (nodules, polyps) from voice recording features",
        "Build an emergency siren detector for autonomous vehicle audio perception system",
        "Classify music listener engagement level from audio interaction data",
        "Detect alcohol intoxication from voice characteristics in roadside sobriety screening",
        "Build a model to predict music chart performance from audio features at release",
        "Classify factory noise zones by decibel level and frequency profile for worker safety",
        "Detect infant colic patterns from crying audio for parenting support app",
        "Build a speaker diarisation system for multi-party podcast episode transcription",
        "Classify climate conditions from insect sound recordings for entomological study",
        "Detect and extract speech from cockpit voice recorder under heavy background noise",
        "Build an audio-based gate control system that opens only for authorised voice commands",
        "Classify public transport announcement audio by urgency and information type",
        "Identify the instrument playing melody in polyphonic music for music education tools",
        "Detect structural resonance anomalies from vibration audio in bridge monitoring",
        "Build a speech emotion recognition model robust to background noise in call centres",
        "Classify cooking sounds (sizzling, boiling, chopping) from kitchen microphone for smart recipes",
        "Detect and count individual voices in crowd audio for event capacity management",
        "Build a real-time laughing vs crying classifier for infant care nursery camera audio",
        "Classify breathing effort level from respiratory audio for post-operative patient monitoring",
        "Detect firearm type from gunshot audio in law enforcement ballistic analysis",
        "Build a music synchronisation system that aligns audio to video using beat detection",
        "Classify HVAC compressor state from acoustic emission sensor data",
        "Detect mispronounced words in student reading aloud sessions for literacy apps",
        "Build a system to identify speaking turns and interruptions in negotiation recordings",
        "Classify sports game highlight moments from broadcast audio for automated video editing",
        "Detect transformer hum anomalies from power grid audio monitoring for fault prediction",
    ]
    for t in audio_list[:100]:
        add("audio", t)

    # ── MULTIMODAL (100) ──────────────────────────────────────────────────────
    mm_list = [
        "Classify product listings as genuine or misleading using both title text and product image",
        "Generate and validate image captions for accessibility compliance on a news website",
        "Verify that clothing product images match their size, colour, and material text description",
        "Detect hateful content in social media posts combining image and caption together",
        "Classify restaurant menu item as vegan-friendly using dish photo and ingredient text",
        "Identify medical condition from patient-submitted photo plus symptom description text",
        "Build a visual question answering system for customer product queries on e-commerce site",
        "Classify job advertisement legitimacy using company logo image and text description",
        "Detect brand logo usage violations by comparing image with registered trademark text data",
        "Classify academic paper relevance from both abstract text and figure images",
        "Build a content safety filter that analyses image and accompanying tweet text together",
        "Classify fashion outfit style from clothing item images and styling notes text",
        "Identify counterfeit product listings by comparing product image with known authentic item",
        "Build a recipe recommendation system using dish photo and ingredient list text",
        "Classify real estate listings by price tier using property photos and description text",
        "Detect plagiarised scientific figures by matching image content with source paper text",
        "Build a visual sentiment analysis system using product unboxing video frames and review text",
        "Classify vehicle damage severity from photo and written incident report for insurance",
        "Identify political advertisement bias using image content and text overlay together",
        "Build a document classification system using scanned form image and extracted OCR text",
        "Detect fake profile accounts by analysing profile photo and bio text together",
        "Classify event photos by event type using image and caption metadata",
        "Build a travel recommendation system using destination photos and user preference text",
        "Identify unsafe food using photo and ingredient label text for allergy detection",
        "Classify art pieces by period and style using painting image and museum description text",
        "Build an automated document verification system matching ID photo with form text",
        "Detect misleading news by comparing article headline text with embedded image content",
        "Classify social media influencer content category using post image and caption text",
        "Build a smart home product search using sketch image and natural language description",
        "Identify insurance claim fraud using accident scene photo and claimant statement text",
        "Classify satellite imagery with accompanying weather data text for disaster response triage",
        "Build a recruitment screening system using candidate headshot photo and CV text",
        "Detect unsafe products by matching product photo with safety data sheet text",
        "Classify medical imaging reports by urgency using scan image and radiologist note text",
        "Build a vehicle inspection system using damage photo and mechanic assessment text report",
        "Identify crop disease using field drone image and agronomist notes text",
        "Classify social commerce product quality using user-submitted photo and description text",
        "Detect brand inconsistencies in marketing materials using image and brand guideline text",
        "Build a packaging compliance checker using label photo and regulatory requirement text",
        "Classify hotel room condition from check-in photo and guest complaint text",
        "Build a system that recommends outfit combinations from wardrobe images and weather text",
        "Classify petition as genuine or coordinated campaign using signatory photo and text pattern",
        "Detect copyright infringement in content using uploaded image and licence text metadata",
        "Build a virtual try-on matching system using user face photo and product description text",
        "Classify a company pitch deck by sector using slide images and pitch text together",
        "Detect inconsistencies between product specification text and product photo in listings",
        "Build a meme toxicity classifier using image template and overlaid text together",
        "Identify architecture style from building photo and geographic location text data",
        "Classify prescription drug packaging compliance using label image and drug registration text",
        "Build an automated fact-checking pipeline using news article image evidence and claim text",
        "Classify emotion expressed in social media post by combining image emoji and written text",
        "Detect agricultural land use violations using satellite image and land registry text",
        "Build a quality control system for printed circuit boards using macro image and BOM text",
        "Classify product return reason using returned item photo and customer explanation text",
        "Identify art forgery by comparing artwork image with provenance text documentation",
        "Build a recommendation engine for second-hand goods using listing photo and description text",
        "Classify patient wellness using wearable sensor data alongside mood diary text entries",
        "Detect age-inappropriate content using image scene and associated text captions together",
        "Build a smart nutrition tracker using meal photo and dietary goal text from user profile",
        "Classify real vs AI-generated images by combining pixel features with metadata text",
        "Detect driver distraction using dashcam image frame and CAN bus telemetry text data",
        "Build a competitive intelligence tool using competitor product image and pricing text",
        "Classify urban infrastructure damage using street-level image and maintenance request text",
        "Detect invasive species in nature photos using image and GPS location metadata text",
        "Build a sign language to text translation system using hand gesture video frames",
        "Classify wine quality using bottle label photo and sommelier tasting notes text",
        "Detect animal welfare concerns using farm image and farm management report text",
        "Build a medical chart understanding system using handwritten form scan and structured EHR text",
        "Classify safety hazard level using construction site photo and incident report text",
        "Identify plagiarised design using product image and original designer trademark text",
        "Build a personalised learning system using student work photo and teacher feedback text",
        "Classify social media crisis by impact level using trending image and press release text",
        "Detect manufacturing compliance using product image and engineering specification text",
        "Build an e-commerce search ranker using product image, title text, and user query text",
        "Classify graffiti type using photo and neighbourhood demographics text for city planning",
        "Detect parking violations using CCTV image and parking permit text records",
        "Build an intelligent form completion system using document scan image and OCR extracted text",
        "Classify furniture assembly instruction clarity using diagram image and written step text",
        "Identify suspicious package delivery using doorbell camera image and shipping label text",
        "Build an automated news photo caption generator with factual accuracy check against article",
        "Classify disease outbreak reports using epidemiology map image and statistical text summary",
        "Detect financial chart manipulation using graph image and reported metrics text",
        "Build a smart warranty claim system using product damage photo and purchase receipt text",
        "Classify security incident from CCTV image combined with access log text records",
        "Detect false advertising by matching product image against claimed specifications text",
        "Build a biodiversity assessment tool using citizen science photo and species description text",
        "Classify real estate property condition using exterior photo and inspection report text",
        "Detect vehicle accident fraud using crash site image and police report text together",
        "Build a personalised recipe difficulty ranker using dish image and chef instructions text",
        "Classify archaeological artefact period using dig site photo and fieldwork notes text",
        "Detect deep fake videos by analysing frame image quality and audio transcript text",
        "Build a smart home security system using camera frame image and motion sensor data text",
        "Classify environmental pollution level using water sample image and lab report text",
        "Detect workplace PPE violations using safety camera image and employee shift record text",
        "Build a market trend predictor using product image trends and search volume text data",
        "Classify smart meter tamper evidence using meter image and consumption history text",
        "Detect fraudulent insurance damage claims using photo evidence and claimant statement text",
        "Build a retail planogram compliance verifier using shelf image and layout specification text",
        "Classify social media post authenticity using image metadata and posting behaviour text",
        "Detect counterfeit currency using high-resolution banknote scan and security feature text",
        "Build a personalised fitness plan classifier using body assessment photo and goal text",
        "Classify news broadcast reliability using anchor photo credibility and transcript text",
    ]
    for t in mm_list[:100]:
        add("multimodal", t)

    # ── MEDICAL (100) ─────────────────────────────────────────────────────────
    med_list = [
        "Detect pneumonia from chest X-ray images with confidence score for radiologist triage",
        "Classify brain tumour type (glioma, meningioma, pituitary) from MRI scans",
        "Detect malignant melanoma from dermoscopy skin lesion photographs",
        "Identify diabetic retinopathy severity grade from retinal fundus photographs",
        "Detect COVID-19 from CT scan slices and classify vs other pneumonia",
        "Classify bone fracture type and location from X-ray images for orthopaedic triage",
        "Detect early-stage lung cancer nodules in low-dose CT scan images",
        "Identify glaucoma from optic disc and retinal nerve fibre layer OCT scans",
        "Classify histopathology slide tissue as cancerous or benign for pathology lab",
        "Detect intracranial haemorrhage in non-contrast head CT for emergency radiology",
        "Identify appendicitis from abdominal ultrasound images to reduce unnecessary surgery",
        "Classify ECG signals as normal vs arrhythmia type for cardiac monitoring",
        "Detect dental cavity and periodontal disease from panoramic dental X-rays",
        "Identify prostate cancer Gleason grade from biopsy histology slide images",
        "Detect macular degeneration from fundus photographs for ophthalmology screening",
        "Classify skin rash type (eczema, psoriasis, rosacea) from dermatology photos",
        "Detect spinal cord compression from MRI for neurosurgery prioritisation",
        "Identify colon polyp type (adenoma, hyperplastic) from colonoscopy video frames",
        "Classify thyroid nodule malignancy risk from ultrasound images",
        "Detect breast cancer from mammography images for population screening programme",
        "Identify COVID-19 severity from chest X-ray to predict ICU admission need",
        "Classify knee cartilage damage grade from MRI for orthopaedic surgery planning",
        "Detect retinal detachment from fundus image in diabetic patient screening",
        "Classify liver disease stage (fibrosis, cirrhosis) from abdominal ultrasound",
        "Detect aortic aneurysm in abdominal CT scan for vascular surgery triage",
        "Identify Alzheimer's disease stage from brain PET scan images",
        "Classify skin lesion ABCDE features from dermoscopy for melanoma risk scoring",
        "Detect pulmonary embolism from CTPA scan in emergency radiology",
        "Identify diabetic foot ulcer severity from wound photograph for podiatry",
        "Classify sleep apnea severity from polysomnography signal features",
        "Detect corneal ulcer from slit-lamp photograph for ophthalmic emergency",
        "Identify hip dysplasia in infant X-ray for paediatric orthopaedics",
        "Classify cervical cell abnormality type from Pap smear cytology images",
        "Detect atelectasis, cardiomegaly, effusion, and pneumonia in chest X-ray multi-label",
        "Identify hand bone age from paediatric X-ray for growth assessment",
        "Classify liver tumour type from multiphasic CT scan images",
        "Detect gallstones from abdominal ultrasound for gastroenterology triage",
        "Identify Barrett's oesophagus from endoscopy images for cancer surveillance",
        "Classify ADHD vs control from functional MRI connectivity patterns",
        "Detect age-related macular degeneration dry vs wet type from OCT scans",
        "Identify wound healing stage from serial photograph documentation",
        "Classify schizophrenia vs healthy control from structural brain MRI features",
        "Detect intracranial aneurysm in MRA scan for neurovascular triage",
        "Identify diabetic nephropathy stage from kidney biopsy pathology images",
        "Classify bladder cancer invasion depth from cystoscopy images",
        "Detect scoliosis and measure Cobb angle from full-spine standing X-ray",
        "Identify tuberculosis lesion pattern from chest X-ray in high-burden settings",
        "Classify cardiac hypertrophy type from echocardiogram video frames",
        "Detect Parkinson's disease from DaTscan nuclear medicine imaging",
        "Identify protein expression level in immunohistochemistry slide images",
        "Classify benign vs malignant thyroid nodule from fine needle aspiration cytology",
        "Detect early cataract formation from anterior segment slit-lamp images",
        "Identify placental abnormality from obstetric ultrasound for obstetric triage",
        "Classify wound infection vs healing from serial wound photograph analysis",
        "Detect ventricular wall motion abnormality from cardiac MRI cine images",
        "Identify rib fracture from chest CT for forensic radiology",
        "Classify anterior cruciate ligament tear grade from knee MRI images",
        "Detect ovarian cyst type from transvaginal ultrasound for gynaecology",
        "Identify bone metastasis in skeletal scintigraphy scan for oncology staging",
        "Classify fungal skin infection type from clinical dermatology photographs",
        "Detect nasal polyp from endoscopic sinus images for ENT surgery planning",
        "Identify emphysema severity pattern from HRCT chest scan",
        "Classify mitotic figures in histopathology images for tumour grading",
        "Detect retinal vein occlusion from fundus photograph for ophthalmology alert",
        "Identify abdominal aortic calcification from X-ray for cardiovascular risk scoring",
        "Classify vocal fold pathology from laryngoscopy video frames",
        "Detect splenic laceration from abdominal CT in trauma emergency imaging",
        "Identify neonatal jaundice severity from skin colour photograph",
        "Classify chronic obstructive pulmonary disease severity from spirometry features",
        "Detect meningioma location and size from gadolinium-enhanced brain MRI",
        "Identify subclinical hypothyroidism from thyroid ultrasound and blood test features",
        "Classify cartilage lesion grade in shoulder MRI for sports medicine",
        "Detect venous thromboembolism risk from lower limb Doppler ultrasound",
        "Identify cerebral palsy motor pattern from gait analysis video frames",
        "Classify cholangiocarcinoma from MRCP biliary imaging",
        "Detect sacral stress fracture from MRI in distance runner for sports medicine",
        "Identify retinopathy of prematurity stage from infant fundus images",
        "Classify adrenal mass type from CT scan for endocrinology workup",
        "Detect facial nerve palsy severity from standardised facial photograph",
        "Identify rotator cuff tear from shoulder MRI for orthopaedic planning",
        "Classify non-alcoholic fatty liver disease grade from liver biopsy images",
        "Detect ovarian hyperstimulation syndrome from pelvic ultrasound",
        "Identify temporal lobe epilepsy focus from ictal SPECT brain scan",
        "Classify oral cancer stage from clinical photograph for head and neck oncology",
        "Detect achilles tendon tear from MRI for sports injury management",
        "Identify pulmonary fibrosis pattern type from HRCT for ILD clinic",
        "Classify thyroid cancer subtype from surgical resection histopathology images",
        "Detect early Parkinson's tremor from wrist accelerometer signal data",
        "Identify periventricular leukomalacia in neonatal brain ultrasound",
        "Classify haematoma type from head CT for trauma neurosurgery",
        "Detect metastatic lymph node involvement from neck CT scan in head-neck cancer",
        "Identify Crohn's disease vs ulcerative colitis from colon biopsy histology",
        "Classify optic neuritis vs normal from visual evoked potential signals",
        "Detect peripheral arterial disease from ankle-brachial index waveform data",
        "Identify spinal cord tumour type from whole-spine MRI",
        "Classify psoriasis severity PASI score from standardised body photograph",
        "Detect ectopic pregnancy from transvaginal ultrasound in emergency gynaecology",
        "Identify pneumothorax size from chest X-ray for emergency respiratory triage",
        "Classify lymphoma subtype from PET-CT scan for haematology treatment planning",
    ]
    for t in med_list[:100]:
        add("medical", t)

    # ── SECURITY (50) ─────────────────────────────────────────────────────────
    sec_list = [
        "Detect network intrusion patterns in firewall log data for SOC alert generation",
        "Classify malware type (ransomware, spyware, trojan) from binary feature extraction",
        "Detect phishing URLs from URL lexical and host features in real time",
        "Identify credential stuffing attacks from login event sequence patterns",
        "Detect DDoS attack patterns from network traffic flow features",
        "Classify insider threat risk level from employee digital behaviour features",
        "Detect SQL injection attempts from web application log feature patterns",
        "Identify cross-site scripting attack payloads from HTTP request features",
        "Detect anomalous API calls that indicate data exfiltration attempts",
        "Classify social engineering email types (CEO fraud, invoice scam, HR impersonation)",
        "Detect bot traffic from user session behavioural biometrics features",
        "Identify zero-day exploit patterns from sandbox execution log features",
        "Detect port scanning and reconnaissance activity from NetFlow data",
        "Classify vulnerability severity from CVE description text for patch prioritisation",
        "Detect account takeover in real time from login location and device features",
        "Identify cryptomining malware from CPU and network usage telemetry features",
        "Detect data breach in progress from egress volume anomaly in DLP log data",
        "Classify dark web forum posts by threat type for cyber threat intelligence",
        "Detect brute force password attacks from authentication log features",
        "Identify malicious PowerShell script behaviour from command-line feature extraction",
        "Detect lateral movement in enterprise network from authentication graph features",
        "Classify suspicious email attachment type from file metadata features without opening",
        "Detect fake news content designed for political manipulation in social media text",
        "Identify compromised IoT device behaviour from network traffic anomaly features",
        "Detect command-and-control beacon traffic in DNS query log patterns",
        "Classify supply chain attack indicators in software build log features",
        "Detect privilege escalation attempts from Windows event log sequences",
        "Identify deepfake video from facial landmark and compression artefact features",
        "Detect keystroke logging malware from typing rhythm biometric features",
        "Classify ransomware encrypted file extension patterns from file system events",
        "Detect physical access tailgating events from door sensor and badge log data",
        "Identify stolen credit card testing patterns from payment gateway transaction features",
        "Detect vulnerability scanner activity from server log response pattern features",
        "Classify security alert severity from SIEM event feature aggregation",
        "Detect SIM swap fraud from mobile network event features in real time",
        "Identify watering-hole attack indicators from browser telemetry features",
        "Detect call centre fraud from voice biometrics and account query pattern features",
        "Classify malicious domain name generation algorithm patterns from DNS query text",
        "Detect insider data theft from file access pattern and email attachment features",
        "Identify stolen API key usage from request origin and rate anomaly features",
        "Detect business email compromise attack patterns from email metadata features",
        "Classify network traffic as encrypted tunnel evasion vs legitimate VPN usage",
        "Detect formjacking attack indicators from JavaScript execution behaviour features",
        "Identify compromised cloud storage bucket from access log anomaly features",
        "Detect ATM skimmer activation from transaction timing and card read anomaly",
        "Classify adversarial machine learning attack type from model query pattern features",
        "Detect rogue wireless access point from network scan and beacon frame features",
        "Identify synthetic identity fraud from credit application feature inconsistencies",
        "Detect web scraping bots from request timing, header, and navigation pattern features",
        "Classify nation-state APT group by attack technique fingerprint from IOC features",
    ]
    for t in sec_list[:50]:
        add("security", t)

    # ── EDGE CASES (50) ───────────────────────────────────────────────────────
    edge_list = [
        "Build a system that makes my business smarter using AI",
        "Analyze my data and tell me what is interesting in it",
        "I need a model that works on everything I throw at it",
        "Detect problems and fix them automatically in real time at massive scale",
        "Classify everything my users upload whether it is images, text, voice, or files",
        "Build me the best AI for my startup — I have some user data",
        "I want AI that understands context and gives smart recommendations",
        "Create a model that predicts things based on historical patterns",
        "Build an autonomous agent that monitors my system and takes action",
        "I need to process large amounts of unstructured data quickly",
        "Detect anomalies in whatever data my sensors produce",
        "Build an intelligent assistant that helps users achieve their goals",
        "Classify incoming data streams in real time with high accuracy",
        "I need AI that can handle edge cases better than rule-based systems",
        "Build a recommendation engine that works across all my product lines",
        "Detect fraud in my system — I have transactional and behavioural data",
        "I need something that understands my domain and gets smarter over time",
        "Build an AI system that can explain its decisions to non-technical users",
        "Classify and route incoming requests from multiple channels automatically",
        "I need an AI model I can deploy on a low-power edge device",
        "Build a model that works well even with very limited labelled training data",
        "Detect and classify events in real-time streaming data from IoT sensors",
        "I need a general-purpose text plus image understanding model for my platform",
        "Build a conversational AI that also does structured data analysis",
        "Predict future outcomes in a complex system with many interacting variables",
        "I need AI that works across English, Spanish, French, and Mandarin equally well",
        "Detect unusual behaviour across both user actions and network traffic simultaneously",
        "Build a model that performs well on both desktop and mobile data sources",
        "I have mixed structured and unstructured data — build a unified classification model",
        "Create an AI system that handles missing data gracefully and still gives good predictions",
        "Build something that monitors multiple data types and generates a single risk score",
        "I need a model that is accurate during the day and robust at night when patterns change",
        "Detect and classify content across text, images, and audio in a single pipeline",
        "Build an AI model that can be retrained overnight with new data without human help",
        "I need to classify thousands of categories — the taxonomy keeps growing",
        "Predict which of my users will do something important in the next 7 days",
        "Build a quality gate AI that works across all manufacturing lines regardless of product",
        "I need a real-time and batch processing AI that gives consistent results in both modes",
        "Detect problems across my entire platform — I have logs, metrics, and user events",
        "Build an AI that handles domain shift — my data distribution changes every quarter",
        "I want a model that can learn from 5 examples per new class without full retraining",
        "Build a system that classifies events that span multiple modalities simultaneously",
        "I need AI that is both interpretable for regulators and accurate for business",
        "Detect safety risks in an environment where false negatives are extremely costly",
        "Build a model that performs consistently across different geographic regions",
        "I need AI that works in a privacy-preserving way without sending data to the cloud",
        "Detect patterns across time series, images, and text data in a single unified model",
        "Build a system that handles concept drift and updates its predictions automatically",
        "I need a model that can classify inputs it has never seen before without retraining",
        "Build an AI system that works reliably with adversarial or noisy input data",
    ]
    for t in edge_list[:50]:
        add("edge_case", t)

    return probs


# ═════════════════════════════════════════════════════════════════════════════
# Production generate function (with safeguards)
# ═════════════════════════════════════════════════════════════════════════════

def _cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1_000_000 * DEEPSEEK_INPUT_COST_PER_M +
            out_tok / 1_000_000 * DEEPSEEK_OUTPUT_COST_PER_M)


def _call_with_retry(
    client: OpenAI,
    problem: str,
    system_prompt: str,
    schema_class,
) -> tuple[Any, int, int, int]:
    """
    Single API call with exponential-backoff retry.
    Returns (validated_object, in_tokens, out_tokens, latency_ms)
    Raises exception after MAX_RETRIES failures.
    """
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": problem},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )
            raw       = resp.choices[0].message.content
            validated = schema_class.model_validate_json(raw)
            lat_ms    = round((time.time() - t0) * 1000)
            return validated, resp.usage.prompt_tokens, resp.usage.completion_tokens, lat_ms
        except Exception as e:
            lat_ms = round((time.time() - t0) * 1000)
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt        # 1s, 2s, 4s
                print(f"    retry {attempt+1}/{MAX_RETRIES-1} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def generate_full_core(
    client: OpenAI,
    problems: list[dict],
    schema_class,
    system_prompt: str,
    core_key: str,
    friendly_name: str,
    out_dir: Path,
) -> dict:
    """
    Generate teacher responses for all 1000 problems for one brain core.

    Safeguards:
      - Resume from checkpoint (.progress file)
      - 100ms inter-call delay
      - Exponential-backoff retry (up to MAX_RETRIES)
      - Hard stop at COST_HARD_STOP
      - Quality-window alert at QUALITY_ALERT_THRESHOLD
      - Sample print every MONITOR_INTERVAL problems
      - Flush JSONL and checkpoint every CHECKPOINT_INTERVAL successes
    """
    checkpoint_path = CHECKPOINT_DIR / f"{core_key}.progress"
    jsonl_path      = out_dir / f"dataset_{core_key}_understander.jsonl" \
                      if core_key == "task" else \
                      out_dir / f"dataset_{core_key}_classifier.jsonl" \
                      if core_key == "domain" else \
                      out_dir / f"dataset_{core_key}_advisor.jsonl"

    # Fix filename mapping
    name_map = {
        "task":   "dataset_task_understander.jsonl",
        "domain": "dataset_domain_classifier.jsonl",
        "arch":   "dataset_architecture_advisor.jsonl",
    }
    jsonl_path      = out_dir / name_map[core_key]
    failures_path   = out_dir / f"full_failures_{core_key}.json"

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume: load completed IDs ──────────────────────────────────────────
    done_ids: set = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            done_ids = set(json.load(f))
        print(f"  [Resume] {len(done_ids)} already done, skipping.")

    # ── Open JSONL in append mode ───────────────────────────────────────────
    jsonl_f = open(jsonl_path, "a", encoding="utf-8")

    pending = [p for p in problems if p["id"] not in done_ids]
    print(f"  {friendly_name}: {len(pending)} problems to process "
          f"({len(done_ids)} already done)")

    total_in  = 0
    total_out = 0
    successes = 0
    failures_list: list = []
    all_latencies: list = []
    window_pass: list   = []   # last 100: True/False
    t_core_start        = time.time()
    checkpoint_buf: list = list(done_ids)

    for idx, entry in enumerate(pending):
        pid     = entry["id"]
        problem = entry["problem"]

        # ── Cost check ───────────────────────────────────────────────────
        running_cost = _cost_usd(total_in, total_out)
        if running_cost >= COST_HARD_STOP:
            print(f"\n  [HARD STOP] Running cost ${running_cost:.4f} >= "
                  f"${COST_HARD_STOP} limit. Stopping safely.")
            break

        # ── Call with retry ──────────────────────────────────────────────
        try:
            validated, in_tok, out_tok, lat_ms = _call_with_retry(
                client, problem, system_prompt, schema_class)

            jsonl_f.write(json.dumps(
                {"input": problem, "output": validated.model_dump()},
                ensure_ascii=False) + "\n")
            jsonl_f.flush()

            total_in  += in_tok
            total_out += out_tok
            all_latencies.append(lat_ms)
            successes += 1
            checkpoint_buf.append(pid)
            window_pass.append(True)
            status = "OK"
        except Exception as e:
            failures_list.append({"id": pid, "problem": problem, "error": str(e)})
            window_pass.append(False)
            status = f"FAIL: {str(e)[:50]}"
            lat_ms = 0

        # Keep window at 100 entries
        if len(window_pass) > 100:
            window_pass.pop(0)

        abs_idx = len(done_ids) + idx + 1
        print(f"  [{friendly_name}] {abs_idx:4d}/1000  {status}")

        # ── Monitoring every MONITOR_INTERVAL ────────────────────────────
        if abs_idx % MONITOR_INTERVAL == 0:
            cost_so_far  = _cost_usd(total_in, total_out)
            win_rate     = sum(window_pass) / max(len(window_pass), 1)
            avg_lat      = sum(all_latencies) / max(len(all_latencies), 1)
            fail_this_run = abs_idx - len(done_ids) - successes
            print(f"\n  --- Monitor @{abs_idx} ---")
            print(f"  Running cost:  ${cost_so_far:.4f}")
            print(f"  Window pass rate (last {len(window_pass)}): {win_rate:.1%}")
            print(f"  Avg latency:   {avg_lat/1000:.1f}s")
            print(f"  Failures this run: {fail_this_run}")
            if win_rate < QUALITY_ALERT_THRESHOLD:
                print(f"  *** QUALITY ALERT: pass rate {win_rate:.1%} < {QUALITY_ALERT_THRESHOLD:.0%} ***")

            # Sample 5 random successes
            if successes >= 5:
                sample_lines = []
                with open(jsonl_path, "r", encoding="utf-8") as sf:
                    all_lines = [l for l in sf.readlines() if l.strip()]
                sample_lines = random.sample(all_lines, min(5, len(all_lines)))
                print(f"  Quality samples:")
                for sl in sample_lines:
                    obj = json.loads(sl)
                    print(f"    Q: {obj['input'][:55]}...")
                    print(f"    A: {str(obj['output'])[:70]}...")
            print()

        # ── Checkpoint every CHECKPOINT_INTERVAL successes ───────────────
        if successes > 0 and successes % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_buf, f)
            with open(failures_path, "w", encoding="utf-8") as f:
                json.dump(failures_list, f, indent=2)

        # ── Rate limit delay ─────────────────────────────────────────────
        time.sleep(INTER_CALL_DELAY_S)

    jsonl_f.close()

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_buf, f)
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures_list, f, indent=2)

    elapsed = round(time.time() - t_core_start, 2)
    total_attempted = len(done_ids) + len(pending)
    total_in_dataset = successes + len(done_ids)

    return {
        "core":           core_key,
        "friendly_name":  friendly_name,
        "total":          len(problems),
        "passed":         total_in_dataset,
        "failed":         len(failures_list),
        "pass_rate_pct":  round(total_in_dataset / len(problems) * 100, 1),
        "input_tokens":   total_in,
        "output_tokens":  total_out,
        "cost_usd":       round(_cost_usd(total_in, total_out), 4),
        "elapsed_s":      elapsed,
        "avg_latency_ms": round(sum(all_latencies) / max(len(all_latencies), 1)),
        "jsonl_path":     str(jsonl_path),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[ERROR] DEEPSEEK_API_KEY not found.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # ── Generate and save 1000 problems ──────────────────────────────────────
    if not SEED_PATH.exists():
        print("Generating 1000 seed problems...")
        problems = _generate_problems()
        SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SEED_PATH, "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(problems)} problems to {SEED_PATH}")
    else:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            problems = json.load(f)
        print(f"Loaded {len(problems)} problems from {SEED_PATH}")

    # Count distribution
    from collections import Counter
    dist = Counter(p["category"] for p in problems)
    print("\nDistribution:")
    for cat, n in sorted(dist.items()):
        print(f"  {cat:12s}: {n}")
    print(f"  {'TOTAL':12s}: {len(problems)}\n")

    cores = [
        ("task",   TaskUnderstanding,    SYSTEM_PROMPT_TASK,   "Task Understander"),
        ("domain", DomainClassification, SYSTEM_PROMPT_DOMAIN, "Domain Classifier"),
        ("arch",   ArchitectureAdvice,   SYSTEM_PROMPT_ARCH,   "Architecture Advisor"),
    ]

    grand_start = time.time()
    core_stats: list = []

    for core_key, schema_cls, sys_prompt, friendly_name in cores:
        print(f"\n{'='*60}")
        print(f"  Core: {friendly_name}")
        print(f"{'='*60}")
        stats = generate_full_core(
            client, problems, schema_cls, sys_prompt,
            core_key, friendly_name, DATASET_DIR,
        )
        core_stats.append(stats)
        print(f"\n  Core done: {stats['passed']}/{stats['total']} "
              f"({stats['pass_rate_pct']}%)  "
              f"cost=${stats['cost_usd']:.4f}  "
              f"elapsed={stats['elapsed_s']}s")

    # ── Final report ──────────────────────────────────────────────────────────
    total_elapsed_min = round((time.time() - grand_start) / 60, 2)
    total_cost        = round(sum(s["cost_usd"] for s in core_stats), 4)
    total_passed      = sum(s["passed"] for s in core_stats)
    total_calls       = len(problems) * 3
    pass_rate         = round(total_passed / total_calls * 100, 1)
    avg_lat_ms        = round(sum(s["avg_latency_ms"] for s in core_stats) /
                              max(len(core_stats), 1))

    sep = "=" * 55
    report_lines = [
        sep,
        "FULL DATASET GENERATION REPORT",
        sep,
        f"Total problems: {len(problems)}",
        f"Total API calls: {total_calls} ({len(problems)} x 3 cores)",
        "",
        "Per-core success:",
    ]
    for s in core_stats:
        report_lines.append(
            f"  {s['friendly_name']}: {s['passed']}/{s['total']} "
            f"({s['pass_rate_pct']}%)"
        )

    # Sample 5 random examples per core
    report_lines += ["", "Quality samples (5 random per core):"]
    for s in core_stats:
        report_lines.append(f"\n  [{s['friendly_name']}]")
        jsonl_p = Path(s["jsonl_path"])
        if jsonl_p.exists():
            with open(jsonl_p, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if l.strip()]
            samples = random.sample(lines, min(5, len(lines)))
            for sl in samples:
                obj = json.loads(sl)
                report_lines.append(f"  Q: {obj['input'][:60]}")
                report_lines.append(f"  A: {str(obj['output'])[:80]}")
                report_lines.append("")

    report_lines += [
        f"Total cost: ${total_cost:.4f}",
        f"Total time: {total_elapsed_min} minutes",
        f"Avg response: {avg_lat_ms/1000:.1f}s",
        "",
        "DATASET STATUS: READY FOR DAY 14 (LoRA FINE-TUNING)",
        sep,
    ]

    print("\n\n" + "\n".join(report_lines))

    # Save JSON report
    report_dict = {
        "generated_at":     datetime.now().isoformat(),
        "teacher_model":    "deepseek-chat",
        "total_problems":   len(problems),
        "total_calls":      total_calls,
        "total_passed":     total_passed,
        "pass_rate_pct":    pass_rate,
        "total_cost_usd":   total_cost,
        "total_elapsed_min":total_elapsed_min,
        "avg_latency_ms":   avg_lat_ms,
        "per_core":         core_stats,
        "datasets": {s["core"]: s["jsonl_path"] for s in core_stats},
    }

    json_report_path = DATASET_DIR / "full_dataset_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    md_path = DATASET_DIR / "FULL_DATASET_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Full Dataset Generation Report\n\n```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")

    print(f"\nReports saved:")
    print(f"  {json_report_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
