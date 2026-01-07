import json
import os
import csv
from sentence_transformers import SentenceTransformer, util


MODELS = {
    "SBERT": "all-MiniLM-L6-v2",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "SPECTER": "sentence-transformers/allenai-specter",
    "SciNCL": "malteos/scincl"
}

DATA_DIR = "data"                
ANS_DIR = "synthetic_answers"     
OUT_DIR = "experiments/results"
OUT_FILE = f"{OUT_DIR}/method1_truncation.csv"

MAX_TOKENS = 512

def truncate(text, max_tokens=512):
    """Whitespace-based truncation to simulate token limits"""
    return " ".join(text.split()[:max_tokens])


os.makedirs(OUT_DIR, exist_ok=True)

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "qid",
        "topic",
        "difficulty",
        "model",
        "method",
        "answer_type",
        "original_tokens",
        "used_tokens",
        "cosine_similarity"
    ])



    for model_name, model_ckpt in MODELS.items():
        print(f"\nLoading model: {model_name}")
        model = SentenceTransformer(model_ckpt)
        tokenizer = model.tokenizer


        for file in os.listdir(DATA_DIR):
            if not file.endswith(".json"):
                continue

            topic = file.replace(".json", "")
            print(f"Processing topic: {topic}")

            with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as fq:
                questions = json.load(fq)

            with open(os.path.join(ANS_DIR, file), "r", encoding="utf-8") as fa:
                answers = json.load(fa)

            answers_by_qid = {a["qid"]: a for a in answers}

            for difficulty in ["easy", "medium", "hard"]:
                for q in questions["easy"]:
                    qid = q["qid"]

                    if qid not in answers_by_qid:
                        continue

                    fa = answers_by_qid[qid]   # ← THIS is the answer object
                    golden = q["golden_answer"]

                    for ans_type in ["short_answer", "long_answer"]:
                        if ans_type not in fa:
                            continue

                        user_ans = fa[ans_type]   # ✅ FIXED
                        orig_len = len(user_ans.split())

                        used_text = truncate(user_ans)
                        used_len = len(used_text.split())

                        sim = util.cos_sim(
                            model.encode(golden),
                            model.encode(used_text)
                        )[0][0].item()

                        writer.writerow([
                            qid,
                            topic,
                            model_name,
                            "truncation",
                            ans_type,
                            orig_len,
                            used_len,
                            round(sim, 4)
                        ])



print("\n Method-1 Truncation Experiment Completed")
print(f" Results saved to: {OUT_FILE}")
