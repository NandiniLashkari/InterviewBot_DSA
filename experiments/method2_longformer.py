import json, os, csv
import torch
from transformers import LongformerTokenizer, LongformerModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

def embed(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )
    with torch.no_grad():
        return model(**inputs).last_hidden_state.mean(dim=1)

OUT_FILE = "experiments/results/method2_results.csv"

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "qid", "topic", "method",
        "answer_type", "token_length",
        "cosine_similarity"
    ])

    for file in os.listdir("data"):
        topic = file.replace(".json", "")
        qdata = json.load(open(f"data/{file}", "r"))
        adata = json.load(open(f"synthetic_answers/{file}", "r"))

        for q, a in zip(qdata["easy"], adata["easy"]):
            gold_vec = embed(q["golden_answer"])

            for t in ["short_answer", "long_answer"]:
                user_vec = embed(a[t])
                score = cosine_similarity(
                    gold_vec.numpy(),
                    user_vec.numpy()
                )[0][0]

                writer.writerow([
                    q["qid"], topic,
                    "longformer",
                    t,
                    len(a[t].split()),
                    round(score, 4)
                ])
