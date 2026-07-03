import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

load_dotenv()
hft = os.environ.get("HF_TOK")

# init
# ======
model_name = "google/gemma-3-1b-pt"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hft)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading {model_name} on {device}...")
model = (
    AutoModel.from_pretrained(
        model_name, token=hft, output_hidden_states=True, dtype=torch.bfloat16
    )
    .eval()
    .to(device)
)


# poolings
# =========
def last_token_pool(last_hidden_state, attention_mask):  # right-padding.
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden_state.size(0))
    return last_hidden_state[batch_idx, seq_lengths]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def attn_weighted_pool_heuristic(last_hidden_state, attention_mask):
    # weights each token by its l2 norm before summing #

    norms = last_hidden_state.norm(dim=-1)
    norms = norms.masked_fill(attention_mask == 0, float("-inf"))
    weights = torch.softmax(norms, dim=-1).unsqueeze(-1)
    return (last_hidden_state * weights).sum(dim=1)


# extract & eval loop
# ====================
def embed_sentences(sentences, pool_fn, batch_size=32):
    all_embeds = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding"):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)

        pooled = pool_fn(out.last_hidden_state, inputs["attention_mask"])
        # bfloat16 -> float32
        all_embeds.append(pooled.float().cpu())

    return torch.cat(all_embeds, dim=0)


def evaluate(embeds1, embeds2, gold_scores):
    # pairwise cosine similarity
    cos_sims = F.cosine_similarity(embeds1, embeds2, dim=-1).numpy()

    # corr
    spearman = spearmanr(cos_sims, gold_scores).correlation
    pearson = pearsonr(cos_sims, gold_scores)[0]
    return spearman, pearson


if __name__ == "__main__":
    print("Loading STS-B dataset...")
    dataset = load_dataset("sentence-transformers/stsb", split="validation", token=hft)

    sentences1 = dataset["sentence1"]
    sentences2 = dataset["sentence2"]
    gold_scores = dataset["score"]

    poolers = {
        "Last-Token": last_token_pool,
        "Mean Pooling": mean_pool,
        "Heuristic Attention": attn_weighted_pool_heuristic,
    }

    results = {}

    for name, pool_fn in poolers.items():
        print(f"\nEvaluating {name}...")
        embeds1 = embed_sentences(sentences1, pool_fn)
        embeds2 = embed_sentences(sentences2, pool_fn)

        spearman, pearson = evaluate(embeds1, embeds2, gold_scores)
        results[name] = {"Spearman": spearman, "Pearson": pearson}

    print("\n| Method | Spearman | Pearson |")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['Spearman']:.4f} | {metrics['Pearson']:.4f} |")
