import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# 1. Sample ICD-10 Diagnoses
# ----------------------------
diagnoses = [
    ("I21", "Myocardial infarction"),
    ("I20", "Angina pectoris"),
    ("I63", "Stroke"),
    ("I61", "Intracerebral hemorrhage"),
    ("J18", "Pneumonia"),
    ("J15", "Bacterial pneumonia"),
    ("J44", "COPD"),
    ("J45", "Asthma"),
    ("E11", "Type 2 diabetes"),
    ("E10", "Type 1 diabetes")
]

df = pd.DataFrame(diagnoses, columns=["ICD10", "Diagnosis"])
names = df["Diagnosis"].tolist()
print("Original Diagnosis:\n ", df)

# ----------------------------
# 2. Load BioBERT Model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def get_embedding(text):
    """Get BioBERT embedding for a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling across tokens
    return outputs.last_hidden_state.mean(dim=1).numpy()

# ----------------------------
# 3. Compute Similarity Matrix
# ----------------------------
embeddings = np.vstack([get_embedding(n) for n in names])
similarity_matrix = cosine_similarity(embeddings)
sim_df = pd.DataFrame(similarity_matrix, index=names, columns=names)
print("\nSemantic similarity between diagnoses:\n", sim_df.round(2))

# ----------------------------
# 4. Find Cluster Peers Automatically
# ----------------------------
similarity_threshold = 0.75  # tweak as needed

def get_cluster_peers(diag):
    """Return diagnoses in the same semantic cluster as diag."""
    sims = sim_df.loc[diag]
    return [d for d, score in sims.items() if score >= similarity_threshold and d != diag]

# ----------------------------
# 5. Build PRAM Probability Matrix
# ----------------------------
pram_matrix = pd.DataFrame(0, index=names, columns=names, dtype=float)

for diag in names:
    # 80% keep the same
    pram_matrix.loc[diag, diag] = 0.80
    
    # 15% swap to similar (same cluster)
    cluster_peers = get_cluster_peers(diag)
    if cluster_peers:
        prob_each = 0.15 / len(cluster_peers)
        for peer in cluster_peers:
            pram_matrix.loc[diag, peer] = prob_each
    
    # 5% swap to unrelated
    unrelated = [d for d in names if d != diag and d not in cluster_peers]
    if unrelated:
        prob_each = 0.05 / len(unrelated)
        for u in unrelated:
            pram_matrix.loc[diag, u] = prob_each

print("\nPRAM Probability Matrix:\n", pram_matrix.round(2))

# ----------------------------
# 6. Apply PRAM to Fake Dataset
# ----------------------------
def pram_replace(original):
    """Replace a diagnosis according to PRAM probabilities."""
    probs = pram_matrix.loc[original].values
    return np.random.choice(names, p=probs)

# Example fake patient diagnoses
patient_data = ["Myocardial infarction", "Stroke", "Asthma", "Type 2 diabetes"]

print("\nOriginal diagnoses:", patient_data)
print("Anonymized diagnoses:", [pram_replace(d) for d in patient_data])
