from embeddings import get_embedding
from embeddings import comp_embs
from pathlib import Path

BASE_DIR = Path.cwd().parent

# Job description
job_path = BASE_DIR / "define" / "jobdesc.txt"
with open(job_path, "r", encoding="utf-8") as f:
    job_text = f.read()

# Resumes
resumes = {}
data_dir = BASE_DIR / "data"

for path in data_dir.glob("*.txt"):
    with open(path, "r", encoding="utf-8") as f:
        resumes[path.name] = f.read()



# Get Job embedding

job_emb = get_embedding(job_text)

# Get Resumes Embeddings + how well they match with the job description
emb_res, matches = comp_embs(job_emb,resumes)



print(f"The resume with the highest score, {list(matches.values())[0]}%, was {list(matches.keys())[0]}, followed by {list(matches.keys())[1]}, {list(matches.keys())[2]} and {list(matches.keys())[3]}.")
