from sentence_transformers import SentenceTransformer
from matcher import match_score

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text)

def comp_embs(job_emb,res): #job_emb is the embedding of the job description, res is a dictionary of resumes
    emb = {}
    match = {}
    for filename, text in resumes.items():
        emb[filename] =  get_embedding(text) 
        match[filename] = match_score(emb[filename], job_emb)
        
    return emb,dict(sorted(match.items(), key=lambda item: item[1], reverse=True))