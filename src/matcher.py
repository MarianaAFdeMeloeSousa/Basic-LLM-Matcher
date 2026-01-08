from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#match the numbers
def match_score(resume_emb, job_emb):
    score = cosine_similarity(
        resume_emb.reshape(1, -1),
        job_emb.reshape(1, -1)
    )[0][0]
    return round(score * 100, 2)  #round percentually