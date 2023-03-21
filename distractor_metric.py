from context_list import context_list, correct_answer_list, incorrect_answer_list
import math
import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import nltk

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") #using a relatively smaller size model from the api

# make sure they are all lists!
context = context_list[2]
print("context :", context)
context_sens = nltk.sent_tokenize(context)
correct_answer = list(correct_answer_list[2])
print("correct answer: ", correct_answer_list[2], "\n")
incorrect_answers = incorrect_answer_list[2]
option_insim = []
context_sim = []


# Compute encodings for option in_sim
correct_answer_vectors = model.encode(correct_answer_list[2])
incorrect_answers_vectors = model.encode(incorrect_answer_list[2])

cosine_scores1 = util.cos_sim(correct_answer_vectors, incorrect_answers_vectors)
temp = []
for i in range(len(incorrect_answers_vectors)):
    temp.append(1 - cosine_scores1[0][i])
option_insim.append(temp)

# computing 
context_sens_vectors = model.encode(context_sens)
cosine_scores2 = util.cos_sim(context_sens_vectors, incorrect_answers_vectors)

temp = []
for i in range(len(incorrect_answers_vectors)):
    temp_max_score = 0;
    for j in range(len(context_sens_vectors)):
        if cosine_scores2[j][i] > temp_max_score:
            temp_max_score = cosine_scores2[j][i]
    temp.append(temp_max_score)
context_sim.append(temp)

# print(option_insim)
# print(context_sim)

alpha = 0.3
beta = 1 - alpha

for i, (insim, sim) in enumerate(zip(option_insim[0],context_sim[0])):
    print("distracor ", i, ": ", incorrect_answer_list[2][i], "option_insim: ", float(insim), "context_sim: ", float(sim), "final score: ", float(alpha * insim + beta * sim))
    print("="*20)
    







