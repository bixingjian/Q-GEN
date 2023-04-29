from context_list import context_list, question_list, correct_answer_list, incorrect_answer_list, generated_distractor_list
import math
import re
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") #using a relatively smaller size model from the api

# scores for the original distractor
option_dissim = []
context_sim = []
bleu_score = []
for i in range(10):
    context = context_list[i]
    context_sens = nltk.sent_tokenize(context)
    correct_answer = list(correct_answer_list[i])
    incorrect_answers = incorrect_answer_list[i] # change here for incorrect and distractor

    # bleu score for the original distractor
    sent_tokenized = word_tokenize(question_list[i])
    # temp = []
    # for _ in range(len(incorrect_answer_list[i])):
    #     answer_tokenized = word_tokenize(incorrect_answers[_])
    #     temp_score = nltk.translate.bleu_score.sentence_bleu(sent_tokenized, answer_tokenized, weights=(1, 0, 0, 0), smoothing_function=None)
    #     temp.append(temp_score)
    # bleu_score.append(temp)
    # incorrect_answer_tokenized = [word_tokenize(incorrect_answers[_]) for _ in range(len(incorrect_answers))]
    # answer_score = nltk.translate.bleu_score.sentence_bleu(incorrect_answer_tokenized, sent_tokenized, weights=(1.0, 0, 0, 0), smoothing_function=None)
    temp = []
    temp.append(nltk.translate.bleu_score.sentence_bleu(word_tokenize(incorrect_answer_list[i][0]), correct_answer, weights=(1.0, 0, 0, 0)))
    temp.append(nltk.translate.bleu_score.sentence_bleu(word_tokenize(incorrect_answer_list[i][1]), correct_answer, weights=(1.0, 0, 0, 0)))
    temp.append(nltk.translate.bleu_score.sentence_bleu(word_tokenize(incorrect_answer_list[i][2]), correct_answer, weights=(1.0, 0, 0, 0)))
    bleu_score.append(temp)

    # Compute encodings for option in_sim
    correct_answer_vectors = model.encode(correct_answer_list[i])
    incorrect_answers_vectors = model.encode(incorrect_answer_list[i])

    cosine_scores1 = util.cos_sim(correct_answer_vectors, incorrect_answers_vectors)
    temp = []
    for _ in range(len(incorrect_answers_vectors)):
        temp.append(1 - cosine_scores1[0][_])
    option_dissim.append(temp)

    # computing 
    context_sens_vectors = model.encode(context_sens)
    cosine_scores2 = util.cos_sim(context_sens_vectors, incorrect_answers_vectors)

    temp = []
    for _ in range(len(incorrect_answers_vectors)):
        temp_max_score = 0;
        for __ in range(len(context_sens_vectors)):
            if cosine_scores2[__][_] > temp_max_score:
                temp_max_score = cosine_scores2[__][_]
        temp.append(temp_max_score)
    context_sim.append(temp)


    # print(option_dissim)
    # print(context_sim)


# scores for the generated
gen_option_dissim = []
gen_context_sim = []
gen_bleu_score = []
for i in range(10):
    context = context_list[i]
    context_sens = nltk.sent_tokenize(context)
    correct_answer = list(correct_answer_list[i])
    incorrect_answers = generated_distractor_list[i] # change here for incorrect and distractor

    # compute gen bleu score
    temp = []
    temp.append(nltk.translate.bleu_score.sentence_bleu(word_tokenize(generated_distractor_list[i][0]), correct_answer, weights=(1.0, 0, 0, 0)))
    temp.append(nltk.translate.bleu_score.sentence_bleu(word_tokenize(generated_distractor_list[i][1]), correct_answer, weights=(1.0, 0, 0, 0)))
    temp.append(nltk.translate.bleu_score.sentence_bleu(word_tokenize(generated_distractor_list[i][2]), correct_answer, weights=(1.0, 0, 0, 0)))
    gen_bleu_score.append(temp)


    # Compute encodings for option in_sim
    correct_answer_vectors = model.encode(correct_answer_list[i])
    incorrect_answers_vectors = model.encode(generated_distractor_list[i])

    cosine_scores1 = util.cos_sim(correct_answer_vectors, incorrect_answers_vectors)
    temp = []
    for _ in range(len(incorrect_answers_vectors)):
        temp.append(1 - cosine_scores1[0][_])
    gen_option_dissim.append(temp)

    # computing 
    context_sens_vectors = model.encode(context_sens)
    cosine_scores2 = util.cos_sim(context_sens_vectors, incorrect_answers_vectors)

    temp = []
    for _ in range(len(incorrect_answers_vectors)):
        temp_max_score = 0;
        for __ in range(len(context_sens_vectors)):
            if cosine_scores2[__][_] > temp_max_score:
                temp_max_score = cosine_scores2[__][_]
        temp.append(temp_max_score)
    gen_context_sim.append(temp)

    # print(option_dissim)
    # print(context_sim)


alpha = 0.3
beta = 1 - alpha

print(bleu_score)
print("+++" *50 )
print(gen_bleu_score)


for i in range(10):
    print(i, "*"*30)
    print("CONTEXT: \n", context_list[i], "\n")
    print("QUESTION: \n", question_list[i], "\n")
    print("CORRECT ANSWER: \n", correct_answer_list[i], "\n")
    for j, (dissim, sim) in enumerate(zip(option_dissim[i],context_sim[i])):
        print("distracor ",j, ": ", incorrect_answer_list[i][j], "\n","option_dissim: %.2f |" %float(dissim), "context_sim:  %.2f |" %float(sim), "final score:  %.2f | " %float(alpha * dissim + beta * sim))
        print("bleu_score: %.2f" %bleu_score[i][j])
        # print(j+1, ": ", incorrect_answer_list[i][j], "\n")
        print("*"*10)
    for k, (gen_dissim, gen_sim) in enumerate(zip(gen_option_dissim[i], gen_context_sim[i])):
        print("gen_distracor ",k, ": ", generated_distractor_list[i][k], "\n", "gen_option_dissim: %.2f |" %float(gen_dissim), "gen_context_sim:  %.2f |" %float(gen_sim), "gen_final_score:  %.2f | " %float(alpha * gen_dissim + beta * gen_sim))
        print("gen_blue_score: %.2f" %gen_bleu_score[i][k])
        # print(k+4, ": ", generated_distractor_list[i][k], "\n")
        print("*"*10)
    
    

