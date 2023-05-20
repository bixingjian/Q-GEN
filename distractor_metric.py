from context_list import context_list, question_list, correct_answer_list, incorrect_answer_list, generated_distractor_list
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore') # to ignore warning in 

model = SentenceTransformer('bert-base-nli-mean-tokens') #using a relatively smaller size model from the api

# scores for the original distractor
option_dissim = [] # [ [ six scores ] ]
context_sim = []
sim_score = []
sim_index = []
sim_sent = []
bleu_score = []
for i in range(10):
    context = context_list[i].replace(".", ". ") 
    context = context.replace(" .", ". ") 
    context_sens = nltk.sent_tokenize(context) # [" ", " "]
    # print(context_sens)
    # print("========="*20)
    # for i in range(len(context_sens)):
    #     print("**********")
    #     print(context_sens[i])

    # bleu score for the original and generated.
    temp = []
    for j in range(3):
        temp.append(nltk.translate.bleu_score.sentence_bleu([word_tokenize(incorrect_answer_list[i][j])], word_tokenize(correct_answer_list[i]), weights=(1.0, 0, 0, 0)))
    for k in range(3):
        temp.append(nltk.translate.bleu_score.sentence_bleu([word_tokenize(generated_distractor_list[i][k])], word_tokenize(correct_answer_list[i]), weights=(1.0, 0, 0, 0)))
    bleu_score.append(temp)

    # score for disim
    sentences = []
    sentences.append(correct_answer_list[i])
    for j in range(3):
        sentences.append(incorrect_answer_list[i][j])
    for k in range(3):
        sentences.append(incorrect_answer_list[i][k])
    sentence_embeddings = model.encode(sentences)
    res = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
    res = res.tolist()
    option_dissim.append([1-x for x in res[0]])
    assert len(res[0]) == 6
    

    # score for context sim
    temp = []
    temp_sim_score = []
    temp_sim_index = []
    temp_sim_sent = []
    for j in range(3):
        sentences = []
        sentences.append(incorrect_answer_list[i][j])
        sentences += context_sens
        sentence_embeddings = model.encode(sentences)
        res = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
        res = res.tolist()
        res = [[round(x, 3) for x in res[0]]]
        temp.append(max(res[0]))
        # print(res[0])
        temp_sim_score.append(res[0])
        # print("round_ori: ", i, j+1, "\nindex: ", res[0].index(max(res[0])), "\n", context_sens[res[0].index(max(res[0]))])
        temp_sim_index.append(res[0].index(max(res[0])))
        temp_sim_sent.append(context_sens[res[0].index(max(res[0]))])


    for k in range(3):
        sentences = []
        sentences.append(generated_distractor_list[i][k])
        sentences += context_sens
        sentence_embeddings = model.encode(sentences)
        res = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
        res = res.tolist()
        res = [[round(x, 3) for x in res[0]]]
        temp.append(max(res[0]))
        # print(res)
        temp_sim_score.append(res[0])
        # print("round_g: ", i, k+1, "\nindex: ", res[0].index(max(res[0])), "\n",context_sens[res[0].index(max(res[0]))])
        temp_sim_index.append(res[0].index(max(res[0])))
        temp_sim_sent.append(context_sens[res[0].index(max(res[0]))])

    context_sim.append(temp)
    sim_score.append(temp_sim_score)
    sim_index.append(temp_sim_index)
    sim_sent.append(temp_sim_sent)

        




alpha = 0.5
beta = 1 - alpha

for i in range(10):
    print(i, "*"*30)
    print("CONTEXT: \n", context_list[i], "\n")
    print("QUESTION: \n", question_list[i], "\n")
    print("CORRECT ANSWER: \n", correct_answer_list[i], "\n")
    for j, (dissim, sim, bleu, index, score, sent) in enumerate(zip(option_dissim[i],context_sim[i], bleu_score[i], sim_index[i], sim_score[i], sim_sent[i])):
        if (j <3):
            print("-- original ",j+1, ": ", incorrect_answer_list[i][j])
            print("option_dissim: %.3f |" %float(dissim), "context_sim:  %.3f |" %float(sim), "final score:  %.3f | " %float(alpha * dissim + beta * sim))
            print("bleu_score: %.3f" %bleu)
            print("sim index: ", index)
            print("sim score: ", score)
            print("sim sent: ", sent)
            print("*"*10)   
        else:
            print("-- generated ",j-2, ": ", generated_distractor_list[i][j-3])
            print("option_dissim: %.3f |" %float(dissim), "context_sim:  %.3f |" %float(sim), "final score:  %.3f | " %float(alpha * dissim + beta * sim))
            print("bleu_score: %.3f" %bleu)
            print("sim index: ", index)
            print("sim score: ", score)
            print("sim sent: ", sent)
            print("*"*10)   

# res = 0
# for i in range(10):
#     res += bleu_score[i][2]

# print(res/10)

    
    






