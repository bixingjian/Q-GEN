# 2 passages from te. It is for dis and que gen. 10 passages from the race dataset. That is for metric.
import pandas as pd
test_df = pd.read_csv("./dataset/race/race_test_df.csv")

te_list = [
'''Perhaps no company embodies the ups and downs of Chinese big tech better than its biggest tech firm of all — Tencent. \
Two years ago the online empire seemed unstoppable. More than a billion Chinese were using its ubiquitous services to pay, play and do much else besides. \
Its video games, such as “League of Legends”, were global hits. \
Tencent’s market value exceeded $900bn, and the firm was on track to become China’s first trillion-dollar company. \
Then the Communist Party said, enough. \
Xi Jinping, China’s paramount leader, decided that big tech’s side-effects, from distracted teenagers to the diversion of capital from strategically important sectors such as semiconductors, were unacceptable. \
Tencent was, along with the rest of China’s once-thriving digital industry, caught up in a sweeping 18-month crackdown.  '''
,

'''It might have been a one-off. svb’s business—banking for techies—was unusual. \
Most clients were firms, holding in excess of the $250,000 protected by the Federal Deposit Insurance Corporation (fdic), a regulator. \
If the bank failed they faced losses. And svb used deposits to buy long-dated bonds at the peak of the market. \
“One might have supposed that Silicon Valley Bank would be a good candidate for failure without contagion,” says Larry Summers, a former treasury secretary. \
Nevertheless, withdrawal requests at other regional banks in the following days showed “there was in fact substantial contagion.'''
]

context_list = [] # the context
for i in range(10):
    context_list.append(test_df.iloc[i][0])

question_list = []
for i in range(10):
    question_list.append(test_df.iloc[i][1])

correct_answer_list = []
for i in range(10):
    correct_answer_list.append(test_df.iloc[i][2])

incorrect_answer_list = []
for i in range(10):
    temp = []
    temp.append(test_df.iloc[i][3])
    temp.append(test_df.iloc[i][4])
    temp.append(test_df.iloc[i][5])
    incorrect_answer_list.append(temp)



generated_distractor_list = [['Help her to get up early', 'Help her to find a job', 'Help her to recover from the flood'], 
                             ['Nancy climbed into the school gym and took her to the school.', "Nancy climbed into the helicopter but didn't catch it any more", 'Nancy climbed into the emergency shelter after the flood disappeared'], 
                             ['They rescued them from the helicopter.', 'They took them to the school gym and set up an emergency shelter', 'They raised them into the helicopter'], 
                             ['the clothes that we choose to wear have something to do with our values and lifestyles', 'the clothes that we choose to wear are of no importance', 'the clothes that we choose to wear have nothing to do with our values and lifestyles'], 
                             ['men were not interested in what they wore while women did', "women were not interested in what they wore when men didn't", 'women were very proud of what they wore'], 
                             ['they are not interested in fashion', "they don't want to be judged", 'they think their values and lifestyles are more important'], 
                             ['people have changed their clothes to make them more attractive', 'people are more aware of the importance of dress', 'women wear more dresses than men'], 
                             ['was very busy with his study', 'kissed his mother hello', 'was very hard at math'], 
                             ['Tommy kissed his mother hello', 'Tommy was a catholic school', 'Tommy was not good at math'], 
                             ['mistakes are not easy', 'mistakes can be made easily', 'mistakes are not easy to make']]

# print(context_list[0])
# print(question_list[0])
# print(correct_answer_list[0])
# print(incorrect_answer_list[0])