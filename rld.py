from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

x =["Measure the depth of the river", "Look for a fallen tree trunk", "Run away from the flooded farm", "Help her to get up early", "Help her to find a job", "Help her to recover from the flood"]

a,_ = model(x)

print(a)



