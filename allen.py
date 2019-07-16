
###
# Using the pre-trained allenNLP Textual Entailment model to generate the label
# from selected evidences for each claim.
###
import json
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")

labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

sentence_dict = {}
prefix = "./wiki-pages-text/wiki-"
suffix  = ".txt"
for i in range(109):
    if i <9:
        wikiNum = "00"+str(i+1)
    elif i<99:
        wikiNum = "0"+str(i+1)
    else:
        wikiNum = str(i+1)
    filename = prefix+wikiNum+suffix
    for line in open(filename):
        
        doc_title,sent_id,sent_content = line.split(" ",2)
        if sent_id.isalpha():
            pass
        else:
            sentence_dict[(doc_title,sent_id)] = sent_content

print("===========start predict===========")
def getSentence(evidence_list, sent_dict):
	sentences = []
	for evidence in evidence_list:
		try:
			sentences.append(sent_dict[(evidence[0],str(evidence[1]))])
		except Exception as e:
			pass
	if len(sentences) < 1:
		sentences = ["place holder"]

		
	return sentences

with open('unlabel_evidence_23.json', 'r') as f:
	data = json.load(f)
print(len(data))

test_allen = {}
processed = 0
not_label = 0
for claim_id in data:
	print(claim_id)
	test_allen[claim_id] = {}
	test_allen[claim_id]['claim'] = data[claim_id]['claim']
	test_allen[claim_id]['label'] = ""
	test_allen[claim_id]['evidence'] = data[claim_id]['evidence']
	label_index = 2
	evidence_list = data[claim_id]['evidence']
	if len(evidence_list) == 0:
		label_index = 2
		not_label += 1
		test_allen[claim_id]['label'] = labels[label_index]
	else:
		
		evidences = getSentence(evidence_list,sentence_dict)
	
		
		input_sent = ""
		for evidence in evidences:
			input_sent += evidence
		res_prob = predictor.predict(hypothesis=test_allen[claim_id]['claim'],premise=input_sent)['label_probs']
		print(res_prob)
		
		if res_prob.index(max(res_prob)) == 2:
			if res_prob[0]>res_prob[1]:
				label_index = 0
			else:
				label_index = 1
		else:
			label_index = res_prob.index(max(res_prob))
		test_allen[claim_id]['label'] = labels[label_index]
	print(test_allen[claim_id]['label'])
	processed += 1
	print(processed)
		

json_str = json.dumps(test_allen)
with open('testoutput_1.json', 'w') as json_file:
    json_file.write(json_str)
print(not_label)



