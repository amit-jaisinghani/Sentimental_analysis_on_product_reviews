import json

with open('unlabeled.json', 'r') as fp:
    obj = json.load(fp)

s = ''
ind =0
negative = 0 
positive = 0
neutral = 0
for item in obj:
	print('here')
	if True:
		s+='"'+item['review_title'].replace('"','')+'","'+item['review_data'].replace('"','')+'","'+item['polarity'].replace('"','')+'"'
		s+='\n'
		if item['polarity'] == 'positive':
			positive += 1
		elif item['polarity'] == 'negative':
			negative += 1 
		else:
			neutral += 1		
		print(item['review_data'])
		ind+=1
		print(ind)
f = open('unlabeled.csv','x')
f.write(s)
print(positive, negative, neutral)
