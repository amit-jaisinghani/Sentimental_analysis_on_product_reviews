"""
Author: Ishaan Thakker
Program to scrape amazon's data
"""
from bs4 import BeautifulSoup
import requests

count = 1
id = 0
f1 = open('unlabeled.json','x')
json_string = '['
csv_str = ''
row_count = 1
for i in range(300,450):
	with open('html_files/'+str(i)+'.html') as f:
		soup = BeautifulSoup(f, 'lxml')
	#articles = soup.find_all('div', class_='article')

	title = soup.find('div',class_='a-row product-title').find('h1').find('a').text
	reviews = soup.find_all(attrs={"data-hook","review"})


	print(count)
	count+=1
	
	ind = 0
	for review in reviews:
		if True:	
			l = len(reviews) 
					
			#user_name = review.find('div').find('div').find('a').find('span').text.strip()
			rating = review.find('div').find('span',class_='a-icon-alt').text.strip()
			#review_date = review.find('div').find('span', class_='review-date').text.strip()
			if float(rating.split(' ')[0])>0.0:
				json_string+='{'		
				print(row_count)
				row_count+=1	
				review_title = review.find('div').find('a',class_='a-size-base a-link-normal review-title a-color-base a-text-bold').text.strip().replace('"','').replace(',','')
				review_data = review.find('div').find('span',class_='a-size-base review-text').text.strip().replace('"','').replace(',','')	
				json_string+= '"id":"'+str(id)+'","review_title":'+'"{}"'.format(review_title)+',"review_data":'+'"{}"'.format(review_data)+',"polarity":""'
				id+=1
				csv_str+='"'+review_title+'","'+review_data+'",""'
				csv_str+='\n'
				json_string+='\n'
			#f1.write("\""+title+"\",\""+user_name+"\",\""+rating+"\","+review_date+"\",\""+review_title+"\",\""+review_data+"\"\n")
				json_string+='}'
				if True:
					json_string+=','	
			ind+=1
		
json_string +=']'
f1.write(json_string)
f.close()
