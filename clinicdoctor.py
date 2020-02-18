#install package NLTK 
pip install nltk 

#install the newspaper3k package
pip install newspaper3k

#Import libraries
from newspaper import article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

#Ignore any future warning 
warnings.filterwarnings('ignore')

#Ignore any future warning 
warnings.filterwarnings('ignore')

#get the text from any article and i will be using mayo clicnc for cancer/ kidney chat bot
article = Article('https://www.mayoclinic.org/search/search-results?q=kidney')
article.download()
article.parse()
article.nlp()
corpus = article.text

#print corpus
print(corpus)

#Tokenization
text = corpus
sent_token = nltk.sent_tokenize(text) #converrting the text into lit of sentences

#now print the text in a list
print(sent_token)

#Remove punctuations in the sentences using the dictionary library
remove_punct_dic = dict( (ord(punct),None) for punct in string.punctuation)

#print the punctuation tobe removed
print(remove_punct_dic)

#print the punctuations to be removed 
print(string.punctuation)

#to print the sTring as numbers use
#ode(punct) as a method parameter

#Function to return lower case words after removing the punctuations
def LemNormalize(text):
  return nltk.word_tokenize(text.lower().translate(remove_punct_dic))

#print the words
print(LemNormalize(text))

#to make them print in small letters use
#text.lower()


#to remove the punctuations we use .translate(remove_punct_dict)

#create some key words
#Greeting inputs 

GREETING_INPUT = ["hey", "hello","greetings","greeting","watsup","niaje","morning","afternoon","evening","vipi"]
#sample Response

GREETING_RESPONSE = ["morning too","evening too","afternoon too","poa sana","hey too","greeting","what's good","hey there","Hope uko poa"]
#function to return random greeting to the user,

def greeting(sentence):
  #if user is input, return any respose to user
  for word in sentence.split():
    if word.lower() in GREETING_INPUT:
      return random.choice(GREETING_RESPONSE)

#getting sample user response
user_response = 'what is chronic kidney disease'
user_response = user_response.lower() #make the response to lower case
###print the user response

print(user_response)
#create a var and set the robot response to Empty string

robo_response=''
#append the response to user list 

sent_token.append(user_response)
###print the list to see if the data is appended to the list 

print(sent_token)
#Now create a TdifVectorizer
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

###Convert the text to a matrix TF-IDF features
tfidf=TfidfVec.fit_transform(sent_token)

###print the tdif feature
print(tfidf)

#get the measure of similarity
vals = cosine_similarity(tfidf[-1],tfidf)

#print similarity score 
print(vals) 

# get the response to the most similar user response
idx = vals.argsort()[0][-2]

#reduce domentionality od vals
flat = vals.flatten()

#sort the list is ascendign oder
flat.sort()

#get the most similar score to the user response
score = flat[-2]
print(score)

#if the var score is 0 then there is no text similar to the user response
if(score==0):
  robo_response = robo_response + " Hey I dont get you well.How may I help you?"
else:
    robo_response = robo_response + sent_token[idx]
print(robo_response)

#set back the response
sent_token.remove(user_response)
