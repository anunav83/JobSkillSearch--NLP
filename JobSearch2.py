import bs4 as bs
import urllib.request as rq
import urllib.parse as par
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
import re
#pip install -U gensim
from gensim.models import Word2Vec 


# unused packages for later use
#from nltk.collocations import BigramCollocationFinder
#from nltk.metrics import BigramAssocMeasures
#from nltk.collocations import TrigramCollocationFinder
#from nltk.metrics import TrigramAssocMeasures
#from nltk import FreqDist
#from nltk.corpus import wordnet
#import itertools
#from itertools import chain



# Below are the URLS, Keyword is words we are searching the job site
# Source AJE is American Jobs Exchange
# Source DEA is NLx
# Source CB is Career Builder
# Source  IN is Indeed .com
JobListFilePath = "C://Users//anupama//Desktop//Jobs//JobList.html"
FinalDF = "C://Users//anupama//Desktop//Jobs//df_JobsTable.csv"
ModelLocation = "C://Users//anupama//Desktop//Jobs//model//mymodel.sav"

jobSources=["AJE","CB","IN"]

keywords ="data scientist"
JobtableAppend = ""
for JobSource in jobSources:
    if (JobSource =="AJE"):
        pagesize = 500
    else:
        pagesize = 100
        
    url = "https://www.careeronestop.org/Toolkit/Jobs/find-jobs.aspx"
    mydict = {'source': JobSource,'pagesize': pagesize,"keyword": keywords,"ajax":"0","location":"94016","radius":"10" }
    
    #,"sortcolumns":"accquisitiondate","sortdirections":"DSC"
    data = par.urlencode(mydict)
    url = url + "?" + data
    print(url)
    
    sauce = rq.urlopen(url)
    soup = bs.BeautifulSoup(sauce,'lxml')
    
    # Extract Entire job table
    Jobtable = soup.find('table', {'class': 'res-table'})
    
    # we donot need the Table Header as it is redundant.
    
    for b in Jobtable.find_all("thead"):
        b.replaceWith("")
    
    for b in Jobtable.find_all("img"):
        b.replaceWith("")
        
    allrows = Jobtable.findAll('tr')
    rowhtml = ' '.join([row.prettify() for row in allrows])
    
    rowhtml = "<tr><td colspan=4> Job Source = " + JobSource +"</td></tr>" + rowhtml
        
    JobtableAppend = JobtableAppend + rowhtml
    
rq.urlcleanup()        
# Convert the html table to Pandasframe
JobtableAppend ="<html><table><tr><td>Title</td><td>Company</td><td>Location</td><td>PostingDt</td></tr>" + JobtableAppend + "</table></html>"
df_JobsTable = pd.read_html(JobtableAppend, header=0)
# Read_html givea list of Dataframes. in our case we have only 1 DF
df_JobsTable = df_JobsTable[0]

with open(JobListFilePath, "a") as myfile:
    myfile.write(JobtableAppend)
    myfile.close()

# List of Employers/Cities
df_JobsTable = df_JobsTable.apply(lambda x: x.astype(str).str.lower())
lstJobPostingsByEmployers = df_JobsTable.Company.value_counts()
lstcities = df_JobsTable.Location.value_counts()


# Getting Each Job Link

html = open(JobListFilePath).read()
soup = bs.BeautifulSoup(html,'lxml')
table = soup.find("table")



# NLTK's default english stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))
default_stopwords.add('br')
default_stopwords.add('hiringcompanylnk')
default_stopwords.add('href')
default_stopwords.add('http')

# Remove rows in the Data frame that contains text link " job source = aje" the source column will
# represent the cource of the posting

df_JobsTable= df_JobsTable[~df_JobsTable.Title.str.contains("job source")]
df_JobsTable.reset_index(inplace = True, drop=True)

is_looping = True
AllWords = []
Jobcounter = 0 
df_JobsTable["keywords"]= ""
df_JobsTable["FullTime"] = ""
df_JobsTable["ExperienceLevel"] =""
df_JobsTable["Education"]=""
df_JobsTable["Source"]=""
df_JobsTable["link"]=""
df_JobsTable["JobID"]=""

for table_row in table.findAll('tr'):
# Get link of each Job in the HTML FIle
    link = table_row.find('a',href=True)
    WOneJob =[]
    WOneLine =[]
    fulltime = ""
    experience = ""
    education = ""
    if link:
        sauce = rq.urlopen(link['href'])
        soup = bs.BeautifulSoup(sauce,'lxml')
        if len(link['href'].split('careerbuilder'))> 1 :
            next_s = soup.find('div', {'id': 'jdp_description'})
            if next_s != None :
                next_s = next_s.find('div',{'class' :'col big col-mobile-full'})
                for br in next_s.findChildren(recursive=True):
                    next_s = br.get_text().lower()
                    tokenizer = RegexpTokenizer(r'\w+')
                    next_s = re.sub('[^A-Za-z0-9]+', ' ', next_s).strip()
                    filterdText=tokenizer.tokenize(str(next_s).strip())
                    if len(filterdText) > 0:
                        #Remove stopwords
                        WOneLine = [word for word in filterdText if word not in default_stopwords]
                        WOneJob.extend(WOneLine)
            #WOneJob = list(set(WOneJob))
            
            df_JobsTable["keywords"][Jobcounter] = ','.join(WOneJob)
            df_JobsTable["FullTime"][Jobcounter] = fulltime
            df_JobsTable["ExperienceLevel"][Jobcounter] = experience
            df_JobsTable["Education"][Jobcounter] = education
            df_JobsTable["Source"][Jobcounter] = "CB"
            df_JobsTable["link"][Jobcounter] = link['href']
            df_JobsTable["JobID"][Jobcounter]  = link['href'].split("DID=")[1]
            AllWords.append(WOneJob)
        
        if len(link['href'].split('indeed'))> 1 :
            next_s = soup.find('div',{'id':'jobDescriptionText'})
            if next_s != None :
                for br in next_s.findChildren(recursive=True):
                    next_s = br.get_text().lower()
                    tokenizer = RegexpTokenizer(r'\w+')
                    next_s = re.sub('[^A-Za-z0-9]+', ' ', next_s).strip()
                    filterdText=tokenizer.tokenize(str(next_s).strip())
                    if len(filterdText) > 0:
                        #Remove stopwords
                        WOneLine = [word for word in filterdText if word not in default_stopwords]
                        WOneJob.extend(WOneLine)
            #WOneJob = list(set(WOneJob))
           # Jobcounter = Jobcounter + 1
            df_JobsTable["keywords"][Jobcounter] = ','.join(WOneJob)
            df_JobsTable["FullTime"][Jobcounter] = fulltime
            df_JobsTable["ExperienceLevel"][Jobcounter] = experience
            df_JobsTable["Education"][Jobcounter] = education
            df_JobsTable["Source"][Jobcounter] = "IN"
            df_JobsTable["link"][Jobcounter] = link['href']
            df_JobsTable["JobID"][Jobcounter]  = link['href'].split("jk=")[1].split('&')[0]
            AllWords.append(WOneJob)
        
        if len(link['href'].split('americasjobexchange'))> 1 :
            fulltime = soup.findAll('br')[3].nextSibling
            experience = soup.findAll('br')[6].nextSibling
            education = soup.findAll('br')[7].nextSibling
    #Get Job Description, JOB description is included with multiple Br tags and it is not around any Span tag.
            for br in soup.findAll('br'):
                next_s = br.nextSibling
                if str(type(next_s)) == "<class 'bs4.element.NavigableString'>" :
                    next_s = next_s.lower()
                    tokenizer = RegexpTokenizer(r'\w+')
                    next_s = re.sub('[^A-Za-z0-9]+', ' ', next_s).strip()
                    filterdText=tokenizer.tokenize(str(next_s).strip())
                    if len(filterdText) > 0:
                        #Remove stopwords
                        WOneLine = [word for word in filterdText if word not in default_stopwords]
                        WOneJob.extend(WOneLine)
            #WOneJob = list(set(WOneJob))
            #Jobcounter = Jobcounter + 1
            df_JobsTable["keywords"][Jobcounter] = ','.join(WOneJob)
            df_JobsTable["FullTime"][Jobcounter] = fulltime
            df_JobsTable["ExperienceLevel"][Jobcounter] = experience
            df_JobsTable["Education"][Jobcounter] = education
            df_JobsTable["Source"][Jobcounter] = "AJE"
            df_JobsTable["link"][Jobcounter] = link['href']
            df_JobsTable["JobID"][Jobcounter]  = link['href'].split("AJE-")[1].split('?')[0]
            AllWords.append(WOneJob)
        
        Jobcounter = Jobcounter + 1
        rq.urlcleanup()  

# Implement Word2Vec SkipGram model to get the context words. In here we will not be checking if the word is non english or not. The model will convert each word in ALL of the job descriptsions ( AllWords varible) into a vector. The vector will be the size of the window we specify. This will be training phase. Then we specify a word, e.g skill , and the model will predict words that are closely associated with that word. The CBOW model is different where we specify the context words and it will specify the words associate with it.
# 
model = Word2Vec(size=150, window=10, min_count=2, sg=1, workers=5)
model.build_vocab(AllWords)

#sentences (iterable of iterables, optional) – The sentences iterable can be simply a list of lists of tokens, but for larger corpora, consider an iterable that streams the sentences directly from disk/network. 
#size (int, optional) – Dimensionality of the word vectors.
#window (int, optional) – Maximum distance between the current and predicted word within a sentence.
#min_count (int, optional) – Ignores all words with total frequency lower than this.

model.train(sentences=AllWords, total_examples=len(AllWords), epochs=model.iter)

import pickle
pickle.dump(model, open(ModelLocation, 'wb'))

 
skills = model.wv.most_similar(positive='programming',topn=50)
 
skills_list = [word[0] for word in skills]

skills = model.wv.most_similar(positive='languages',topn=50)

skills_list.extend( [word[0] for word in skills])

#skills_list.remove('programming')

skills_list.remove('languages')
 
df_JobsTable["TotalSkills"] =0
for skill in skills_list:
    df_JobsTable[skill] =df_JobsTable["keywords"].str.contains(','+skill+',')
    df_JobsTable["TotalSkills"] = df_JobsTable.apply(lambda row: row.TotalSkills + row[skill], axis=1)

df_JobsTable["IsaDataScientist"]= df_JobsTable["TotalSkills"] > 1

  
wanted_vocab = dict((k, model.wv.vocab[k]) for k in skills_list)

X = model[wanted_vocab]
 
from sklearn.manifold import TSNE
tsne_model = TSNE(random_state=23)

Y = tsne_model.fit_transform(X)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(Y[:, 0], Y[:, 1])
words = list(wanted_vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
#ax.set_yticklabels([]) #Hide ticks
#ax.set_xticklabels([]) #Hide ticks
_ = plt.show()

df_JobsTable.to_csv(FinalDF)
