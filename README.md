## Job Search Results

This project has 2 goals
  * Identify the most common list of skills associated to the job tile. 
  * Refine the search results of a job title. e.g 'Data Scientist' job search does not resturn results such as "Optical Research   Scientist"

![alt text](https://raw.githubusercontent.com/anunav83/JobSkillSearch--NLP/tree/master/Images/JobSearchResults.png)

### Problem Summary:

* Goal 1: Get the list of technical skills associated to a job title.
Using web scrapping package BeautifulSoup, we collect all job listings associated with the job title from major jobs website.

![Image](https://github.com/anunav83/JobSkillSearch--NLP/tree/master/Images/)


We shall then tokenize the words in the job listings and using NLTK packages remove any stop words, punctuations etc(pre-processsing). Using Word2Vec model (skip gram algorithm) identify the technical skills required for that job title. Below is the TNSE graph of the words obtained. 

![Image](https://github.com/anunav83/JobSkillSearch--NLP/tree/master/Images/)

The Word2Vec model uses close to 5000 words from 218 job descriptions. 


* Goal 2: Refine search results based on the skills.
Scan through each Job listing in the dataset to find if it has at least two of the skills listed below. If the job does not have it, it is flagged as erroneous. The resulting dataset will be used as the input to train our model, where target is the flag which indicates if job listing is erroneous or not.

![Image](https://github.com/anunav83/JobSkillSearch--NLP/tree/master/Images/)


![Image](https://github.com/anunav83/JobSkillSearch--NLP/tree/master/Images/)


* Goal 3: Based on the Job description text, identify appropriate Job Title.
Test the model by providing any job description. The model should predict if the job description is a valid for the Data Scientist Job tile.  IsDataScientist would be used as the response variable we would need to predict given a job description. We use Naïve bayes and Logistic Regression algorithms to predict if the Job title is Data Scientist or not. 

### Enhancements:
Below are some of the other ways this data set could be used

* Determine how long it takes for Job to be filled.(Regression).“Posting_date” feature could be used determine this.
  
* How many Jobs we can expect to see on the market, based on company/ location etc (Regression)

* Rate the Job Descriptions: Clear Vs unclear. E.g based on following the description can be tagged as unclear (Classification)
  * Too many words
  * Too few technical skills compared to other listings 
  * Jobs that have not been filled for a long time 
  
* Currently used only 215 Job listing, should train the model with more data.

* Use LDA to determine the Job Title

### References

* Word2Vec : https://www.kaggle.com/liananapalkova/simply-about-word2vec
* Joblisting Website( has jobs from 3 different sources, careerbuilder, indeed etc) : https://www.careeronestop.org/

