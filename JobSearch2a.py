import numpy as np
import pandas as pd
import sys

JobDesc = open("C://Users//anupama//Desktop//JobDesc.txt").read()
JobDesc = JobDesc.lower()
tokenizer = RegexpTokenizer(r'\w+')
JobDesc = re.sub('[^a-z0-9]+', ' ', JobDesc).strip()
JobDesc=tokenizer.tokenize(str(JobDesc).strip())
if len(JobDesc) > 0:
    #Remove stopwords
    WOneLine = [word for word in JobDesc if word not in default_stopwords]
    WOneJob.extend(WOneLine)


NoOfSkills = 0
SkilsList = df_JobsTable.columns 
for skill in SkilsList :
    if WOneJob.contains(skill):
        df_JobsTable[skill] = 1
        NoOfSkills = NoOfSkills + 1

df_JobsTable["TotalSkills"] = NoOfSkills
