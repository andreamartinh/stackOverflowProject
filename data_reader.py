###Import the necessary libraries
import os
import numpy as np
import pandas as pd
from profanity import profanity 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

### Import the dataset
df = pd.read_csv('data_set.csv')

###Cleaning some features

##Comments Features:

#There are two types of posts that can be edited
# I use 0 for Editing a question
# I use 1 for editing an answer
df.loc[df['PostTypeId'] == 1, 'PostTypeId'] = 0
df.loc[df['PostTypeId'] == 2, 'PostTypeId'] = 1

#Checks if the post was edited before
df['LastEditDate']=df['LastEditDate'].fillna(0)
df.loc[df['LastEditDate'] != 0, 'LastEditDate'] = 1

#Comments Length
df['CommentLength'] = df['Comment'].apply(len)

#Check if the title of the post was edited
df["TitleChange1"] = df['Title'].fillna('False')
df.loc[df['TitleChange1'] != 'False', 'TitleChange1'] = 'True'

df.loc[df['Title'] == df['Title.1'], "TitleChange2"] = 'True'
df.loc[df['Title'] != df['Title.1'], "TitleChange2"] = 'False'

df.loc[df['TitleChange1'] == 'False', "TitleChange1"] = 0
df.loc[df['TitleChange1'] == 'True', "TitleChange1"] = 1

df.loc[df['TitleChange2'] == 'False', "TitleChange2"] = 0
df.loc[df['TitleChange2'] == 'True', "TitleChange2"] = 1

df['TitleChange'] = df['TitleChange1']^df['TitleChange2']

# check for profanity in the comments and the editions
df['CommentProfanity'] = df['Comment'].apply(lambda x: profanity.contains_profanity(x))
df['Text']=df['Text'].fillna('0')
df['TextProfanity'] = df['Text'].apply(lambda x: profanity.contains_profanity(x))

##User Features

#The user has a WebstieURL
df['WebsiteUrl']=df['WebsiteUrl'].fillna(0)
df.loc[df['WebsiteUrl'] != 0, 'WebsiteUrl'] = 1

#The user stated a Location
df['WebsiteUrl']=df['WebsiteUrl'].fillna(0)
df.loc[df['Location'] != 0, 'Location'] = 1

#the user wrote an AboutMe
df['AboutMe']=df['AboutMe'].fillna(0)
df.loc[df['AboutMe'] != 0, 'AboutMe'] = 1


## Output
#output 0 notApprove, 1 approve
df['Y'] = df['ApprovalDate'].fillna(0)
df.loc[df['Y'] != 0, 'Y'] = 1


###Organizing the data in a new dataframe
##New DataFrame
data = pd.DataFrame()


#Qestion 0, answer 1
data['PostType']= df['PostTypeId']
# Not Edited before 0, Edited before 1
data['Edited']= df['LastEditDate']
#length of comment
data['LenComment']= df['CommentLength']
#Title Change
data['TitleChange'] = 'Nan'
data.loc[df['TitleChange'] == True, 'TitleChange'] = 1
data.loc[df['TitleChange'] == False, 'TitleChange'] = 0
#CommentProfanity
data['ComProf'] = 'Nan'
data.loc[df['CommentProfanity'] == True, 'ComProf'] = 1
data.loc[df['CommentProfanity'] == False, 'ComProf'] = 0
#TextProfanity
data['TxtProf'] = 'Nan'
data.loc[df['TextProfanity'] == True, 'TxtProf'] = 1
data.loc[df['TextProfanity'] == False, 'TxtProf'] = 0

#Total Reputation
data['Reputation']= df['Reputation']
#totalUpvotes
data['UpVotes']= df['UpVotes']
#totalDownVotes
data['DownVotes']= df['DownVotes']
#Completion of profile 0 nothing 3 all complete
data['ProfileCompletion'] = df['Location'] + df['AboutMe'] + df['WebsiteUrl']


#output
data['Output'] = df['Y']


