


#Data Collection

!pip install extract-wc-data

from ExtractWCData.get_latest_data import GetData

data = GetData()
df = data.get_data()

df.tail()

df.to_csv('latest_data.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load all the CSV datasets

world_cup = pd.read_csv("World_cup_2023.csv")
world_cup.head()

results = pd.read_csv("results.csv")
results.head()

latest = pd.read_csv('latest_data.csv')
latest.head()

print(f'World_cup data shape : {world_cup.shape}')
print(f'results data shape : {results.shape}')
print(f'Latest world_cup data shape : {latest.shape}')

results = pd.concat([results,latest],axis=0)

results = results.reset_index(drop = True)
results.tail(5)

results.shape

results.columns

results.drop(columns=['Unnamed: 0'],axis=1,inplace=True)

results.head()

results.columns

results.drop(columns=['Date','Margin','Ground'],axis=1,inplace=True)

results.head()

world_cup_teams = ['India','South Africa','Australia','New Zealand','Pakistan','Afghanistan','England','Bangladesh','Sri Lanka','Netherlands']

df_teams_1 = results[results['Team_1'].isin(world_cup_teams)]
df_teams_2 = results[results['Team_2'].isin(world_cup_teams)]
df_winners = results[results['Winner'].isin(world_cup_teams)]

df_team = pd.concat((df_teams_1,df_teams_2,df_winners),axis = 0)
df_team.head()

df_team.loc[:,'Winning'] = np.where(df_team['Winner']==df_team['Team_1'],1,2)
df_team.head()

df_team.drop(columns=['Winner'],axis=1,inplace=True)

df_team.head()

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#Apply the encoding
df_team = pd.get_dummies(df_team,prefix=['Team_1','Team_2'], columns = ['Team_1','Team_2'],dtype = int , sparse=False)
df_team.head()

x = df_team.drop(columns = ['Winning'],axis = 1)
y = df_team['Winning']

#Splitting the data in the training and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=34)

x_train

y_train

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

classifiers = {
    'Random Forest' : RandomForestClassifier(),
    'Logistic Regression' : LogisticRegression(),
    'Decision Tree' : DecisionTreeClassifier(),
}

for name,clf in classifiers.items():
  pipeline = Pipeline([('classifier',clf)])

  pipeline.fit(x_train,y_train)

  #Make Predictions
  y_pred = pipeline.predict(x_test)

  #Calculate the accuracy
  acc = accuracy_score(y_test,y_pred)

  print(f'{name}:')
  print(f"Accuracy : {acc:.4f}")

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

predictions = rf.predict(x_test)

label_to_team = {1: 'Team_1',2: 'Team_2'}
Winner = [label_to_team[label] for label in predictions]
print(Winner)

rankings = pd.read_csv('Icc_ranking.csv')
rankings.head()

fixtures = pd.read_csv('Fixtures.csv')
fixtures.head()

pred_set = []

fixtures.insert(1,'first_position',fixtures['Team_1'].map(rankings.set_index('Team_name')['Team_ranking']))
fixtures.insert(2,'second_position',fixtures['Team_2'].map(rankings.set_index('Team_name')['Team_ranking']))

fixtures = fixtures.iloc[:80, :]
fixtures.head()

for index, row in fixtures.iterrows():
  if row['first_position'] < row['second_position']:
    pred_set.append({'Team_1' : row['Team_1'] , 'Team_2': row['Team_2'] , 'Winning_team' : None})
  else:
    pred_set.append({'Team_1' : row['Team_2'] , 'Team_2': row['Team_1'] , 'Winning_team' : None})


pred_set = pd.DataFrame(pred_set)

pred_set.head()

backup_pred_set = pred_set

ped_set = pd.get_dummies(pred_set,prefix = ['Team_1', 'Team_2'] , columns = ['Team_1', 'Team_2'], dtype = int)

missing_cols = set(df_team.columns) - set(pred_set.columns)

for cols in missing_cols :
  pred_set[cols] = 0

pred_set = pred_set[df_team.columns]

pred_set = pred_set.drop(['Winning'],axis=1)
pred_set.head()

prdictions = rf.predict(pred_set)
for i in range(fixtures.shape[0]):
  print(backup_pred_set.iloc[i,1] + " Vs " + backup_pred_set.iloc[i,0])
  if predictions[i] == 1 :
    print('Winner :' + backup_pred_set.iloc[i,1])
  else:
    print('Winner :' + backup_pred_set.iloc[i,0])
  print("")

latest.head()

latest.drop(columns = ['Unnamed: 0'],axis=1,inplace=True)

latest.head()

top_winners = latest['Winner'].value_counts().head(4).index.tolist()
print(f"Top 4 teams : {top_winners}")

#Predict the single match results of future

def predict_single_match(model,rankings,team_1,team_2):
  single_match_data = pd.DataFrame({
      'Team_1': [team_1],
      'Team_2': [team_2],
  })

  #Insert the team ranking data
  single_match_data.insert(1,'first_position',single_match_data['Team_1'].map(rankings.set_index("Team_name")['Team_ranking']))
  single_match_data.insert(2,'second_position',single_match_data['Team_2'].map(rankings.set_index("Team_name")['Team_ranking']))

  #Apply one hot encoding
  single_match_data = pd.get_dummies(single_match_data,prefix = ['Team_1', 'Team_2'] , columns = ['Team_1', 'Team_2'], dtype = int)

  #Find the missing columns
  missing_cols = set(df_team.columns) - set(single_match_data.columns)

  #Set the missing columns to 0 and then keep only the columns present

  for col in missing_cols:
    single_match_data[col] = 0

  single_match_data = single_match_data[df_team.columns]

  #Drop the winning column
  single_match_data = single_match_data.drop(['Winning'],axis = 1)

  #Making the predictions
  prediction = model.predict(single_match_data)

  #print the result
  print(f"{team_1} vs {team_2}")

  if prediction[0] == 1:
    print(f"Winner: {team_1}")
  else:
    print(f"Winner: {team_2}")

    print((""))

#predicting the semifinal results:
predict_single_match(rf,rankings,"India","New Zealand")

#predicting the final results:
predict_single_match(rf,rankings,"India","Australia")

"""INDIA WILL WIN ICC WORLD CUP 2023 !!!!"""