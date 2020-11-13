import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(27601)

# Specify data source
MainTableLocation = "MainTable.csv"
CodeStateTableLocation = "LinkTables/CodeStates.csv"
SubjectTableLocation = "LinkTables/Subject.csv"

# Reading original data
all_main_df = pd.read_csv(MainTableLocation) 
codestate_df = pd.read_csv(CodeStateTableLocation)
subject_df = pd.read_csv(SubjectTableLocation)

# Train test split and saving
train_ID,test_ID = train_test_split(np.unique(all_main_df['SubjectID']),test_size=0.2)
print("Training set students:",train_ID.size,", test set students:", test_ID.size)

all_main_df[all_main_df['SubjectID'].isin(train_ID)].to_csv("Splitted_data/Train/TrainMainTable.csv",index=False) 
all_main_df[all_main_df['SubjectID'].isin(test_ID)].to_csv("Splitted_data/Test/TestMainTable.csv",index=False)

codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['CodeStateID'])].to_csv("Splitted_data/Train/TrainCodeStates.csv",index=False)
codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['CodeStateID'])].to_csv("Splitted_data/Test/TestCodeStates.csv",index=False)

subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['SubjectID'])].to_csv("Splitted_data/Train/TrainSubject.csv", index = False)
subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['SubjectID'])].to_csv("Splitted_data/Test/TestSubject.csv", index = False)

# Generating label for Task 1
subject_problem_df = all_main_df[['SubjectID', 'ProblemID']]
median_problemID = all_main_df.groupby(['ProblemID'])[['Attempt']].median() 
median_problemID.rename(columns = {'Attempt': 'Median_Attempt'}, inplace = True)
subject_problem_df = subject_problem_df.join(median_problemID, on=['ProblemID'], rsuffix = '_median')
subject_problem_df['StudentLabels'] = [0 if all_main_df['Attempt'][i] > subject_problem_df['Median_Attempt'][i] else 1 for i in range(len(subject_problem_df))]
subject_problem_df.to_csv("SubjectProblem.csv",index=False)
