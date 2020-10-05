import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(27601)

MainTableLocation = "MainTable.csv"
CodeStateTableLocation = "LinkTables/CodeStates.csv"

SubjectTableLocation = "LinkTables/Subject.csv"

all_main_df = pd.read_csv(MainTableLocation) # Should be subjects.csv: Confirm?

# Task 3 needs to also split main table: make them train/test folder
codestate_df = pd.read_csv(CodeStateTableLocation)
subject_df = pd.read_csv(SubjectTableLocation)

# Assigning label for Prediction models

## group by three columns -- mulitple problems from multiple data sources
## course section > ass > prob
median_thresholds = all_main_df.groupby(['CourseSectionID','AssignmentID','ProblemID'])[['Attempt']].median() 
all_main_df = all_main_df.join(median_thresholds, on=['CourseSectionID','AssignmentID','ProblemID'], rsuffix='Median') 
all_main_df['StudentPerformance'] = np.where((all_main_df['Attempt'] > all_main_df['AttemptMedian']), 1, 0)



train_ID,test_ID = train_test_split(np.unique(all_main_df['SubjectID']),test_size=0.2)
print("Training set students:",train_ID.size,", test set students:", test_ID.size)

all_main_df[all_main_df['SubjectID'].isin(train_ID)].to_csv("Splitted_data/Train/TrainMainTable.csv",index=False) # 2 folders be better,
all_main_df[all_main_df['SubjectID'].isin(test_ID)].to_csv("Splitted_data/Test/TestMainTable.csv",index=False)

codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['CodeStateID'])].to_csv("Splitted_data/Train/TrainCodeStates.csv",index=False)
codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['CodeStateID'])].to_csv("Splitted_data/Test/TestCodeStates.csv",index=False)

subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['SubjectID'])].to_csv("Splitted_data/Train/TrainSubject.csv", index = False)
subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['SubjectID'])].to_csv("Splitted_data/Test/TestSubject.csv", index = False)



# SubjectProblem.csv

subject_problem_df = all_main_df[['SubjectID', 'ProblemID']]
median_problemID = all_main_df.groupby(['ProblemID'])[['Attempt']].median() 
subject_problem_df = subject_problem_df.join(median_problemID, on=['ProblemID'], rsuffix = '_median')
subject_problem_df.to_csv("Splitted_data/SubjectProblem.csv",index=False)
