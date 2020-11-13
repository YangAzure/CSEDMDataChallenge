import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ProgSnap2 import ProgSnap2Dataset, PS2

np.random.seed(27601)

data = ProgSnap2Dataset('data/CodeWorkout')


all_main_df = data.get_main_table()

# Task 3 needs to also split main table: make them train/test folder
codestate_df = data.get_code_states_table()
subject_df = data.load_link_table('Subject')

# Assigning label for Prediction models

# group by three columns -- mulitple problems from multiple data sources
# course section > ass > prob
# TODO: This is definitely not the correct way to calculate the median number of attempts, since the Attempt column is
# the attempts number, starting at 1 and going up...
median_thresholds = all_main_df.groupby(['CourseSectionID', 'AssignmentID', 'ProblemID'])[['Attempt']].median()
all_main_df = all_main_df.join(median_thresholds, on=['CourseSectionID', 'AssignmentID', 'ProblemID'], rsuffix='Median')
all_main_df['StudentPerformance'] = np.where((all_main_df['Attempt'] > all_main_df['AttemptMedian']), 1, 0)


train_ID, test_ID = train_test_split(data.get_subject_ids(), test_size=0.2)
print("Training set students:", train_ID.size, ", test set students:", test_ID.size)

all_main_df[all_main_df['SubjectID'].isin(train_ID)].to_csv("Splitted_data/Train/TrainMainTable.csv",index=False) # 2 folders be better,
all_main_df[all_main_df['SubjectID'].isin(test_ID)].to_csv("Splitted_data/Test/TestMainTable.csv",index=False)

codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['CodeStateID'])].to_csv("Splitted_data/Train/TrainCodeStates.csv",index=False)
codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['CodeStateID'])].to_csv("Splitted_data/Test/TestCodeStates.csv",index=False)

subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['SubjectID'])].to_csv("Splitted_data/Train/TrainSubject.csv", index = False)
subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['SubjectID'])].to_csv("Splitted_data/Test/TestSubject.csv", index = False)



# SubjectProblem.csv

# SubjectProblem.csv
subject_problem_df = all_main_df[['SubjectID', 'ProblemID']]
median_problemID = all_main_df.groupby(['ProblemID'])[['Attempt']].median() 
median_problemID.rename(columns = {'Attempt': 'Median_Attempt'}, inplace = True)
subject_problem_df = subject_problem_df.join(median_problemID, on=['ProblemID'], rsuffix = '_median')
subject_problem_df['StudentLabels'] = [0 if all_main_df['Attempt'][i] > subject_problem_df['Median_Attempt'][i] else 1 for i in range(len(subject_problem_df))]
subject_problem_df.to_csv("SubjectProblem.csv",index=False)
