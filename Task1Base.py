import pandas as pd
import numpy as np
from ProgSnap2 import ProgSnap2Dataset
import datetime
import re

from sklearn.model_selection import cross_validate 
from sklearn.neural_network import MLPClassifier

np.random.seed(27601)

# Read tables from files
TRAIN_PATH = "data/CodeWorkout/S19"
data = ProgSnap2Dataset(TRAIN_PATH)

all_main_df = data.get_main_table()
codestate_df = data.get_code_states_table()
# subjProb = pd.read_csv('SplittedData/Train/Train_SubjectProblem.csv')

# remove duplicates
main_df = all_main_df.drop_duplicates(subset='CodeStateID', keep='first') 
main_df.reset_index(inplace = True)

# only keep rows when program is run ("Run.Program")
for i in range(len(main_df)):
    if main_df['EventType'][i] != 'Run.Program':
        main_df.drop(i)
main_df.reset_index(inplace = True, drop=True)

# get maximum time for each student
grouper_time = main_df.groupby(['SubjectID'])
max_times = grouper_time['ServerTimestamp'].max().to_frame(name = "MaxTimeStamp").reset_index()

# choose only the first time student submits problem with Score = 1
score_1 = main_df[main_df['Score'] == 1]
subj_time = {} 

# store first timestamp (i[4]) score == 1 for each student (i[1])
for i in score_1.values:
    if i[1] not in subj_time:
        subj_time[i[1]] = i[3]

# remove all rows with time stamp after score =1 first time
for i in range(len(main_df)):
    if main_df['SubjectID'][i] in subj_time:
        if main_df['ServerTimestamp'][i] > subj_time[main_df['SubjectID'][i]]:
            main_df.drop(i)
            
# remove all subject and problem id duplicate rows only keeping the last occurence
main_df = main_df.drop_duplicates(subset=['SubjectID','ProblemID'], keep='last') 

# sort by order
main_df.sort_values(by=['Order'], inplace = True)

# convert servertime to datetime type
main_df['ServerTimestamp'] = pd.to_datetime(main_df['ServerTimestamp']).astype('int')

# Getting a full list of students
student_list = pd.unique(main_df['SubjectID'])

# Getting a full problem dataframe
problem_df = main_df[['CourseSectionID','AssignmentID','ProblemID']].drop_duplicates()

# Calcuating mean attempts for problems
problem_attempt_mean = main_df.groupby(['CourseSectionID','AssignmentID','ProblemID'])[['Attempt']].mean()
problem_df = problem_df.join(problem_attempt_mean, on=['CourseSectionID','AssignmentID','ProblemID'])

# Calcuating mean scores for problems
problem_score_mean = main_df.groupby(['CourseSectionID','AssignmentID','ProblemID'])[['Score']].mean()
problem_df = problem_df.join(problem_score_mean, on=['CourseSectionID','AssignmentID','ProblemID'])

student_problem_df_list = []
for student in student_list:
    
    # Selecting the submissions by students
    student_problem_df = main_df[main_df['SubjectID'] == student]
    student_problem_df = student_problem_df.sort_values(['Order'], ascending=True) 
    
    student_problem_prev_attempt = [] 
    student_problem_prev_max_attempt = []  
    student_problem_prev_score = []
    student_problem_attempt_mean = [] 
    student_problem_score_mean = []

    for i in range(student_problem_df.shape[0]): 
        if i == 0:
            student_problem_prev_attempt.append(1) 
            student_problem_prev_max_attempt.append(1) 
            student_problem_prev_score.append(0.5) 
        else:
            # Calculate the average previous attempts and scores of the student
            student_problem_prev_attempt.append(np.median(student_problem_df.iloc[:i]['Attempt']))
            student_problem_prev_max_attempt.append(np.max(student_problem_df.iloc[:i]['Attempt']))
            student_problem_prev_score.append(np.mean(student_problem_df.iloc[:i]['Score']))
            
        # Getting problem specific matrics
        current_problem = student_problem_df.iloc[i][['CourseSectionID','AssignmentID','ProblemID']].values

        student_problem_attempt_mean.append(problem_df.query('CourseSectionID == ' + str(current_problem[0])
                                                             + ' and AssignmentID == ' + str(current_problem[1])
                                                             + ' and ProblemID == ' + str(current_problem[2]))['Attempt'].values[0])

              
        
        student_problem_score_mean.append(problem_df.query('CourseSectionID == ' + str(current_problem[0])
                                                             + ' and AssignmentID == ' + str(current_problem[1])
                                                             + ' and ProblemID == ' + str(current_problem[2]))['Score'].values[0])
        
    student_problem_df['PrevMedianAttempt'] = student_problem_prev_attempt
    student_problem_df['PrevMaxAttempt'] = student_problem_prev_max_attempt
    student_problem_df['PrevScore'] = student_problem_prev_score
    student_problem_df['AttemptMean'] = student_problem_attempt_mean
    student_problem_df['ScoreMean'] = student_problem_score_mean
    
    student_problem_df_list.append(student_problem_df)

student_problem_concat_df = pd.concat(student_problem_df_list)



# Cross validation
clf = MLPClassifier()

# X and y for MLP
X = student_problem_concat_df[['PrevMedianAttempt', 'PrevMaxAttempt', 'PrevScore', 'AttemptMean', 'ScoreMean']]
y = student_problem_concat_df['StudentPerformance']

# cv with accuracy and F1 score
cv_results = cross_validate(clf, X, y, groups = main_df['SubjectID'],
                            cv=5, scoring=('accuracy', 'f1'), return_train_score=True)

print('Val Accuracy:',np.mean(cv_results['test_accuracy']), 'Val F1:', np.mean(cv_results['test_f1']))
