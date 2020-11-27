import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os import path
from ProgSnap2 import ProgSnap2Dataset, PS2


np.random.seed(27601)
TARGET_TRAIN_PATH = "SplittedData/Train"
TARGET_TEST_PATH = "SplittedData/Test"

data = ProgSnap2Dataset('.')
train_data = ProgSnap2Dataset(TARGET_TRAIN_PATH)
test_data = ProgSnap2Dataset(TARGET_TEST_PATH)

all_main_df = data.get_main_table()
codestate_df = data.get_code_states_table()
subject_df = data.load_link_table('Subject')


# Splitting
train_ID, test_ID = train_test_split(data.get_subject_ids(), test_size=0.2)
print("Training set students:", train_ID.size, ", test set students:", test_ID.size)

# Getting tables
all_main_df[all_main_df['SubjectID'].isin(train_ID)].to_csv(train_data.path(ProgSnap2Dataset.MAIN_TABLE_FILE),index=False) 
all_main_df[all_main_df['SubjectID'].isin(test_ID)].to_csv(test_data.path(ProgSnap2Dataset.MAIN_TABLE_FILE),index=False)

codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['CodeStateID'])].to_csv(train_data.path(ProgSnap2Dataset.CODE_STATES_TABLE_FILE),index=False)
codestate_df[codestate_df['CodeStateID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['CodeStateID'])].to_csv(test_data.path(ProgSnap2Dataset.CODE_STATES_TABLE_FILE),index=False)

subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(train_ID)]['SubjectID'])].to_csv(train_data.path(path.join(ProgSnap2Dataset.LINK_TABLE_DIR, 'Subject.csv')), index = False)
subject_df[subject_df['SubjectID'].isin(all_main_df[all_main_df['SubjectID'].isin(test_ID)]['SubjectID'])].to_csv(test_data.path(path.join(ProgSnap2Dataset.LINK_TABLE_DIR, 'Subject.csv')), index = False)

# Generating label for Task 1

# dataframe of submitted score
submitted_df = all_main_df[all_main_df['Score'] == 1]

# dataframe of unsubmitted score (get all student and problem ids that aren't from submitted)
sub_student_prob = set([tuple(i) for i in submitted_df[['SubjectID', 'ProblemID']].values])
unsub_student_prob = []
for i in range(len(all_main_df)):
    x = all_main_df['SubjectID'][i]
    y = all_main_df['ProblemID'][i]
    if (x,y) not in sub_student_prob:
        unsub_student_prob.append(all_main_df.iloc[i].values)
unsubmitted_df = pd.DataFrame(unsub_student_prob,
                             columns = list(all_main_df.columns))

# order unsubmitted rows and get last row
group_unsub = unsubmitted_df.groupby(['SubjectID', 'ProblemID'])
group_unsub_df = group_unsub.apply(
    lambda x: x.sort_values(['Order'], ascending=True).tail(1))

# order submitted rows and get first attempt
group_sub_df = submitted_df.groupby(['SubjectID', 'ProblemID']).apply(
    lambda x: x.sort_values(['Order'], ascending=True).head(1))

# concat group_sub group_unsub and shuffle
group_df = pd.concat([group_unsub_df, group_sub_df],ignore_index=True)

# reset index
group_df.reset_index(drop=True, inplace = True)

# shuffle dataframe
group_df = group_df.sample(frac=1)

# TODO: This is definitely not the correct way to calculate the median number of attempts, since the Attempt column is
# the attempts number, starting at 1 and going up...

# calculate median attempts from group_df
median_problemID = all_main_df.groupby(['ProblemID'])[['Attempt']].median() 
median_problemID.rename(columns = {'Attempt': 'Median_Attempt'}, inplace = True)

# get labels based on median 
subject_problem_df = all_main_df[['SubjectID', 'ProblemID']]
subject_problem_df = subject_problem_df.join(median_problemID, on=['ProblemID'], rsuffix = '_median')
subject_problem_df['StudentLabels'] = [0 if all_main_df['Attempt'][i] > subject_problem_df['Median_Attempt'][i] else 1 for i in range(len(subject_problem_df))]

# storing in train and test
subject_problem_df[subject_problem_df['SubjectID'].isin(train_ID)].to_csv(
    train_data.path(path.join(ProgSnap2Dataset.LINK_TABLE_DIR, 'SubjectProblem.csv')), index = False) 
subject_problem_df[subject_problem_df['SubjectID'].isin(test_ID)].to_csv(
    test_data.path(path.join(ProgSnap2Dataset.LINK_TABLE_DIR, 'SubjectProblem.csv')), index = False)
