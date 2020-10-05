import pandas as pd
import numpy as np
# pd.set_option('display.max_rows', 500)
np.random.seed(27601)

MAIN_TABLE_LOCATION = "Splitted_data/Train/TrainMainTable.csv"
CODE_STATE_TABLE_LOCATION = "Splitted_data/Train/TrainCodeStates.csv"
SUBJECT_TABLE_LOCATION = "Splitted_data/Train/TrainSubject.csv"
EARLY_PROBLEM_NUM = 10 # Using a date (Check the cutoff days, see the clusters of the dates or the weekdays) (How many weeks of data would it take to do the early prediction)


all_main_df = pd.read_csv(MAIN_TABLE_LOCATION)
codestate_df = pd.read_csv(CODE_STATE_TABLE_LOCATION)
subject_df = pd.read_csv(SUBJECT_TABLE_LOCATION)

# Only keeping one row for each submmission
main_df = all_main_df[all_main_df['EventType'] == "Run.Program"]
main_df = main_df.sort_values(by=['ServerTimestamp','Order'])

subjects = list(np.unique(main_df['SubjectID']))




main_df['ServerTimestamp'] = pd.to_datetime(main_df['ServerTimestamp'])

main_df = main_df[['CourseSectionID', 'AssignmentID', 'ProblemID', 'SubjectID', 'Score', 'Attempt']]
early_problems = main_df[['AssignmentID','ProblemID']].drop_duplicates().head(n = EARLY_PROBLEM_NUM).values

# Geting all early problems of the students
all_student_early_problem = []
for subject in subjects:
    student_grade = subject_df[subject_df['SubjectID'] == subject]['X-Grade']
    if student_grade.shape[0] == 0:
        continue
    student_all_records = main_df[main_df['SubjectID'] == subject]
    student_participated_problems = student_all_records[['CourseSectionID','AssignmentID','ProblemID']].drop_duplicates()
    
    # Getting early problems of the student
    #early_problems = student_participated_problems.head(n = EARLY_PROBLEM_NUM).values
    student_predicting = []
    for problem in early_problems:
        student_problem = student_all_records.query('AssignmentID == ' + str(problem[0]) + 
                         ' and ProblemID == ' + str(problem[1]))
        
        if len(student_problem) == 0:

            student_problem = student_problem.append({'CourseSectionID': 200,
                                    'AssignmentID': problem[0],
                                    'ProblemID': problem[1],
                                    'SubjectID': subject,
                                    'Score': 0.0,
                                    'Attempt': 150}, ignore_index=True)
        for i in range(student_problem.shape[0]):
            if student_problem.iloc[i]['Score'] == 1:
                attempted_student_problem = student_problem.iloc[:i+1]
                student_predicting.append(attempted_student_problem)
                break
            if i == student_problem.shape[0]-1:
                student_predicting.append(student_problem)
        
        # Getting the rank of the student in the problem
        
    student_predicting_df = pd.concat(student_predicting)
    all_student_early_problem.append(student_predicting_df)

all_student_early_problem_df = pd.concat(all_student_early_problem)
    
    

    


# Getting the rank of the students in every problem
ranked_student_early_problem = []
for problem in early_problems:
    student_problem = all_student_early_problem_df.query('AssignmentID == ' + str(problem[0]) + 
                     ' and ProblemID == ' + str(problem[1]))
    ranked_first_submissions = student_problem.drop_duplicates(subset=['SubjectID', 'AssignmentID','ProblemID'], keep='first').sort_values(['Score'])    
    ranked_first_submissions['FirstScoreRank'] = list(range(len(ranked_first_submissions)))

    first_submission_score_rank = []
    for i in range(student_problem.shape[0]):
        checking_subject = student_problem.iloc[i]['SubjectID']
        first_submission_score_rank.append(ranked_first_submissions[ranked_first_submissions['SubjectID'] == checking_subject]['FirstScoreRank'].values[0])
    student_problem['FirstScoreRank'] = first_submission_score_rank
    final_submissions = student_problem.drop_duplicates(subset=['SubjectID', 'AssignmentID','ProblemID'], keep='last')
    
    finally_finished = []
    for i in range(final_submissions.shape[0]):
        if final_submissions.iloc[i]['Score'] == 0:
            finally_finished.append(0)
        else:
            finally_finished.append(1)
    final_submissions["Finished"] = finally_finished

    # Last priority is 'FirstScoreRank', then 'Score', then 'Attempt'
    student_problem_on_final = final_submissions.sort_values(['Finished','Score','Attempt','FirstScoreRank'], ascending=[False, False, True, True])

    student_problem_on_final['ProblemRank'] = list(range(len(student_problem_on_final)))
    
    
    student_problem_rank = []
    for i in range(student_problem.shape[0]):
        checking_subject = student_problem.iloc[i]['SubjectID']
        student_problem_rank.append(student_problem_on_final[student_problem_on_final['SubjectID'] == checking_subject]['ProblemRank'].values[0])
    
    student_problem['ProblemRank'] = student_problem_rank
    
    ranked_student_early_problem.append(student_problem)


    
ranked_student_early_problem_df = pd.concat(ranked_student_early_problem)


X = []
y = []
for subject in subjects:

    student_grade = subject_df[subject_df['SubjectID'] == subject]['X-Grade']
    student_predicting_df = ranked_student_early_problem_df[ranked_student_early_problem_df['SubjectID'] == subject]
    
    # Feature extraction: Student average attempts, student average first scores, student median attempts and scores
    
    student_avg_first_score = np.mean(student_predicting_df.drop_duplicates(subset=['AssignmentID','ProblemID'], keep='first')['Score'])
    student_med_first_score = np.median(student_predicting_df.drop_duplicates(subset=['AssignmentID','ProblemID'], keep='first')['Score'])
    student_avg_attempt = np.mean(student_predicting_df.drop_duplicates(subset=['AssignmentID','ProblemID'], keep='last')['Attempt'])
    student_med_attempt = np.median(student_predicting_df.drop_duplicates(subset=['AssignmentID','ProblemID'], keep='last')['Attempt'])
    student_avg_rank = np.mean(student_predicting_df.drop_duplicates(subset=['AssignmentID','ProblemID'], keep='last')['ProblemRank'])
    student_med_rank = np.median(student_predicting_df.drop_duplicates(subset=['AssignmentID','ProblemID'], keep='last')['ProblemRank'])
                                 
    # Generating training sample
    if student_grade.tolist():
        y.append(student_grade.tolist()[0])
        X.append([student_avg_first_score, student_med_first_score, student_avg_attempt, student_med_attempt, student_avg_rank, student_med_rank])
    # Rank the students on the problems and use the rank (avg score rank)
    # Standardization attempts and score
    
    # Ask how did the grade consist of 
    

from sklearn.model_selection import cross_validate # Cross validation on students
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
# reg = MLPRegressor()
reg = LinearRegression()
cv_results = cross_validate(reg, 
                            X, 
                            y, 
                            cv=5, scoring='neg_root_mean_squared_error')

print(-np.mean(cv_results['test_score']))
