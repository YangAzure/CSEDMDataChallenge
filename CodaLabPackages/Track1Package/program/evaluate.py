import sys, os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res') 
truth_dir = os.path.join(input_dir, 'ref')


if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')              
    output_file = open(output_filename, 'wb')
    
    true_csv = pd.read_csv(os.path.join(truth_dir, "truth.csv"))
    pred_csv = pd.read_csv(os.path.join(submit_dir, "predictions.csv"))
    
    assert len(true_csv) == len(pred_csv), "Submission with wrong number of entries: Should be " + str(len(true_csv))
    assert 'SubjectID' in pred_csv.columns and 'ProblemID' in pred_csv.columns and 'Label' in pred_csv.columns, \
        "Submission columns should be: SubjectID, ProblemID, Label"
    assert set(true_csv['SubjectID']) == set(pred_csv['SubjectID']), "Submission SubjectIDs do not match."
    assert set(true_csv['ProblemID']) == set(pred_csv['ProblemID']), "Submission ProblemIDs do not match."
    assert set(pred_csv["Label"]) != set([False,True]), \
        "Submission should include probabilities rather than binary results."
    
    df = true_csv.set_index(['SubjectID','ProblemID']).join(pred_csv.set_index(['SubjectID','ProblemID']), rsuffix="ScorePrediction")
    df["LabelPrediction"] = df["LabelScorePrediction"] > 0.5
    
    f1_negative = f1_score(1-df["Label"], 1-df["LabelPrediction"])
    f1_positive = f1_score(df["Label"], df["LabelPrediction"])
    f1 = (f1_negative + f1_positive)/2
    acc = accuracy_score(df["Label"], df["LabelPrediction"])
    auc = roc_auc_score(df["Label"], df["LabelScorePrediction"])

   
    print("MACRO F1: %f" % f1)
    print("POSITIVE F1: %f" % f1_positive)
    print("ACC: %f" % acc)
    print("AUC: %f" % auc)

    output_file.write(b"MACRO_F1: %f \n" % f1)
    output_file.write(b"POSITIVE_F1: %f \n" % f1_positive)
    output_file.write(b"ACC: %f \n" % acc)
    output_file.write(b"AUC: %f \n" % auc)
    output_file.close()
