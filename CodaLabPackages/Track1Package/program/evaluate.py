import sys, os
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd



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
    
    df = true_csv.set_index(['SubjectID','ProblemID']).join(pred_csv.set_index(['SubjectID','ProblemID']), rsuffix="Prediction")
    
    f1_negative = f1_score(1-df["Label"], 1-df["LabelPrediction"])
    f1_positive = f1_score(df["Label"], df["LabelPrediction"])
    f1 = (f1_negative + f1_positive)/2
    acc = accuracy_score(df["Label"], df["LabelPrediction"])

   
    print("MACRO F1: %f" % f1)
    print("ACC: %f" % acc)

    output_file.write(b"MACRO F1: %f" % f1)
    output_file.close()
