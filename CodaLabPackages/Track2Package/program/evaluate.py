import sys, os
from sklearn.metrics import mean_squared_error
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

    assert len(true_csv) == len(pred_csv), "Submission with wrong number of entries: Should be " + str(len(true_csv))
    assert 'SubjectID' in pred_csv.columns and 'X-Grade' in pred_csv.columns, \
        "Submission columns should be: SubjectID, X-Grade"
    assert set(true_csv['SubjectID']) == set(pred_csv['SubjectID']), "Submission SubjectIDs do not match."
    assert set(pred_csv["X-Grade"]) != set(
        [False, True]), "Submission should be a continuous grade prediction, not binary."
    
    df = true_csv.set_index('SubjectID').join(pred_csv.set_index('SubjectID'), rsuffix="Prediction")
    
    mse = mean_squared_error(df["X-Grade"], df["X-GradePrediction"])
    print(mse)

    output_file.write(b"MSE: %f" % mse)
    output_file.close()
