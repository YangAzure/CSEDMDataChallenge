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
    
    true_csv = pd.read_csv(os.path.join(truth_dir, "truth.csv"), names=["subject", "truth"], header=None)
    pred_csv = pd.read_csv(os.path.join(submit_dir, "predictions.csv"), names=["subject", "pred"], header=None)
    
    df = true_csv.set_index('subject').join(pred_csv.set_index('subject'))
    
    mse = mean_squared_error(df["pred"], df["truth"])
    print(mse)

    output_file.write(b"MSE: %f" % mse)
    output_file.close()
