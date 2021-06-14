
import pandas as pd
import numpy as np
from ProgSnap2 import ProgSnap2Dataset
from ProgSnap2 import PS2

PATH = "data/CodeWorkout/S19"
data = ProgSnap2Dataset(PATH)

main_table = data.get_main_table()
student_table = data.load_link_table('Subject')

# Get the SubjectIDs where the final grade is non-0
# A 0 grade indicates the student did not take the final
subject_ids = set(student_table[student_table['X-Grade'] != 0][PS2.SubjectID].unique())
subject_ids = subject_ids.intersection(set(student_table['SubjectID'].unique()))


student_table.groupby(by=['SubjectID'])[PS2.ProblemID].unique()

print(subject_ids)
