import pandas as pd
import os
from os import path


class PS2:
    """ A class holding constants used to get columns of a PS2 dataset
    """

    Order = 'Order'
    SubjectID = 'SubjectID'
    ToolInstances = 'ToolInstances'
    ServerTimestamp = 'ServerTimestamp'
    ServerTimezone = 'ServerTimezone'
    CourseID = 'CourseID'
    CourseSectionID = 'CourseSectionID'
    AssignmentID = 'AssignmentID'
    ProblemID = 'ProblemID'
    Attempt = 'Attempt'
    CodeStateID = 'CodeStateID'
    IsEventOrderingConsistent = 'IsEventOrderingConsistent'
    EventType = 'EventType'
    Score = 'Score'
    CompileResult = 'CompileResult'
    CompileMessageType = 'CompileMessageType'
    CompileMessageData = 'CompileMessageData'
    EventID = 'EventID'
    ParentEventID = 'ParentEventID'
    SourceLocation = 'SourceLocation'
    Code = 'Code'


class ProgSnap2Dataset:

    MAIN_TABLE_FILE = 'MainTable.csv'
    LINK_TABLE_DIR = 'LinkTables'
    CODE_STATES_TABLE_FILE = os.path.join('CodeStates', 'CodeStates.csv')

    def __init__(self, directory):
        self.directory = directory
        self.main_table = None
        self.code_states_table = None

    def path(self, local_path):
        return path.join(self.directory, local_path)

    def get_main_table(self):
        """ Returns a Pandas DataFrame with the main event table for this dataset
        """
        if self.main_table is None:
            self.main_table = pd.read_csv(self.path(ProgSnap2Dataset.MAIN_TABLE_FILE))
            self.main_table.sort_values(by=[PS2.SubjectID, PS2.Order], inplace=True)
        return self.main_table.copy()

    def get_code_states_table(self):
        """ Returns a Pandas DataFrame with the code states table form this dataset
        """
        if self.code_states_table is None:
            self.code_states_table = pd.read_csv(self.path(ProgSnap2Dataset.CODE_STATES_TABLE_FILE))
        return self.code_states_table.copy()

    def __link_table_path(self):
        return self.path(ProgSnap2Dataset.LINK_TABLE_DIR)

    def list_link_tables(self):
        """ Returns a list of the link tables in this dataset, which can be loaded with load_link_table
        """
        path = self.__link_table_path()
        dirs = os.listdir(path)
        return [f for f in dirs if os.isfile(path.join(path, f)) and f.endswith('.csv')]

    def load_link_table(self, link_table):
        """ Returns a Pandas DataFrame with the link table with the given name
        :param link_table: The link table nme or file
        """
        if not link_table.endswith('.csv'):
            link_table += '.csv'
        return pd.read_csv(path.join(self.__link_table_path(), link_table))

    @staticmethod
    def __to_one(lst, error):
        if len(lst) == 0:
            return None
        if len(lst) > 1:
            raise Exception(error or 'Should have only one result!')
        return lst.iloc[0]

    def get_code_for_id(self, code_state_id):
        if code_state_id is None:
            return None
        code_states = self.get_code_states_table()
        code = code_states[code_states[PS2.CodeStateID] == code_state_id][PS2.Code]
        return ProgSnap2Dataset.__to_one(code, 'Multiple code states match that ID.')

    def get_code_for_event_id(self, row_id):
        events = self.get_main_table()
        code_state_ids = events[events[PS2.EventID == row_id]][PS2.CodeStateID]
        code_state_id = ProgSnap2Dataset.__to_one(code_state_ids, 'Multiple rows match that ID.')
        return self.get_code_for_id(code_state_id)

    def get_subject_ids(self):
        events = self.get_main_table()
        return events[PS2.SubjectID].unique()

    def get_problem_ids(self):
        events = self.get_main_table()
        return events[PS2.ProblemID].unique()

    def get_trace(self, subject_id, problem_id):
        events = self.get_main_table()
        rows = events[(events[PS2.SubjectID] == subject_id) & (events[PS2.ProblemID] == problem_id)]
        ids = rows[PS2.CodeStateID].unique()
        return [self.get_code_for_id(code_state_id) for code_state_id in ids]


if __name__ == '__main__':
    data = ProgSnap2Dataset('data/CodeWorkout/S19')
    for code in data.get_trace('4d230b683bf9840553ae57f4acc96e81', 32):
        print(code)
        print('-------')
