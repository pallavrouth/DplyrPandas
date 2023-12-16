import pandas as pd
import numpy as np
import inspect
import re

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

@pd.api.extensions.register_dataframe_accessor("ddf")
class DplyrDataFrame:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        frame = inspect.currentframe().f_back
        self.local_vars = frame.f_locals
        self.global_vars = frame.f_globals
        self.groups = None
        self.sequence = []
        self.info = []

    def select(self, cols):
        self.sequence.append("select")
        input_shape = self._obj.shape
        if not isinstance(cols, list): cols = [cols]
        all_columns = self._obj.columns.tolist()
        for i, col in enumerate(cols):
            if ":" in col:   
                col_range = col.split(":")
                start_idx = all_columns.index(col_range[0].strip())
                end_idx = all_columns.index(col_range[1].strip())
                cols[i] = all_columns[start_idx:end_idx + 1]
        cols_flat = flatten_list(cols)
        cols_string = ", ".join(cols_flat)
        self._obj = self._obj.loc[:, cols_flat]
        self.info.append(f"Columns {cols_string} were selected out from {input_shape[1]} columns. There are now {self._obj.shape[1]} columns in the data. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def drop(self, cols):
        self.sequence.append("drop")
        input_shape = self._obj.shape
        if not isinstance(cols, list): cols = [cols]
        all_columns = self._obj.columns.tolist()
        for i, col in enumerate(cols):
            if ":" in col:   
                col_range = col.split(":")
                sIdx = all_columns.index(col_range[0].strip())
                eIdx = all_columns.index(col_range[1].strip())
                cols[i] = all_columns[sIdx:eIdx + 1]
        cols_flat = flatten_list(cols)
        cols_string = ", ".join(cols_flat)
        self._obj = self._obj.drop(cols_flat, axis = 1)
        self.info.append(f"Columns {cols_string} were dropped out of {input_shape[1]} columns. There are now {self._obj.shape[1]} columns in the data. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def mutate(self, exprs):
        self.sequence.append("mutate")
        input_shape = self._obj.shape
        if len(self.sequence) > 1 and self.sequence[-2] == "groupby":
            for key, value in exprs.items():
                col = re.search(r'(?<=d\.)\w+', value).group().strip()
                expr_x = re.sub(r'd\.(\w+)', "x", value).strip()
                lambda_expr = f"lambda x: {expr_x}"
                self._obj = self._obj.assign(_new_col = self._obj.groupby(self.groups)[col].transform(eval(lambda_expr, self.global_vars, self.local_vars))).rename(columns = {"_new_col" : key.strip()})
        else:
            for key, value in exprs.items():
                self._obj = self._obj.assign(_new_col = eval(f"lambda d: {value.strip()}", self.global_vars, self.local_vars)).rename(columns = {"_new_col" : key.strip()})  
        new_col_string = ", ".join([key for key, value in exprs.items()])
        self.info.append(f"Column(s) {new_col_string} were added to the data. There are now {self._obj.shape[1]} columns in the data. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def mutate_ifelse(self, exprs):
        self.sequence.append("mutate")
        input_shape = self._obj.shape
        for key, value in exprs.items():
            input_condition, trueVal, falseVal = [item.strip() for item in value.split(",")]
            if '&' in value:
                subconditions = [item.strip() for item in input_condition.split('&')]
                condition_statement = ' & '.join(f'({subcondition})' for subcondition in subconditions)
            else:
                condition_statement = input_condition.strip()
            self._obj = self._obj.assign(_new_col = lambda d: np.where(eval(condition_statement),int(trueVal),int(falseVal))).rename(columns = {"_new_col" : key.strip()})
        new_col_string = ", ".join([key for key, value in exprs.items()])
        self.info.append(f"Column(s) {new_col_string} were added to the data. There are now {self._obj.shape[1]} columns in the data. Previous dimension: {tuple(input_shape)}.")
        return self
    
    # groupby compatible
    def mutate_across(self, cols, function, prefix):
        self.sequence.append("mutate")
        input_shape = self._obj.shape
        mutated_cols = self._obj[cols].apply(function)
        mutated_dict = {f"{prefix}{col}": mutated_cols[col] for col in mutated_cols}
        self._obj = pd.concat([self._obj, pd.DataFrame(mutated_dict)], axis = 1)
        new_col_string = ", ".join([f"{prefix}{col}" for col in cols])
        self.info.append(f"Column(s) {new_col_string} were added to the data. There are now {self._obj.shape[1]} columns in the data. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def filter(self, exprs):
        self.sequence.append("filter")
        input_shape = self._obj.shape
        if '&' in exprs:
            subconditions = exprs.split('&')
            subcondition_edits = []
            for subcondition in subconditions:
                if '.contains(' in subcondition:
                    new_subcondition = subcondition.replace('.contains','.str.contains')
                    subcondition_edits.append(new_subcondition)
                elif '.in(' in subcondition:
                    new_subcondition = subcondition.replace('.in','.isin')
                    subcondition_edits.append(new_subcondition)
                else:
                    subcondition_edits.append(subcondition)
            condition_statement = ' & '.join(f'({condition})' for condition in subcondition_edits)
        else:
            if '.contains(' in exprs:
                condition_statement = exprs.replace('.contains','.str.contains')
            elif '.in(' in exprs:
                condition_statement = exprs.replace('.in','.isin')
            else:
                condition_statement = exprs
        self._obj = self._obj.loc[eval(f"lambda d: {condition_statement}", self.global_vars, self.local_vars)]
        rows_removed = list(input_shape)[0] - self._obj.shape[0] 
        self.info.append(f"{rows_removed} rows were removed from the data. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def arrange(self, cols, ascending = True):
        self.sequence.append("arrange")
        if ascending:
            self._obj = self._obj.sort_values(cols)
            arranged_col_string = ", ".join(cols)
            self.info.append(f"Rows were arranged by {arranged_col_string}. They were arrranged in ascending order.")
        else:
            self._obj = self._obj.sort_values(cols, ascending = False)
            arranged_col_string = ", ".join(cols)
            self.info.append(f"Rows were arranged by {arranged_col_string}. They were arrranged in decending order.")
        return self
    
    def groupby(self, cols):
        self.sequence.append("groupby")
        input_shape = self._obj.shape
        self.groups = cols
        group_col_string = ", ".join(cols)
        self.info.append(f"Data will be grouped by {group_col_string}.")
        return self
    
    def summarize(self, exprs):
        self.sequence.append("summarize")
        input_shape = self._obj.shape
        if len(self.sequence) > 1 and self.sequence[-2] == "groupby":
            self._obj = self._obj.groupby(self.groups)
            aggFns, colNames = {}, {}
            for key, value in exprs.items():
                col = re.search(r'(?<=d\.)\w+', value).group().strip()
                expr_x = re.sub(r'd\.(\w+)', "x", value).strip()
                lambda_expr = f"lambda x: {expr_x}"
                aggFns[col] = eval(lambda_expr, self.global_vars, self.local_vars)
                colNames[col] = key
            self._obj = self._obj.agg(aggFns).rename(columns = colNames)
            self.info.append(f"Data was aggregated. The new shape of the data is {tuple(self._obj.shape)}. Previous dimension: {tuple(input_shape)}.")
        else:
            KeyError
        return self
    
    def summarize_across(self, cols, function, prefix):
        self.sequence.append("summarize")
        input_shape = self._obj.shape
        self._obj = self._obj.groupby(self.groups)
        column_functions = {col: function if isinstance(function, str) else function for col in cols}
        column_names = {col: f"{prefix}_{col}" for col in cols}
        self._obj = self._obj.agg(column_functions).rename(columns = column_names)
        self.info.append(f"Data was aggregated. The new shape of the data is {tuple(self._obj.shape)}. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def ungroup(self):
        self.sequence.append("ungroup")
        self._obj = self._obj.reset_index(drop = True)
        self.info.append(f"Data was ungrouped.")
        return self
    
    def to_long(self, id_cols, var_name, value_name):
        self.sequence.append("longform")
        input_shape = self._obj.shape
        self._obj = self._obj.melt(id_vars = id_cols, var_name = var_name, value_name = value_name)
        self.info.append(f"Data was transformed from wide to long form. New dimension is {tuple(self._obj.shape)}. Previous dimension: {tuple(input_shape)}.")
        return self
    
    def evaluate(self, printout = True): # give information -- if filter how many rows were filtered out, if ungroup the ungrouped columns
        if printout:
            for i, item in enumerate(self.info):
                print(f"{i+1}. {item}")
        return self._obj