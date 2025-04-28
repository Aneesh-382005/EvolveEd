import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    """Preprocesses the DataFrame for machine learning models.
    Works only for personalized_learning_dataset.csv."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # Avoid modifying the original DataFrame
        self.labelEncoders = {}
        self.scaler = StandardScaler()
        self.categoricalColumns = ['Gender', 'Education_Level', 'Course_Name', 'Learning_Style', 'Engagement_Level', 'Dropout_Likelihood']
        self.numericalColumns = ['Age', 'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores',
                                 'Forum_Participation', 'Assignment_Completion_Rate']

    def encodeCategorical(self) -> pd.DataFrame:
        """Encodes categorical columns using LabelEncoder."""
        for column in self.categoricalColumns:
            if column not in self.df.columns:
                raise ValueError(f"Column {column} not found in DataFrame")
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column].astype(str))  # Ensure all data is string
            self.labelEncoders[column] = le
        return self.df
    
    def scaleNumerical(self) -> pd.DataFrame:
        """Scales numerical columns using StandardScaler."""
        for column in self.numericalColumns:
            if column not in self.df.columns:
                raise ValueError(f"Column {column} not found in DataFrame")
        self.df[self.numericalColumns] = self.scaler.fit_transform(self.df[self.numericalColumns])
        return self.df
    
    def preprocess(self) -> pd.DataFrame:
        """Encodes categorical columns and scales numerical columns."""
        self.df = self.encodeCategorical()
        self.df = self.scaleNumerical()
        return self.df