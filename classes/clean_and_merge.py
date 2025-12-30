
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Clean_and_Merge(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # List of categorical and numerical features
        categorical_attributes = ['preferred_foot', 'attacking_work_rate', 'defensive_work_rate']
        numerical_attributes = ['potential', 'crossing', 'finishing', 'heading_accuracy',
            'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
            'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility',
            'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength',
            'long_shots', 'aggression', 'interceptions', 'positioning', 'vision', 'penalties',
            'marking', 'standing_tackle', 'sliding_tackle', 'gk_diving', 'gk_handling', 'gk_kicking',
            'gk_positioning', 'gk_reflexes']

        player_ids = df['player_fifa_api_id'].unique()
        new_df = pd.DataFrame()

        for player_id in player_ids:
            index = np.where(df['player_fifa_api_id'] == player_id)[0]

            # Aggregate numerical features by mean
            temp_num = df.iloc[index][numerical_attributes].mean()

            # Aggregate categorical features by mode and handle empty mode result
            temp_cat = df.iloc[index][categorical_attributes].mode()
            if not temp_cat.empty:
                temp_cat = temp_cat.iloc[0]
            else:
                temp_cat = pd.Series([np.nan] * len(categorical_attributes), index=categorical_attributes)

            # Create a new DataFrame for this player's aggregated data
            temp_df = pd.DataFrame(data=[temp_num.values], columns=temp_num.index)
            temp_df[temp_cat.index] = temp_cat.values

            # Use pd.concat instead of append to combine DataFrames
            new_df = pd.concat([new_df, temp_df], ignore_index=True)

        # Data cleaning (replacing unwanted values)
        # Data cleaning (replacing unwanted values)
        to_drop = ['norm', 'y', 'le', 'stoc']
        for col in to_drop:
            new_df['attacking_work_rate'] = new_df['attacking_work_rate'].replace(col, np.nan)

        to_drop = ['ormal', 'ean', 'es', 'tocky', '_0', 'o']
        for col in to_drop:
            new_df['defensive_work_rate'] = new_df['defensive_work_rate'].replace(col, np.nan)

        # Adjust the work rate into categories (using assignment instead of inplace=True)
        low_class = ['0', '1', '2']
        medium_class = ['3', '4', '5', '6']
        high_class = ['7', '8', '9']

        for i in low_class:
            new_df['defensive_work_rate'] = new_df['defensive_work_rate'].replace(i, 'low')

        for i in medium_class:
            new_df['defensive_work_rate'] = new_df['defensive_work_rate'].replace(i, 'medium')

        for i in high_class:
            new_df['defensive_work_rate'] = new_df['defensive_work_rate'].replace(i, 'high')


        # Return the cleaned and merged DataFrame
        return new_df.reset_index(drop=True)


