import pandas as pd


def get_activity_levels(df):
    levels = df.loc[:, ['act_0','act_1','act_2']].copy()
    levels['total'] = levels.apply(lambda row : row.act_0 + row.act_1 + row.act_2, axis=1)
    levels['stationaryLevel'] = levels.apply(lambda row : row.act_0 / row.total, axis=1)
    levels['walkingLevel'] = levels.apply(lambda row : row.act_1 / row.total, axis=1)
    levels['runningLevel'] = levels.apply(lambda row : row.act_2 / row.total, axis=1)
    return levels.loc[:, ['stationaryLevel','walkingLevel','runningLevel']]

    
def addSedentaryLevel(df, metValues=(1.3, 5, 8.3)):
    '''
    Calculates de metLevel feature from the metValues

    '''

    levels = get_activity_levels(df)

    metLevel = (levels['stationaryLevel'] * metValues[0] +
                levels['walkingLevel'] * metValues[1] +
                levels['runningLevel'] * metValues[2])
    df['slevel'] = metLevel
    return df


def addSedentaryClasses(df, drop_slevel= True):
    """
    Generate an sclass column in the dataframe with true if sedentary and false if not sedentary

    """
    dfcopy = df.copy()
    dfcopy['sclass'] = (df['slevel'] < 1.5) * 1.0
    if drop_slevel:
        dfcopy.drop(['slevel'], inplace=True, axis=1)
    return dfcopy


def makeDummies(df):
    '''
    Transforms categorical features into dummy features (one boolean feature for each categorical possible value)

    '''
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.concat([df, pd.get_dummies(df[categorical_cols])], axis=1)
    df.drop(categorical_cols, inplace=True, axis=1)
    return df


def downgrade_datatypes(df):
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric, downcast='signed')
    df[converted_int.columns] = converted_int
    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric, downcast='float')
    df[converted_float.columns] = converted_float
    return df


def delete_user(df, user):
    '''
    Deletes a specific user.

    '''
    return df.copy().loc[df.index.get_level_values(0) != user]


def delete_sleep_buckets(df):
    return df.loc[(df['slevel'] >= 1.5) |
                  ((df.index.get_level_values(1).hour < 22) &
                   (df.index.get_level_values(1).hour > 5))]