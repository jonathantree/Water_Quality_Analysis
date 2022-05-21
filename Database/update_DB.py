import sqlite3
from db_utils import db_connect
import pandas as pd
import os

con = db_connect() # connect to the database

cur = con.cursor() # instantiate a cursor obj

#path to the data
data_dir = os.path.join('..', 'Resources', 'Data', 'Cleaned_Data')

print('Connected to the database')

print('Loading the data and renaming some columns')
#Load the dataframes and rename columns to match the DB
utils_df = pd.read_csv(os.path.join(data_dir, 'master_utilities.csv'))
utils_df = utils_df.rename(columns={'Population Served':'Population_Served'})

contaminants_df = pd.read_csv(os.path.join(data_dir, 'all_contaminants.csv'))
contaminants_df = contaminants_df.rename(columns={
    'People served' : 'People_served',
    'Utility Measuremnt' : 'Utility_Measurement',
    'EWG HEALTH GUIDELINE' : 'EWG_Health_Guideline',
    'Legal Limit' : 'Legal_Limit'
    
})

print('Updating the Utilities Table')
#Add the utilities df to a temp table in DB to check against the existing Utilities table
utils_df.to_sql('utilities_temp', con, if_exists='replace', index=False, dtype={'Contaminant_Factor': 'INTEGER', 'county_FIPS' : 'REFERENCES FIPS_Codes(fips)' })

con.execute('''
    INSERT INTO Utilities(State, Zip, Utility, Population_Served, Contaminant_Factor, County_FIPS)
    SELECT 
        utilities_temp.State,
        utilities_temp.Zip,
        utilities_temp.Utility,
        utilities_temp.Population_Served,
        utilities_temp.Contaminant_Factor,
        utilities_temp.county_FIPS
     
    FROM 
        utilities_temp
    WHERE NOT EXISTS (
        SELECT 1 FROM Utilities WHERE Utilities.Utility = utilities_temp.Utility
    )
''')

con.execute('DROP TABLE IF EXISTS utilities_temp')

print('Updating the all_conaminants Table')

contaminants_df.to_sql('all_contaminants', con, if_exists='replace', index=False, dtype={'Contaminant_Factor': 'INTEGER' })

con.execute('DROP TABLE IF EXISTS Contaminant_Summary')

print('Creating the ContaminantSummary Table')
con.execute('''
CREATE TABLE Contaminant_Summary AS SELECT Utilities.County_FIPS,
                                           COUNT(county_FIPS) AS Num_Contaminants,
                                           SUM(Population_Served) AS Sum_Population_Served,
                                           SUM(Contaminant_Factor) AS Sum_ContaminantFactor,
                                           min(Contaminant_Factor) AS Min_Contaminant_Factor,
                                           max(Contaminant_Factor) AS Max_Contaminant_Factor,
                                           round(avg(Contaminant_Factor), 2) AS Avg_Contaminant_Factor
                                      FROM Utilities
                                     GROUP BY County_FIPS;
''')

#Add Primary Key on the summary table

con.execute('''
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM Contaminant_Summary;

DROP TABLE Contaminant_Summary;

CREATE TABLE Contaminant_Summary (
    County_FIPS             PRIMARY KEY,
    Num_Contaminants,
    Sum_Population_Served,
    Sum_ContaminantFactor,
    Min_Contaminant_Factor,
    Max_Contaminant_Factor,
    Avg_Contaminant_Factor
);

INSERT INTO Contaminant_Summary (
                                    County_FIPS,
                                    Num_Contaminants,
                                    Sum_Population_Served,
                                    Sum_ContaminantFactor,
                                    Min_Contaminant_Factor,
                                    Max_Contaminant_Factor,
                                    Avg_Contaminant_Factor
                                )
                                SELECT County_FIPS,
                                       Num_Contaminants,
                                       Sum_Population_Served,
                                       Sum_ContaminantFactor,
                                       Min_Contaminant_Factor,
                                       Max_Contaminant_Factor,
                                       Avg_Contaminant_Factor
                                  FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

PRAGMA foreign_keys = 1;
''')



con.commit()
con.close()