import sqlite3
import pandas as pd
import os

con = sqlite3.connect('Test_Data_DB.db') # connect to the database

cur = con.cursor() # instantiate a cursor obj

#path to the data
data_dir = os.path.join('AR')

print('Connected to the database')

print('Loading the data and renaming some columns')
# #Load the dataframes and rename columns to match the DB
utils_df = pd.read_csv(os.path.join(data_dir, 'cleaned.csv'))
utils_df = utils_df.rename(columns={'Population Served':'Population_Served'})

print('Updating the Utilities Table')
#Add the utilities df to a temp table in DB to check against the existing Utilities table
utils_df.to_sql('utilities', con, if_exists='replace', index=False, dtype={'Contaminant_Factor': 'INTEGER', 'county_FIPS' : 'TEXT REFERENCES FIPS_Codes(fips)' })


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
con.commit()

# con.execute('''
# PRAGMA foreign_keys = 0;

# CREATE TABLE sqlitestudio_temp_table AS SELECT *
#                                           FROM Contaminant_Summary;

# DROP TABLE Contaminant_Summary;

# CREATE TABLE Contaminant_Summary (
#     County_FIPS             PRIMARY KEY,
#     Num_Contaminants,
#     Sum_Population_Served,
#     Sum_ContaminantFactor,
#     Min_Contaminant_Factor,
#     Max_Contaminant_Factor,
#     Avg_Contaminant_Factor
# );
# ''')
# con.execute('''
# INSERT INTO Contaminant_Summary (
#                                     County_FIPS,
#                                     Num_Contaminants,
#                                     Sum_Population_Served,
#                                     Sum_ContaminantFactor,
#                                     Min_Contaminant_Factor,
#                                     Max_Contaminant_Factor,
#                                     Avg_Contaminant_Factor
#                                 )
#                                 SELECT County_FIPS,
#                                        Num_Contaminants,
#                                        Sum_Population_Served,
#                                        Sum_ContaminantFactor,
#                                        Min_Contaminant_Factor,
#                                        Max_Contaminant_Factor,
#                                        Avg_Contaminant_Factor
#                                   FROM sqlitestudio_temp_table;

# DROP TABLE sqlitestudio_temp_table;

# PRAGMA foreign_keys = 1;
# ''')



con.commit()
con.close()