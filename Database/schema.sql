CREATE TABLE utility_zips(
	"Utility name" TEXT PRIMARY KEY NOT NULL,	
	"City" TEXT,
	"People served"	TEXT,
	"Zip" INT
)

CREATE TABLE Utilities_TEST (
    Utility            TEXT    NOT NULL,
    Contaminant_Factor INTEGER NOT NULL,
    Zip                INTEGER NOT NULL,
    county_FIPS        INTEGER REFERENCES FIPS_Codes (fips) 
);

PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM Utilities;

DROP TABLE Utilities;

CREATE TABLE Utilities (
    State              TEXT,
    Zip                INTEGER,
    Utility            TEXT,
    Poplulation_Served INTEGER,
    Contaminant_Factor INTEGER,
    County_FIPS  REFERENCES FIPS_Codes (fips) 
);

CREATE TABLE all_contaminants(
State TEXT,
City TEXT,
Zip INTEGER,
Utility TEXT,
People served INTEGER,
Contaminant TEXT,
"Utility Measurement" INTEGER,
"EWG HEALTH GUIDELINE" INTEGER,
"Legal Limit" INTEGER,
Units TEXT,
Contaminant_Factor INTEGER
);

INSERT INTO Utilities (
                          State,
                          Zip,
                          Utility,
                          Poplulation_Served,
                          Contaminant_Factor,
                          County_FIPS
                      )
                      SELECT State,
                             Zip,
                             Utility,
                             Poplulation_Served,
                             Contaminant_Factor,
                             County_FIPS
                        FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

PRAGMA foreign_keys = 1;

CREATE TABLE "Census_Data" (
"county_FIPS" INTEGER PRIMARY KEY,
  "Geographic_Area_Name" TEXT,
  "County" TEXT,
  "GEOID" TEXT,
  "Total_Population" INTEGER,
  "White" INTEGER,
  "Black" INTEGER,
  "Native" INTEGER,
  "Asian" INTEGER,
  "Pacific_Islander" INTEGER,
  "Other" INTEGER,
  "Two_or_more_Races" INTEGER,
  "Hispanic" INTEGER,
  "Not_Hispanic" INTEGER,
  "Not_White" INTEGER,
  "pct_White" REAL,
  "pct_Black" REAL,
  "pct_Native" REAL,
  "pct_Asian" REAL,
  "pct_Pacific_Islander" REAL,
  "pct_Other" REAL,
  "pct_Not_White" REAL,
  "pct_Hispanic" REAL,
  "pct_Not_Hispanic" REAL,
  "pct_Two_or_more_Races" REAL,
  "Simpson_Race_DI" REAL,
  "Simpson_Ethnic_DI" REAL,
  "Shannon_Race_DI" REAL,
  "Shannon_Ethnic_DI" REAL,
  "Gini_Index" REAL
)