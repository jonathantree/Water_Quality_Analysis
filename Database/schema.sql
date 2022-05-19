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