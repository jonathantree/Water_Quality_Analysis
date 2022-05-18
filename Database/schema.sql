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
                                          FROM Utilities_Scrape;

DROP TABLE Utilities_Scrape;

CREATE TABLE Utilities_Scrape (
    Utility            TEXT,
    Contaminant_Factor INTEGER,
    Zip                INTEGER,
    county_FIPS        INTEGER REFERENCES FIPS_Codes (fips) 
);

INSERT INTO Utilities_Scrape (
                                 Utility,
                                 Contaminant_Factor,
                                 Zip,
                                 county_FIPS
                             )
                             SELECT Utility,
                                    Contaminant_Factor,
                                    Zip,
                                    county_FIPS
                               FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

PRAGMA foreign_keys = 1;


CREATE TABLE Contaminant_Summary AS
    SELECT 
        Utilities_Scrape.county_FIPS,
        COUNT(county_FIPS) as Num_Contaminants,
        SUM(Contaminant_Factor) as Sum_ContaminantFactor,
        min(Contaminant_Factor) as Min_Contaminant_Factor,
        max(Contaminant_Factor) as Max_Contaminant_Factor,
        round(avg(Contaminant_Factor),2) as Avg_Contaminant_Factor
    FROM 
        Utilities_Test
    GROUP BY
    county_FIPS;