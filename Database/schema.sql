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
