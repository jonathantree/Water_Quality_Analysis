mydb <- dbConnect(RSQLite::SQLite(), "~/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3")

df <- dbGetQuery(mydb,"SELECT * FROM Census_Data INNER JOIN Contaminant_Summary on Census_Data.county_FIPS = Contaminant_Summary.county_FIPS")

dbDisconnect(mydb)

summary(df)



