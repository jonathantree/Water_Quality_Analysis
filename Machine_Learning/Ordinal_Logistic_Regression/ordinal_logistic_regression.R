require(foreign)
require(ggplot2)
require(MASS)
require(Hmisc)
require(reshape2)
require(RSQLite)

priority.df <- read.csv('data_with_ternary_priority_no_outliers.csv')
head(priority.df)

Priority <- factor(priority.df$Priority, levels = c(0,1,2));

mydb <- dbConnect(RSQLite::SQLite(), "~/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3")

df <- dbGetQuery(mydb,"SELECT * FROM Census_Data INNER JOIN Contaminant_Summary on Census_Data.county_FIPS = Contaminant_Summary.county_FIPS")

dbDisconnect(mydb)

head(df)

data.fr <- cbind(df,Priority)
head(data.fr)

df.priority <- subset(data.fr, select=c("Sum_ContaminantFactor","Simpson_Race_DI","Simpson_Ethnic_DI","Shannon_Race_DI","Shannon_Ethnic_DI","Gini_Index","Priority"))

head(df.priority)

# lapply(df.priority[, c("Sum_ContaminantFactor","Simpson_Race_DI","Simpson_Ethnic_DI","Shannon_Race_DI","Shannon_Ethnic_DI","Gini_Index","Priority")], table)

str(df.priority)

# fit ordered logit model and stores results as 'm'
m <- polr(Priority ~ Sum_ContaminantFactor + Simpson_Race_DI + Simpson_Ethnic_DI
          + Shannon_Race_DI + Shannon_Ethnic_DI + Gini_Index, data=df.priority, Hess=TRUE, method='probit')

#view a summary
summary(m)
