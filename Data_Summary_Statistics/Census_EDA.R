library(RSQLite)
library(ggplot2)
library(ggthemes)
library(dplyr)
library(corrgram)
library(corrplot)

mydb <- dbConnect(RSQLite::SQLite(), "~/uofo-virt-data-pt-12-2021-u-b/Water_Quality_Analysis/Database/database.sqlite3")

df <- dbGetQuery(mydb,"SELECT * FROM Census_Data INNER JOIN Contaminant_Summary on Census_Data.county_FIPS = Contaminant_Summary.county_FIPS")

dbDisconnect(mydb)

head(df)

# Shapiro test

numeric_df <- df[c('Total_Population', 'White', 'Black',
                   'Native', 'Asian', 'Pacific_Islander', 'Other', 'Two_or_more_Races',
                   'Hispanic', 'Not_Hispanic', 'Not_White', 'pct_White', 'pct_Black',
                   'pct_Native', 'pct_Asian', 'pct_Pacific_Islander', 'pct_Other',
                   'pct_Not_White', 'pct_Hispanic', 'pct_Not_Hispanic',
                   'pct_Two_or_more_Races', 'Simpson_Race_DI', 'Simpson_Ethnic_DI',
                   'Shannon_Race_DI', 'Shannon_Ethnic_DI', 'Gini_Index','Num_Contaminants',
                   'Avg_Contaminant_Factor','Sum_ContaminantFactor')]

apply(numeric_df,2,shapiro.test)

#look at correlation
numeric.cols <- sapply(numeric_df,is.numeric)
cor.data <- cor(numeric_df[,numeric.cols])
print(corrplot(cor.data,method = 'color'))

# Corrgram
corrgram(df,order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt)

# histogram on Sum_Contaminant_Factor which is the output variable we are interested in
print(ggplot(df,aes(x=Sum_ContaminantFactor)) + geom_histogram(bins=30, alpha=0.5, fill='blue'))

library(caTools)
set.seed(101)
sample <- sample.split(numeric_df$Sum_ContaminantFactor, SplitRatio = 0.7)
# 70% of my data goes to train
train <- subset(numeric_df, sample == TRUE)
# 30% will be test
test <- subset(numeric_df, sample == FALSE)

model <- lm(Sum_ContaminantFactor ~., data = train)
#run the model
# print summary
print(summary(model))

# Visualize the model by plotting the residuals
res <- residuals(model)
res <- as.data.frame(res)
head(res)

ggplot(res, aes(res)) + geom_histogram(fill='blue',alpha=0.5)

# predictions
contaminant.predictions <- predict(model, test)

results <- cbind(contaminant.predictions,test$Sum_ContaminantFactor)
colnames(results) <- c('predicted','actual')
results <- as.data.frame(results)
print(head(results))

# Take care of negative values
to_zero <- function(x) {
  if (x<0){
    return(0)
  }else{
    return(x)
  }
}

#apply zero function
results$predicted <- sapply(results$predicted,to_zero)

# mean squared error
MSE <- mean( (results$actual - results$predicted)^2)
print('MSE')
print(MSE)
#RMSE
print('Sauare Root of MSE')
print(MSE^0.5)

######### sum of squared error, sum of squared total
SSE <- sum( (results$predicted - results$actual)^2 )
SST <- sum( (mean(df$Sum_ContaminantFactor) - results$actual)^2 )

R2 <- 1 - SSE/SST
print('R2')
print(R2)