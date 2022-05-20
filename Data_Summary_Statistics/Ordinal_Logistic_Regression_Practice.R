library(carData)
library(MASS)

data(WVS)
head(WVS)
summary(WVS)

ggplot2(WVS,aes(x = poverty, y=age, fill = poverty)) + 
  geom_boxplot(size=0.75) + facet_grid(country ~ gender, margins = FALSE) +
   theme(axis.text.x = element_text(angle=45,hjust=1,vjust=1))