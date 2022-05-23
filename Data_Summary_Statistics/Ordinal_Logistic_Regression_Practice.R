require(foreign)
require(ggplot2)
require(MASS)
require(Hmisc)
require(reshape2)

dat <- read.dta("https://stats.idre.ucla.edu/stat/data/ologit.dta")
head(dat)

# Outcome variable: unlikely, somewhat likely, or very likely to apply (coded 1, 2, 3 respecively)
# Pared - 0/1 indicates if at least one parent has a graduate degree
# public (1 public, 0 private)
# gpa 

# one at a time apply table to 
lapply(dat[, c("apply", "pared", "public")], table)

## three way cross tabs (xtabs) and flatten the table
ftable(xtabs(~ public + apply + pared, data = dat))

summary(dat$gpa)
sd(dat$gpa)

ggplot(dat, aes(x = apply, y = gpa)) +
  geom_boxplot(size = .75) +
  geom_jitter(alpha = .5) +
  facet_grid(pared ~ public, margins = TRUE) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))

## fit ordered logit model and store results 'm'
m <- polr(apply ~ pared + public + gpa, data = dat, Hess=TRUE)

## view a summary of the model
summary(m)

## store table
(ctable <- coef(summary(m)))

## calculate and store p values
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2

## combined table
(ctable <- cbind(ctable, "p value" = p))

(ci <- confint(m)) # default method gives profiled CIs

confint.default(m) # CIs assuming normality

## odds ratios
exp(coef(m))

## OR and CI - Proponal Odds Ratios
exp(cbind(OR = coef(m), ci))
