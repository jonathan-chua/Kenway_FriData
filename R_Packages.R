#### This code installs or updates some commonly used packages
# Last update: 2018 April 26

packageNames = c(
  "glmnet", 
  "class", 
  "recommenderlab",
  "devtools",
  "shiny",
  "tidyverse",
  "knitr",
  "readxl",
  "lubridate",
  "broom",
  "hms",
  "XML",
  "kknn",
  "tree",
  "randomForest",
  "rpart",
  "gbm",
  "acepack",
  "data.table",
  "rpart.plot",
  "ROCR",
  "lift",
  "caret",
  "dplyr",
  "baseballr"
  )

for (pkgName in packageNames) {
  if(!(pkgName %in% rownames(installed.packages()))) {
    install.packages(pkgName, dependencies=TRUE)
  }
}
update.packages(ask=FALSE)
