require(mlr3)
require(mlr3verse)
require(mlr3tuning)
require(mlr3learners)
require(mlr3measures)
require(data.table)
require(ggplot2)
require(tidyr)
require(dplyr)
require(caret)
require(gridExtra)
require(DataExplorer)
require(mlr3viz)

set.seed(123)

# Load data
data <- read.csv('bank_personal_loan.csv')
as.data.table(data)

# Pre-processing
data <- as.data.table(data)
data[["Personal.Loan"]] <- as.factor(data[["Personal.Loan"]])
data[["Securities.Account"]] <- as.factor(data[["Securities.Account"]])
data[["CD.Account"]] <- as.factor(data[["CD.Account"]])
data[["Online"]] <- as.factor(data[["Online"]])
data[["CreditCard"]] <- as.factor(data[["CreditCard"]])

# View data
names(data)
skimr::skim(data)
View(data)

# Data exploration and visualization
DataExplorer::plot_bar(data, ncol=3)
DataExplorer::plot_histogram(data, ncol=3)
DataExplorer::plot_boxplot(data, by="Personal.Loan", ncol=3)

# Split dataset
train.index <- createDataPartition(data$Personal.Loan, p=0.8, list=FALSE)
train <- as.data.table(data[train.index, ])
test <- as.data.table(data[-train.index, ])

# Instantiate cross-validation
task <- TaskClassif$new(id="Personal.Loan", backend=train, target="Personal.Loan", positive="1")
cv10 <- rsmp("cv", folds=10)
cv10$instantiate(task)

# Find learners that support the task specifications
supported_learners <- as.data.table(mlr_learners)[
  task_type == "classif" &
    grepl("factor", feature_types) &
    (grepl("numeric", feature_types))
]

# Initial benchmarking
lrn.xgboost <- lrn("classif.xgboost", predict_type="prob")
pl_xgb <- po("encode") %>>% po(lrn.xgboost)
learners_list <- lapply(supported_learners$key, function(key) lrn(key, predict_type="prob"))
learners_list <- c(list("pl_xgb"=pl_xgb), learners_list)

res <- benchmark(data.table(
  task=list(task),
  learner=learners_list,
  resampling=list(cv10)
), store_models=TRUE)

res$aggregate()
metrics <- res$aggregate(list(msr("classif.ce"),
                              msr("classif.acc"),
                              msr("classif.auc"),
                              msr("classif.fpr"),
                              msr("classif.fnr")))

removed <- metrics[classif.ce >= metrics[learner_id == "classif.featureless", classif.ce]]$learner_id
for(model in removed){
  cat("Model removed: ", model, "\n")
}
metrics <- metrics[classif.ce < metrics[learner_id == "classif.featureless", classif.ce]]
metrics <- metrics[, nr := seq_len(.N)]

ggplot(metrics, aes(x=reorder(learner_id, classif.ce), y=classif.ce)) +
  geom_bar(stat="identity", fill="steelblue") +
  labs(title="Classification Error Comparison",
       x="Model",
       y="Classification Error") +
  theme_minimal() +
  theme(axis.text.x=element_text(angle=45, hjust=1))

# Train and test tuned models
lrn.ranger <- lrn("classif.ranger", predict_type="prob", id="ranger")
lrn.rpart <- lrn("classif.rpart", predict_type="prob", id="rpart")
lrn.xgboost <- lrn("classif.xgboost", predict_type="prob", id="xgboost")
lrn.xgboost <- po("encode") %>>% po(lrn.xgboost)

params.ranger <- list(
  num.threads=1,
  mtry=6,
  min.node.size=5,
  sample.fraction=0.9537942
)

params.rpart <- list(
  xval=1,
  cp=0.006470291,
  minsplit=18,
  maxdepth=23
)

params.xgboost <- list(
  encode.method="one-hot",
  xgboost.early_stopping_set="none",
  xgboost.nrounds=200,
  xgboost.nthread=1,
  xgboost.eta=0.04828141,
  xgboost.max_depth=7,
  xgboost.subsample=0.7689715,
  xgboost.colsample_bytree=0.5692649,
  xgboost.min_child_weight=1.908943
)

lrn.ranger$param_set$values <- params.ranger
lrn.rpart$param_set$values <- params.rpart
lrn.xgboost$param_set$values <- params.xgboost

lrn.ranger$train(task)
lrn.rpart$train(task)
lrn.xgboost$train(task)

task.test <- TaskClassif$new(id="Personal.Loan.Test", backend=test, target="Personal.Loan", positive="1")

predict.ranger <- lrn.ranger$predict(task.test)
predict.rpart <- lrn.rpart$predict(task.test)
predict.xgboost <- lrn.xgboost$predict(task.test)

colnames <- c("classif.ce", "classif.acc", "classif.auc", "classif.fpr", "classif.fnr")
metrics.ranger <- as.data.table(list(msr("classif.ce")$score(predict.ranger),
                                     msr("classif.acc")$score(predict.ranger),
                                     msr("classif.auc")$score(predict.ranger),
                                     msr("classif.fpr")$score(predict.ranger),
                                     msr("classif.fnr")$score(predict.ranger)))
colnames(metrics.ranger) <- colnames

metrics.rpart <- as.data.table(list(msr("classif.ce")$score(predict.rpart),
                                    msr("classif.acc")$score(predict.rpart),
                                    msr("classif.auc")$score(predict.rpart),
                                    msr("classif.fpr")$score(predict.rpart),
                                    msr("classif.fnr")$score(predict.rpart)))
colnames(metrics.rpart) <- colnames

metrics.rxgboost <- as.data.table(list(msr("classif.ce")$score(predict.xgboost$xgboost.output),
                                       msr("classif.acc")$score(predict.xgboost$xgboost.output),
                                       msr("classif.auc")$score(predict.xgboost$xgboost.output),
                                       msr("classif.fpr")$score(predict.xgboost$xgboost.output),
                                       msr("classif.fnr")$score(predict.xgboost$xgboost.output)))
colnames(metrics.rxgboost) <- colnames

predict.xgboost
View(metrics.ranger)
View(metrics.rpart)
View(metrics.rxgboost)

ratios <- data.frame(
  Model=rep(c("Ranger", "Rpart", "XGBoost"), each=2),
  Metric=rep(c("FPR", "FNR"), 3),
  Value=c(c(metrics.ranger[["classif.fpr"]], metrics.ranger[["classif.fnr"]]), 
            c(metrics.rpart[["classif.fpr"]], metrics.rpart[["classif.fnr"]]), 
            c(metrics.rxgboost[["classif.fpr"]], metrics.rxgboost[["classif.fnr"]]))
)

accuracies <- data.frame(
  Model=c("Ranger", "Rpart", "XGBoost"),
  Accuracy=c(metrics.ranger[["classif.acc"]], 
               metrics.rpart[["classif.acc"]], 
               metrics.rxgboost[["classif.acc"]])
)

ggplot(ratios, aes(x=Model, y=Value, fill=Metric)) +
  geom_bar(stat="identity", position="dodge") +
  labs(title="False Positive/Negative Comparison", x="Model", y="Rate") +
  scale_fill_manual(values=c("FPR"="steelblue", "FNR"="salmon")) +
  theme_minimal()

autoplot(predict.ranger, type="roc") +
  ggtitle("Ranger Model ROC Curve") +                 
  theme(plot.title=element_text(hjust=0.5)) + 
  geom_line(size=0.9) 

lrn.ranger.new <- lrn("classif.ranger", predict_type="prob", id="ranger")
lrn.rpart.new <- lrn("classif.rpart", predict_type="prob", id="rpart")
lrn.xgboost.new <- lrn("classif.xgboost", predict_type="prob", id="xgboost")
lrn.xgboost.new <- po("encode") %>>% po(lrn.xgboost.new)

res.tuned <- benchmark(data.table(
  task=list(task),
  learner=list(lrn.ranger.new, lrn.rpart.new, lrn.xgboost.new),
  resampling=list(cv10)
), store_models=TRUE)

res.tuned$aggregate()
metrics.tuned <- res.tuned$aggregate(list(msr("classif.ce"),
                                          msr("classif.acc"),
                                          msr("classif.auc"),
                                          msr("classif.fpr"),
                                          msr("classif.fnr")))
metrics.tuned
############################
# Do not run this section (it will take a WHILE to run)
# Parameter tuning used to pick the parameters for ranger, rpart and xgboost
require(mlr3)
require(mlr3verse)
require(mlr3tuning)
require(mlr3learners)
require(data.table)
require(caret)

set.seed(123)

data <- read.csv('bank_personal_loan.csv')

data <- as.data.table(data)
data[["Personal.Loan"]] <- as.factor(data[["Personal.Loan"]])
data[["Securities.Account"]] <- as.factor(data[["Securities.Account"]])
data[["CD.Account"]] <- as.factor(data[["CD.Account"]])
data[["Online"]] <- as.factor(data[["Online"]])
data[["CreditCard"]] <- as.factor(data[["CreditCard"]])

train.index <- createDataPartition(data$Personal.Loan, p=0.9, list=FALSE)
train <- as.data.table(data[train.index, ])

task <- TaskClassif$new(id="Personal.Loan", backend=train, target="Personal.Loan", positive="1")
cv5 <- rsmp("cv", folds=5)
cv5$instantiate(task)

tlrn.ranger <- lrn("classif.ranger", predict_type="prob", id="ranger")
tlrn.rpart <- lrn("classif.rpart", predict_type="prob", id="rpart")
tlrn.xgboost <- lrn("classif.xgboost", predict_type="prob", id="xgboost")
tlrn.xgboost <- po("encode") %>>% po(tlrn.xgboost)

param_set.ranger <- ParamSet$new(list(
  ParamInt$new("mtry", lower=1, upper=12),
  ParamInt$new("min.node.size", lower=1, upper=20),
  ParamDbl$new("sample.fraction", lower=0.1, upper=1)
))

param_set.rpart <- ParamSet$new(list(
  ParamDbl$new("cp", lower=0, upper=1),
  ParamInt$new("minsplit", lower=1, upper=20),
  ParamInt$new("maxdepth", lower=1, upper=30)
))

param_set.xgboost <- ParamSet$new(list(
  ParamInt$new("xgboost.nrounds", lower=1, upper=1000),
  ParamDbl$new("xgboost.eta", lower=0, upper=1),
  ParamInt$new("xgboost.max_depth", lower=1, upper=10),
  ParamDbl$new("xgboost.subsample", lower=0, upper=1),
  ParamDbl$new("xgboost.colsample_bytree", lower=0, upper=1),
  ParamDbl$new("xgboost.min_child_weight", lower=0, upper=10)
))

tuner <- tnr("random_search", batch_size=20)

tune.ranger <- TuningInstanceSingleCrit$new(
  task=task,
  learner=tlrn.ranger,
  resampling=cv5,
  measure=msr("classif.acc"),
  search_space=param_set.ranger,
  terminator=trm("evals", n_evals=300)
)

tune.rpart <- TuningInstanceSingleCrit$new(
  task=task,
  learner=tlrn.rpart,
  resampling=cv5,
  measure=msr("classif.acc"),
  search_space=param_set.rpart,
  terminator=trm("evals", n_evals=300)
)

tune.xgboost <- TuningInstanceSingleCrit$new(
  task=task,
  learner=tlrn.xgboost,
  resampling=cv5,
  measure=msr("classif.acc"),
  search_space=param_set.xgboost,
  terminator=trm("evals", n_evals=300)
)

tuner$optimize(tune.ranger)
tuner$optimize(tune.rpart)
tuner$optimize(tune.xgboost)

tune.ranger$result
tune.rpart$result
tune.xgboost$result

tune.ranger$result$classif.acc
tune.rpart$result$classif.acc
tune.xgboost$result$classif.acc
############################