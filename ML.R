### 加载??? ###
library(tidyverse)
library(klaR)
library(kernlab)
library(caret)
library(pROC)
### 载入数据 ###
train <- read_csv("train.csv")
test <- read_csv("test.csv")
validation <- read_csv("validation.csv")
### 将test集的变量与train和valid一??? ###
train <- train[, colnames(test)]
validation <- validation[, colnames(test)]
### 使用管道符定义变??? ####
train <- train %>%
    mutate_at(-3, as.factor)
test <- test %>%
    mutate_at(-3, as.factor)
validation <- validation %>%
    mutate_at(-3, as.factor)
### 设置种子??? ###
seed <- 1234

### 随机森林 ###

rf_train <- train(outcome ~ ., data = train, method = "rf")
rf_train$finalModel
### 内部验证集valid ###
rf_pred_valid <- predict(rf_train, newdata = validation)
rf_cm_valid <- confusionMatrix(validation$outcome, rf_pred_valid, positive = "1")
rf_cm_valid
### 外部验证集test ###
rf_pred <- predict(rf_train, newdata = test)
rf_cm <- confusionMatrix(test$outcome, rf_pred, positive = "1")
rf_cm
### 计算ROC ###
rf_pred_prob <- predict(rf_train, newdata = test, type = "prob")
rf_roc <- roc(test$outcome, rf_pred_prob[,2])
rf_roc

### 决策???##递归分割???####

rp_train <- train(outcome ~ ., data = train, method = "rpart")
rp_train$finalModel

rp_pred_valid <- predict(rp_train, newdata = validation)
rp_cm_valid <- confusionMatrix(validation$outcome, rp_pred_valid, positive = "1")
rp_cm_valid

rp_pred <- predict(rp_train, newdata = test)
rp_cm <- confusionMatrix(test$outcome, rp_pred, positive = "1")
rp_cm

rp_pred_prob <- predict(rp_train, newdata = test, type = "prob")
rp_roc <- roc(test$outcome, rp_pred_prob[,2])
rp_roc

### 贝叶??? ###
nb_train <- train(outcome ~ ., data = train, method = "nb")
nb_train$finalModel

nb_pred_valid <- predict(nb_train, newdata = validation)
nb_cm_valid <- confusionMatrix(validation$outcome, nb_pred_valid, positive = "1")
nb_cm_valid

nb_pred <- predict(nb_train, newdata = test)
nb_cm <- confusionMatrix(test$outcome, nb_pred, positive = "1")
nb_cm

nb_pred_prob <- predict(nb_train, newdata = test, type = "prob")
nb_roc <- roc(test$outcome, nb_pred_prob[,2])
nb_roc

### KNN ###
knn_train <- train(outcome ~ ., data = train, method = "knn")
knn_train$finalModel

knn_pred_valid <- predict(knn_train, newdata = validation)
knn_cm_valid <- confusionMatrix(validation$outcome, knn_pred_valid, positive = "1")
knn_cm_valid

knn_pred <- predict(knn_train, newdata = test)
knn_cm <- confusionMatrix(test$outcome, knn_pred, positive = "1")
knn_cm

knn_pred_prob <- predict(knn_train, newdata = test, type = "prob")
knn_roc <- roc(test$outcome, knn_pred_prob[,2])
knn_roc

### SVM ###
svm_train <- train(outcome ~ ., data = train %>%
                       mutate(outcome = make.names(outcome)), method = "svmLinear",
                   trControl = trainControl(method = "repeatedcv", repeats = 5, 
                                            classProbs =  TRUE))
svm_train$finalModel

svm_pred_valid <- predict(svm_train, newdata = validation) %>%
    fct_relabel(~{c("0", "1")})
svm_cm_valid <- confusionMatrix(validation$outcome, svm_pred_valid, positive = "1")
svm_cm_valid

svm_pred <- predict(svm_train, newdata = test) %>%
    fct_relabel(~{c("0", "1")})
svm_cm <- confusionMatrix(test$outcome, svm_pred, positive = "1")
svm_cm

svm_pred_prob <- predict(svm_train, newdata = test, type = "prob")
svm_roc <- roc(test$outcome, svm_pred_prob[,2])
svm_roc

### 逻辑回归 ###
glm_train <- train(outcome ~ ., data = train, method = "glm")
glm_train$finalModel

glm_pred_valid <- predict(glm_train, newdata = validation)
glm_cm_valid <- confusionMatrix(validation$outcome, glm_pred_valid, positive = "1")
glm_cm_valid

glm_pred <- predict(glm_train, newdata = test)
glm_cm <- confusionMatrix(test$outcome, glm_pred, positive = "1")
glm_cm

glm_pred_prob <- predict(glm_train, newdata = test, type = "prob")
glm_roc <- roc(test$outcome, glm_pred_prob[,2])
glm_roc

### Summary ###

summ <- list(rf_cm, rp_cm, nb_cm, knn_cm, svm_cm, glm_cm) 
names <- c("Random Forest", "Decision Tree", "Naive Bayes", "KNN", "SVM", "Logistic")

summ_tab <- map2_dfr(summ, names, ~{
    data.frame(Model = .y,
               Accuracy = .x$overall[1],
               Sensitivity = .x$byClass[1],
               Specificity = .x$byClass[2])
})

summ_tab
write_tsv(summ_tab, "summary_table.txt")

### Correlation heatmap ###

train[, -1] %>%
    mutate_all(as.numeric) %>%
    cor() %>%
    as.data.frame() %>%
    rownames_to_column("y") %>%
    pivot_longer(-y, names_to = "x") %>%
    ggplot(aes(x, y)) +
    geom_tile(aes(fill = value)) +
    scale_fill_distiller(palette = "RdYlBu", name = NULL) +
    scale_y_discrete(limit = rev, expand = expansion(c(0, 0))) +
    scale_x_discrete(expand = expansion(c(0, 0))) +
    theme_minimal() +
    coord_equal() +
    theme(axis.title = element_blank(),
          axis.text = element_text(size = rel(1.2),
                                   colour = "black",
                                   face = "bold"),
          axis.ticks = element_line(size = 0.8),
          axis.text.x = element_text(angle = 90,
                                     hjust = 1,
                                     vjust = 0.5),
          legend.key.height = unit(2.1, "cm")) 

ggsave("pearson_heatmap.pdf", width = 6, height = 5)

train[, -1] %>%
    mutate_all(as.numeric) %>%
    cor(method = "spearman") %>%
    as.data.frame() %>%
    rownames_to_column("y") %>%
    pivot_longer(-y, names_to = "x") %>%
    ggplot(aes(x, y)) +
    geom_tile(aes(fill = value)) +
    scale_fill_distiller(palette = "RdYlBu", name = NULL) +
    scale_y_discrete(limit = rev, expand = expansion(c(0, 0))) +
    scale_x_discrete(expand = expansion(c(0, 0))) +
    theme_minimal() +
    coord_equal() +
    theme(axis.title = element_blank(),
          axis.text = element_text(size = rel(1.2),
                                   colour = "black",
                                   face = "bold"),
          axis.ticks = element_line(size = 0.8),
          axis.text.x = element_text(angle = 90,
                                     hjust = 1,
                                     vjust = 0.5),
          legend.key.height = unit(2.1, "cm")) 
###保存??? pdf格式 ###
ggsave("spearman_heatmap.pdf", width = 6, height = 5)

### ROC曲线绘制 ###

rocs <- list(rf_roc, rp_roc, nb_roc, knn_roc, svm_roc, glm_roc) 
names <- c("Random Forest", "Decision Tree", "Naive Bayes", "KNN", "SVM", "Logistic")

plot_df <- map2_dfr(rocs, names, ~{
    data.frame(Sensitivity = rev(.x$sensitivities),
               Specificity = rev(.x$specificities)) %>%
        mutate(Model = .y,
               AUC = as.numeric(.x$auc))
}) %>%
    mutate(Specificity = 1 - Specificity)

auc_df <- plot_df %>%
    dplyr::select(Model, AUC) %>%
    distinct() %>%
    add_row(Model = "AUC:", .before = 1) %>%
    mutate(y = seq(0.5, 0.1, length.out = 7))

plot_df %>%
    ggplot(aes(x = Specificity, y = Sensitivity)) +
    geom_abline(slope = 1) +
    geom_line(aes(group = Model, color = Model), size = 0.8) +
    scale_color_brewer(palette = "Spectral") +
    labs(x = "1 - Specificity") +
    annotate("text", x = 0.8, y = auc_df$y, label = auc_df$Model, hjust = 1) +
    annotate("text", x = 0.83, y = auc_df$y[-1], label = sprintf("%.2f", auc_df$AUC[-1]), hjust = 0) +
    theme_bw() +
    coord_equal() +
    theme(axis.text = element_text(size = rel(1.2),
                                   colour = "black",
                                   face = "bold"),
          axis.title = element_text(size = rel(1.2),
                                    colour = "black",
                                    face = "bold"),
          legend.text = element_text(size = rel(1.1),
                                     colour = "black",
                                     face = "bold"),
          legend.title = element_text(size = rel(1.1),
                                      colour = "black",
                                      face = "bold"),
          panel.grid = element_blank()) 

ggsave("roc.pdf", width = 8, height = 6)