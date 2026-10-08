# reading data
data <- readRDS("handwriting.rds")
dim(data$images) # 25000 x 28 x 28 x 1
head(data$metadata)

# Input data to Keras
library(keras3)
use_backend("tensorflow")
x <- data$images / 255
y_person <- data$metadata$person_id - 1L

# Splitting data by session. Session 1-3 for training, session 4 for validation and session 5 for testing

# Picking rows by session
meta <- data$metadata # this is the table with one row per image

train_rows <- which(meta$session %in% 1:3) # row numbers of session 1-3

val_rows <- which(meta$session == 4)
test_rows <- which(meta$session == 5)

# images and labels
x_train <- x[train_rows, , , , drop = FALSE]
y_train <- y_person[train_rows]

x_val <- x[val_rows, , , , drop = FALSE]
y_val <- y_person[val_rows]

x_test <- x[test_rows, , , , drop = FALSE]
y_test <- y_person[test_rows]

# Checking if the split worked
dim(x_train); dim(x_val); dim(x_test)
table(y_train)

# CNN Model building
model <- keras_model_sequential(input_shape = c(28, 28, 1)) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |> #keeps positive signals only
  layer_max_pooling_2d(pool_size = 2) |>       #shrinks the image by half
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |> 
  layer_max_pooling_2d(pool_size = 2) |>
  layer_flatten() |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 20, activation = "softmax")

# Compiling and training
model |> compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history <- model |> fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_data = list(x_val, y_val)
)

# Checking for leakage using random split
set.seed(123)
idx <- sample(nrow(x))                       
rand_train <- idx[1:15000]
rand_val   <- idx[15001:20000]

x_tr_rand <- x[rand_train, , , , drop = FALSE]
y_tr_rand <- y_person[rand_train]
x_va_rand <- x[rand_val, , , , drop = FALSE]
y_va_rand <- y_person[rand_val]

model_rand <- keras_model_sequential(input_shape = c(28, 28, 1)) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_flatten() |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 20, activation = "softmax")

model_rand |> compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

history_rand <- model_rand |> fit(
  x_tr_r, y_tr_r,
  epochs = 10,
  batch_size = 128,
  validation_data = list(x_va_r, y_va_r)
)


# Accuracy by digit 
pred  <- model |> predict(x_val)
pred_class <- max.col(pred) - 1L
meta_val  <- meta[val_rows, ]

acc_by_digit <- tapply(pred_class == y_val, meta_val$digit, mean)
acc_by_digit

# Evaluating the model
# test accuracy
model |> evaluate(x_test, y_test)

# Accuracy by digit on test set
pred_test <- model |> predict(x_test)
pred_class_test <- max.col(pred_test) - 1L
meta_test <- meta[test_rows, ]

acc_by_digit_test <- tapply(pred_class_test == y_test, meta_test$digit, mean)
acc_by_digit_test
