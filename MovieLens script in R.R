##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Using R 4.2.2 version:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Making sure userId and movieId in validation set are also in edx set
#validation set 

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##############################################################


#Saving edx and validation files
save(edx, file="edx.RData")
save(validation, file = "validation.RData")

#Loading library 
library(tidyverse)
library(ggplot2)
library(dplyr)
library(markdown)
library(knitr)
library(caret)

# Data analysis and summary

# Exploring the data 
str(edx)
str(validation)

# Checking for NA value 
anyNA(edx)

# Nbr of movies and users in data set 
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# Looking for the most ratings
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  

# Nbr of rating per movies 
edx %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue")+
  scale_x_log10()+
  ggtitle("Rating Number Per Movie")+
  theme_gray()

# Nbr of rating per user
edx %>% count(userId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue")+
  ggtitle(" Number of Rating Per User")+
  scale_x_log10()+
  theme_gray()

# Nbr of rating by genres

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
  geom_bar(aes(fill =genres),stat = "identity")+ 
  labs(title = " Number of Rating for Each Genre")+
  theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))+
  theme_light()

# Partition the data 
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Excluding users and movies in the test set that do not appear in the training set:
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# RMSE calculation Function 

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

# 1rst model Calculating the mean of ratings from the training set 
Mu_1 <- mean(train_set$rating)
Mu_1

naive_rmse <- RMSE( Mu_1,test_set$rating)
naive_rmse

# RMSE results

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

# 2nd model 

# Adding the term  b_i to represent average ranking for movie i
#Y_{u,i} = \mu + b_i + \varepsilon_{u,i}

# Adjusting mean by movie effect 

Mu_2 <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - Mu_2))
movie_avgs

# Variability in the estimation 
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# Prediction improvements
predicted_ratings <- Mu_2 + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse))
rmse_results 

# 3rd model 
# Comparing user I with users rating over 100 movies 

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# Variability across users ratings  
 
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - Mu_2 - b_i))
user_avgs

# RMSE improvements with 3rd model

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = Mu_2 + b_i + b_u) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results

## RMSE of the validation set

valid_pred_rating <- validation %>%
  left_join(movie_avgs, by = "movieId" ) %>% 
  left_join(user_avgs , by = "userId") %>%
  mutate(pred = Mu_2 + b_i + b_u ) %>%
  pull(pred)

model_3_valid <- RMSE(validation$rating, valid_pred_rating)
model_3_valid
rmse_results <- bind_rows( rmse_results, 
                           data_frame(Method = "Validation Results" , RMSE = model_3_valid))
rmse_results
