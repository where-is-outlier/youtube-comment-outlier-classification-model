library(openxlsx)
library(tidyverse)
library(readr)
library(stringr)
library(textclean)
library(tidytext)
library(RcppMeCab)
library(tidyr)
library(jsonlite)
setwd("C:/Users/hwwe1/Desktop/23-2")

df <- tibble(read.xlsx("df_sample_eda.xlsx"))

df %>% 
  filter(str_detect(tokenized, "VV|MAG|NNG|VA"))


# --- noraml_comments -----
noraml_comments <-df %>% 
  filter(class == 0) %>% 
  unnest_tokens(input = comment, 
                output = words,
                token = posParallel,
                drop = F) %>%
  count(words) %>%
  arrange(desc(n)) %>% 
  filter(str_detect(words, "vv|mag|nng|va"))

noraml_comments <- noraml_comments %>%
  mutate(word = gsub("[^가-힣]", "", words),
         word_length = str_length(word))

noraml_comments <- noraml_comments %>%
  filter(word_length > 1) %>% 
  arrange(-n)




# --- spam_comments -----
spam_comments <- df %>% 
  filter(class == 1) %>% 
  unnest_tokens(input = comment, 
                output = words,
                token = posParallel,
                drop = F) %>%
  count(words) %>%
  arrange(desc(n)) %>% 
  filter(str_detect(words, "vv|mag|nng|va"))

spam_comments <- spam_comments %>%
  mutate(word = gsub("[^가-힣]", "", words),
         word_length = str_length(word))
  
spam_comments <- spam_comments %>%
  filter(word_length > 1) %>% 
  arrange(-n)

spam_freq <- spam_comments %>% head(10)
write.xlsx(spam_freq, "spam_freq.xlsx")


# -------------
normal_freq = noraml_comments %>% head(10)
write.xlsx(normal_freq, "normal_freq.xlsx")

ggplot(data = normal_freq,
       aes(x = word,
           y = n)) +
  geom_col()
  





















