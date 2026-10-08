##Task 1: Exploration of the data

data <- readRDS("data/handwriting.rds")
dim(data$images)
head(data$metadata)
metadata <- data$metadata

#This gets the data ready for the neural network
library(keras3)
use_backend("tensorflow")
x <- data$images / 255 #makes copy of images with every pixel divided by 255 so 
#values run from 0 to 1. Neural networks learn better with smaller numbers
#takes writer ids and makes it 0 to 19 instead of 1 to 20, keras start counting from 0
y_person <- data$metadata$person_id - 1L
#x is what the model looks at (images) then y is the person (or our answer that we are trying to predict.)


table(metadata$person_id, metadata$session) #250 images per person per session

# Variation of the digits across the different writers --------------------


##In each plot, the digits, session and replicate are fixed
#Function that draws a single image
draw_image <- function(i){
  m <- data$images[i,,,1]
  image(t(m[28:1,]), col = gray.colors(256, start = 1, end = 0),
        axes = FALSE, asp = 1)
  
}

#All 20 writers for one digit
plot_writers <-  function(d,s = 1, r = 1){ #s = session, r = replicate
  plot_settings <- par(mfrow = c(4,5), mar = c(0.3, 0.3, 1.5, 0.3), oma = c(0, 0, 2.5, 0))
  on.exit(par(plot_settings))
  
  for (p in 1:20){
    i <- which(metadata$person_id == p & metadata$digit == d &
               metadata$session == s & metadata$replicate == r)
    draw_image(i)
    title(paste("Writer", p), cex.main = 0.9)
  }
  mtext(paste0("Digit ", d, "(session ", s, ", replicate ", r, ")"),
        outer = TRUE, font = 2)
}

##plots of the digits
for (d in 0:4) plot_writers(d)



# Variation of one digit across sessions, for a random sample of writers -----------------------------------------------


plot_sessions_sample <- function(d, n_people = 10, r = 1, seed = 1){
  # d = digit, r = replicate (fixed), seed = makes the sample reproducible
  set.seed(seed)
  people <- sort(sample(1:20, n_people))
  
  plot_settings <- par(mfrow = c(n_people, 5), mar = c(0.2, 0.2, 1.2, 0.2),
                       oma = c(0, 3.5, 2.5, 0), xpd = NA)
  on.exit(par(plot_settings))
  
  for (p in people){
    for (s in 1:5){
      i <- which(metadata$person_id == p & metadata$digit == d &
                   metadata$session == s & metadata$replicate == r)
      draw_image(i)
      if (p == people[1]) title(paste("Session", s), cex.main = 0.9)  # column labels on top row only
      if (s == 1) mtext(paste("Writer", p), side = 2, line = 0.5, las = 1, cex = 0.7)  # row labels on first column only
    }
  }
  mtext(paste0("Digit ", d, " across sessions (replicate ", r, ")"),
        outer = TRUE, font = 2)
}

for (d in 0:4) plot_sessions_sample(d)


# Variation across replicates (same writer, digit and session) ------------

plot_replicates <- function(d, s = 1, n_people = 10, reps = c(1, 11, 21, 31, 41), seed = 1){
  # d = digit, s = session (fixed), reps = which replicates to show
  set.seed(seed)
  people <- sort(sample(1:20, n_people))
  
  plot_settings <- par(mfrow = c(n_people, length(reps)), mar = c(0.2, 0.2, 1.2, 0.2),
                       oma = c(0, 3.5, 2.5, 0), xpd = NA)
  on.exit(par(plot_settings))
  
  for (p in people){
    for (r in reps){
      i <- which(metadata$person_id == p & metadata$digit == d &
                   metadata$session == s & metadata$replicate == r)
      draw_image(i)
      if (p == people[1]) title(paste("Replicate", r), cex.main = 0.9)
      if (r == reps[1]) mtext(paste("Writer", p), side = 2, line = 0.5, las = 1, cex = 0.7)
    }
  }
  mtext(paste0("Digit ", d, " across replicates (session ", s, ")"),
        outer = TRUE, font = 2)
}

for (d in 0:4) plot_replicates(d)


# Quantifying within-session, across-session and between-writer --------

# Flatten each 28x28 image into one row of 784 pixel values (scaled 0-1)
x_flat <- matrix(data$images / 255, nrow = dim(data$images)[1])

# Find the row number of one specific image
get_i <- function(p, d, s, r){
  which(metadata$person_id == p & metadata$digit == d &
          metadata$session == s & metadata$replicate == r)
}

# Euclidean distance between two images (given their row numbers)
dist_img <- function(a, b) sqrt(sum((x_flat[a, ] - x_flat[b, ])^2))

set.seed(1)
n <- 1000
results <- data.frame(type = character(), digit = integer(), dist = numeric())

for (k in 1:n){
  p <- sample(1:20, 1)                  # a random writer
  d <- sample(0:4, 1)                   # a random digit
  s <- sample(1:5, 2)                   # two different sessions
  r <- sample(1:50, 2)                  # two different replicates
  q <- sample(setdiff(1:20, p), 1)      # a different writer
  
  a <- get_i(p, d, s[1], r[1])          # the reference image
  
  results[nrow(results) + 1, ] <- list("Within-session", d, dist_img(a, get_i(p, d, s[1], r[2])))
  results[nrow(results) + 1, ] <- list("Across-session", d, dist_img(a, get_i(p, d, s[2], r[2])))
  results[nrow(results) + 1, ] <- list("Between-writer", d, dist_img(a, get_i(q, d, s[1], r[2])))
}

results$type <- factor(results$type,
                       levels = c("Within-session", "Across-session", "Between-writer"))

# Numbers
aggregate(dist ~ type, data = results, FUN = function(v) c(mean = mean(v), sd = sd(v)))

# Picture
boxplot(dist ~ type, data = results, ylab = "Pixel distance",
        main = "How different are two images of the same digit?")

# Mean distance for each digit (rows) and pair type (columns)
tab <- with(results, tapply(dist, list(Digit = digit, Type = type), mean))

# Round, and add a column showing how much bigger across-session is than within-session
tab <- as.data.frame(round(tab, 2))
tab$`Across / within` <- round(tab$`Across-session` / tab$`Within-session`, 1)
tab

