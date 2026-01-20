#### Basic ####
#name <- readline("What's your name?")
#greating <- paste0("Hello,", name)
#print(greating) 

# function inside a function
#print(paste("Hello,",readline("what's your name?")))


#### arithmatic #### 

#x <- as.integer(readline("Enter number 1: "))
#y <- as.integer(readline("Enter number 2: "))

#total <- x + y

#print(paste("Total votes: ", total))

#### SUM ####
#x <- as.integer(readline("Enter number 1: "))
#y <- as.integer(readline("Enter number 2: "))

#total <- sum(x,y)

#print(paste("Total votes: ", total))


#### Read Tables ####
votes <- read.csv("votes.csv")

View(votes)

# access rows and collumns
# [row, col]
votes[1,]
votes[,1]

# get the column
votes$candidate
votes$candidate[1]


#sum of a column
sum(votes$poll[1],votes$poll[2],votes$poll[3])
#or
sum(votes$poll)


# sum of row values
votes$poll[1]+votes$mail[1]
# or 
votes$poll + votes$mail

#### add new column with previous values ####
votes$total <- votes$poll + votes$mail
View(votes)

#save the scv file with new column
write.csv(votes, "totals.csv", row.names = FALSE)


#### access the column names and row names

colnames(votes)
rownames(votes)
