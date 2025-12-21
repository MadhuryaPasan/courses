# read table
votes <- read.csv("votes.csv")

# view the table
View(votes)


votes[1, ]
votes$candidate[1]


sum(votes$poll[1],votes$poll[2],votes$poll[3])
sum(votes$poll)


votes$poll[1] + votes$mail[2]

votes$total <- votes$poll + votes$mail
View(votes)

write.csv(votes, "totals.csv" , row.names = FALSE)


colnames(votes)
rownames(votes)
