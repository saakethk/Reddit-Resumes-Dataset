# Reddit Resume Dataset
Creating a empirical dataset using [deepsift](https://github.com/saakethk/deepsift) for posts abouts resumes from Reddit.

## Purpose
I was attempting to create a resume and found a plethora of research concerning the topic. In order to make that information useful, I decided to create this formal dataset. My goals with this dataset are for it to:
1. Be used to develop resume evaluation models
2. Perform trend research
3. Test the deepsift library I developed

## Data Methods
Initial exploration and the process of finding the relevant subreddits I intend to explore was done using PRAW. I found that the r/resumes subreddit was a suitable and sufficiently sized (1,328,408 subscribers) subreddit for my specific use case.
