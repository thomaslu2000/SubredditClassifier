from os.path import isfile
import praw
# Get credentials from DEFAULT instance in praw.ini
r = praw.Reddit()
posts = r.subreddit('creepy').hot(limit=5)
print([p.title for p in posts])
