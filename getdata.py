import praw

r = praw.Reddit()

subs = ["worldnews", "technology", "gaming", "travel"]

for sub in subs:
    with open('%s.txt' % sub, 'w') as f:
        for submission in r.subreddit(sub).top('all', limit=50000):
            f.write("%s\n" % submission.title.encode('ascii', 'ignore').decode())
