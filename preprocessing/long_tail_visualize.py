from matplotlib import pyplot as plt

from config import DATASET_DIR

file_data = open(DATASET_DIR)
# next(file_data) # skip header

reviewers = set([])
items = set([])
n_rating_item = {}
rating_count = 0

for d in file_data:
    reviewer, item, rating, timestamp = d.split(',')

    if n_rating_item.get(item):
        n_rating_item[item] += 1
    else:
        n_rating_item[item] = 1

    reviewers.add(reviewer)
    items.add(item)
    rating_count += 1

n_user = len(reviewers)
n_item = len(items)
density = rating_count / (n_user * n_item)
print("Total Users {}".format(n_user))
print("Total Items {}".format(n_item))
print("Total Ratings {}".format(rating_count))
print("Density {}".format(density))

# Total Reviewers 909314
# Total Items 130005
# metadata (134,838 products)

ratings = sorted(n_rating_item.values(), reverse=True)
print(ratings[0])
print(max(ratings))
plt.axvline(x=len(items) * 0.2, color='red')
plt.axvline(x=len(items) * 0.8, color='red')
plt.axis([-10,len(items), 0, ratings[0]])  # x-start, x-end, y-start, y-end
plt.scatter(range(1, len(items) + 1), ratings, s=1)
plt.show()
