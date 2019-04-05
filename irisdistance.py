from math import sqrt
import numpy as np
import warnings
from collections import Counter, defaultdict
import pandas as pd
import random




def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups! idiot.')
	#retrieve the distances from the predicted value to each entry in the train set
	distances = []
	for group in data:
		for features in data[group]:
			#find the distance from the train set point and store
			euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
			distances.append([euclidean_distance, group])
	vote_result = most_common(distances, k)
	return vote_result

def most_common(distances, k):
	votes = sorted(distances)[:k]
	#store counts into default dict and find class with most votes
	vote_count = defaultdict(int)
	for dist, pred in votes:
		vote_count[pred] += 1
	return sorted([p for p, d in vote_count.items()], key=lambda x: x[1])[0]



def getTrain_Test_Set(df):
	full_data = df.values.tolist()
	#divide train set and test set
	random.shuffle(full_data)
	test_size = 0.2
	train_set = defaultdict(list)
	test_set = defaultdict(list)
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]
	#create model train set and test set
	train_set = defaultdict(list)
	test_set = defaultdict(list)
	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])
	return train_set, test_set

def test_data(train_set, test_set):
	correct = 0
	total = 0
	#predict a value from algorithm using test set, and check for accuracy
	for group in test_set:
		for data in test_set[group]:
			vote = k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct += 1
			total += 1
	return correct, total

def main():
	df = pd.read_csv("iris.data.txt")
	df.replace('?', -99999, inplace=True)
	train_set, test_set = getTrain_Test_Set(df)
	correct, total = test_data(train_set, test_set)
	print('Accuracy:', correct / total)

main()




