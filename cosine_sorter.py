# you must import operator for this to work
# x is a dictionary whose key is the number of the document and value the TF-IDF score (cosine)
sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)[:3] # Gets the sorted list of the dictionary x, with higher values first and picks only the 3 highest values

sorted_x = sorted(sorted_x, key=operator.itemgetter(0)) # Gets the sorted list by document number