
def binarySearch(inputArray):
	# The value I'm searching in between index left and right inclusive
	left = 0
	right = len(inputArray) - 1

	while(left <= right):
		mid = (left + right) / 2

		print(mid)


print(binarySearch([1, 2, 3]))