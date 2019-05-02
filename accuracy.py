final_output = []
f = open("accuracy.txt", "r")
for x in f:
	if x != "\n":
		a = x.split('/')
		final_output.append(a[1])

count = 0
total_count = 0

for i in range(0,len(final_output)//2):
	if final_output[2*i]==final_output[2*i+1]:
		count += 1
	else:
		print("wrong set")
		print(final_output[2*i])
		print(final_output[2*i+1])
	total_count += 1

print("Accuracy : ")
print((count/total_count)*100)
print(count)
print(total_count)