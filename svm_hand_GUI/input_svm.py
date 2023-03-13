from joblib import dump, load

svm = load('svm_hand.joblib')
a = ['arry']


print(a[0])
# a[0] = a[0:-2]
print(a[0][0:-1])
print(svm.predict([a[0][0:-1]]))
