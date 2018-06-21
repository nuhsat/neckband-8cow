import cv2
import numpy as np
import glob
import os
import lvq

sapi_arr = []
codebooks = []

for sapi_dir_path in glob.glob("D:/Data Sapi/data/dataCROP/*"):
    h = 0
    if(sapi_dir_path.split("\\")[-1] == "Class_1"):
        sapi_label = 1
    if(sapi_dir_path.split("\\")[-1] == "Class_2"):
        sapi_label = 2
    if(sapi_dir_path.split("\\")[-1] == "Class_3"):
        sapi_label = 3
    if(sapi_dir_path.split("\\")[-1] == "Class_4"):
        sapi_label = 4
    if(sapi_dir_path.split("\\")[-1] == "Class_5"):
        sapi_label = 5
    if(sapi_dir_path.split("\\")[-1] == "Class_6"):
        sapi_label = 6
    if(sapi_dir_path.split("\\")[-1] == "Class_7"):
        sapi_label = 7
    if(sapi_dir_path.split("\\")[-1] == "Class_8"):
        sapi_label = 8

    #for every image in directory
    for image_path in glob.glob(os.path.join(sapi_dir_path, "*.jpg")):
        img = cv2.imread(image_path)

        #save datasets
        hist = cv2.calcHist([img],[0],None,[60],[16,256])
        prob = hist/sum(hist)
        stddev = np.std(hist)
        data = []
        for x in range(0,60):
            data.append(prob[x][0])
        data.append(sapi_label)
        sapi_arr.append(data)

        #save codebooks
        if(h == 0):
            h = 1
            codebooks.append(data)

print("done, buat csv")
#save data to csv
sapi_np =  np.array(sapi_arr)
codebooks = np.array(codebooks)
np.savetxt("data_sapi.csv", sapi_np, delimiter=',', fmt="%s")
print("save to csv success")

dataset = np.array(sapi_arr)
n_folds = 5
learn_rate = 0.001
n_epochs = 47
n_codebooks = 20

lvq.train_codebooks(dataset, codebooks, learn_rate, n_epochs)
# print(codebooks)
np.savetxt("codebooks2.csv", codebooks, delimiter=',', fmt="%s")

# cek akurasi data
con = np.zeros(shape=(8,8))

tot=0
bener=0
for i in range(0, len(dataset)):
    pred=int(lvq.predict(codebooks, dataset[i]))
    real=int(dataset[i, -1])
    print(pred, real)
    con[real-1, pred-1] += 1

for i in range(0, 8):
    for j in range(0, 8):
        if i==j:
            bener+=con[i,j]
        tot += con[i, j]

akurasi=bener/tot

print("========Confusion Matrix========")
print(con)
print()
print("Akurasi:",akurasi*100,"%")
