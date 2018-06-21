import numpy as np
import praproses_citra
import cv2
from math import sqrt

sapi_test_arr = []

def prediksi(codebooks, datanew):
    jarak = []
    for i in range(0,8):
        dist = 0.0
        for j in range(0,4):
            dist += (codebooks[i][j] - datanew[0][j])**2
        jarak.append((codebooks[i], sqrt(dist)))
    jarak.sort(key=lambda tup: tup[1])
    return jarak[0][0]

def main(filename):
    print("Silakan tunggu, sedang mengolah citra...")
    img = cv2.imread(filename, 0)
    hasil_crop = praproses_citra.proses(img)
    hist2 = cv2.calcHist([hasil_crop],[0],None,[16],[16,256])
    # hist = cv2.calcHist([img],[0],None,[1],[16,256])
    # plt.hist(img.ravel(), 16, [16,256]); plt.show()
    prob = hist2/(sum(hist2))
    stddev = np.std(hist2)
    print("Pengolahan citra berhasil. Melanjutkan ke proses klasifikasi.....")
    data = []
    for x in range(0,60):
        data.append(prob[x][0])
    sapi_test_arr.append(data)

    testdata = np.array(sapi_test_arr)
    codebooks = np.genfromtxt('codebooks2.csv', delimiter=',')

    #evaluate class
    dekat = prediksi(codebooks, testdata)
    hasil = dekat[-1]
    print("Klasifikasi berhasil")
    print("Gambar tersebut masuk ke kelas "+ str(hasil))

print("Masukkan nama file image, pastikan satu folder dengan program ini")
filename = input()
main(filename)
