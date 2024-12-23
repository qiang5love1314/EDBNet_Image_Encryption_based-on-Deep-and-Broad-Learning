from EncryBroNet import getInputImage, getFractionalData, generate_chaos_sequence, hashValue, \
ConvNet, generateFeatureMatrix, updateFeatures
import numpy as np
import cv2
import torch
import time

np.random.seed(0)
torch.manual_seed(8)
batchsize =  10

if __name__ == '__main__':
    start_time = time.time()
    w = 256
    h = 256
    randomNumber = np.random.randint(0, 255)
    targetImage, width, heigh = getInputImage('results/5.png')   # just for get the random pixel, in real applications we don't put it in the at here
    decryImage = cv2.imread('results/en5.png', cv2.IMREAD_GRAYSCALE)
    # decryImage = np.expand_dims(decryImage, -1)
    # print(decryImage)
    tragetLabel = 0
    targetImage = cv2.resize(targetImage, [w, h])
    horizontal, vertical = (100+randomNumber) % w, (100+randomNumber) % h
    plaintextPixel = targetImage[horizontal, vertical]      # let chaotic value be different by input image
    
    initialKey = hashValue('898oaFs09f'.encode('utf8'))     # initial hash value which I set it
    initialKey.update(str(plaintextPixel).encode('utf8'))   # I put a pixel value here
    key_hash = initialKey.hexdigest()
    print('initialKey', key_hash)
    # generate two random chaos sequences using hash function and plaintext pixel
    initialNumber1 = getFractionalData(int(key_hash[1:5], 16) / (w * h))    # generate initial number from hash value for chaotic sequence
    initialNumber2 = getFractionalData(int(key_hash[6:10], 16) / (w * h))
    initialNumber3 = getFractionalData(int(key_hash[11:15], 16) / (w * h))
    chaotic_seq1 = generate_chaos_sequence([initialNumber1, initialNumber2, initialNumber3], w)
    chaotic_seq1 = np.round(np.abs(chaotic_seq1) * 10 **6) % w
    chaotic_seq2 = []
    initialLogistic = getFractionalData(chaotic_seq1[-1] / (w * h))
    for i in range(h):
        initialLogistic = 3.6789 * initialLogistic * (1 - initialLogistic)
        chaotic_seq2.append(initialLogistic)
    chaotic_seq2 = np.round(np.abs(chaotic_seq2) * 10 **6) % h

    # generate feature weight matrix of CNN and BLS
    myModel = ConvNet()
    myModel.load_state_dict(torch.load('ConvNet.pt'))
    fc1_weight = None
    fc2_weight = None
    for name, params in myModel.named_parameters():
        if "fc1" in name and "weight" in name:
            fc1_weight = params.data.numpy()      # (32, 49152)
        elif "fc2" in name and "weight" in name:
            fc2_weight = params.data.numpy()      # (2, 32)
            break
    fc1_feature = generateFeatureMatrix(fc1_weight, w, h)
    fc2_feature = generateFeatureMatrix(fc2_weight, w, h)
    Conv1FeaturesNetwork = updateFeatures(fc1_feature) % 256
    Conv2FeaturesNetwork = updateFeatures(fc2_feature) % 256
    T3 = np.loadtxt('T3.txt')
    BLSFeaturesNetwork = np.round(abs(generateFeatureMatrix(T3, w, h)) * 10 **5) % 256

    # inverse diffusion
    if tragetLabel == 0:
        scrambled_img = BLSFeaturesNetwork.astype('int') ^ Conv1FeaturesNetwork.astype('int') ^ decryImage.astype('int')
    else:
        scrambled_img = BLSFeaturesNetwork.astype('int') ^ Conv2FeaturesNetwork.astype('int') ^ decryImage.astype('int')
    cv2.imshow('scrambled_img', np.uint8(scrambled_img))
    # cv2.imwrite('decryScramble.png', (scrambled_img))

    # inverse scramble
    row_permutation = np.arange(w)
    col_permutation = np.arange(h)
    
    index1 = np.argsort(chaotic_seq1)
    inv_index1 = np.argsort(index1)
    row_permutation = row_permutation[inv_index1]

    index2 = np.argsort(chaotic_seq2)
    inv_index2 = np.argsort(index2)
    col_permutation = col_permutation[inv_index2]

    # inverse process, as well as original process, we should from fisrt-row to last-column
    for i in range(w):
        for j in range(h):
            scrambled_img[:, j] = np.roll(scrambled_img[:, j], index2[j])           # cyclic right shift
        scrambled_img[w-1-i, :] = np.roll(scrambled_img[w-1-i, :], -index1[w-1-i])  # cyclic left shift

    finalDecryImage = scrambled_img[:, col_permutation]
    finalDecryImage = finalDecryImage[row_permutation, :] 
    
    # cv2.imshow('finalDecryImage_img', np.uint8(finalDecryImage))
    # cv2.imwrite('finalDecryImage.png', (finalDecryImage))
    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间为：{:.4}".format(run_time), "秒")