from random import *
import random
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K

def Pitch_Adjusting(new_harmony):
    filter_size = randint(1, 5)
    par_idxselect = random.randint(0, lim_h-1)
    if par_idxselect == 0:
        new_harmony[0] = filter_size
    elif par_idxselect == 1:
        new_harmony[1] = random.choice(act_function)
    elif par_idxselect == 2:
        new_harmony[2] = random.choice(opt)
    elif par_idxselect == 3:
        new_harmony[3] = epoch
    else:
        new_harmony[par_idxselect] = random.choice(num_filter)
    return new_harmony

# [필터사이즈,활성화함수,옵티마이저,에폭,레이어수(5개)]
def CNN(HMlist):
    base_dir = 'C:/Users/WBH/Desktop/배진희'
    train_dir = base_dir + '/' + 'EX_LANDMARK/train'
    test_dir = base_dir + '/' + 'EX_LANDMARK/test'

    train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_generator = ImageDataGenerator(rescale=1. / 255)
    train_set = train_generator.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')
    test_set = test_generator.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='categorical')

    # automl: 파라미터가 알고리즘에 의해 구조 생성
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation=HMlist[1]))

    print(HMlist)

    for i in HMlist[4:-1]:
        if i != 0:
            model.add(Conv2D(i, (HMlist[0], HMlist[0]), activation=HMlist[1]))
            model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=HMlist[2], loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_set, steps_per_epoch=100, epochs=HMlist[3], validation_data=test_set, validation_steps=5)
    scores = model.evaluate_generator(test_set, steps=5)
    acc = round(scores[1] * 100, 1)

    K.clear_session()

    return acc

HM=[]
HMS = 30
lim_h = 7
hmcr = 0.9
par = 0.1
itr = 100

num_filter = [0, 16, 32, 64, 128]
act_function = ['sigmoid', 'relu', 'tanh']
opt = ['sgd', 'adagrad', 'adadelta','rmsprop', 'adam']

for j in range(HMS):  # 초기화[필터사이즈,활성화함수,옵티마이저,에폭,레이어수(5)]
    h_v = []
    size_filter = randint(1, 5)
    epoch = randint(10, 100)

    h_v.append(size_filter) #필터 사이즈
    h_v.append(random.choice(act_function)) #활성화 함수
    h_v.append(random.choice(opt)) #옵티마이저
    h_v.append(epoch) #에폭

    for x in range(lim_h - 4):
        h_v.append(random.choice(num_filter))
    h_v.append(0)
    HM.append(h_v)

    HM[j][-1] = CNN(HM[j])

for iteration in range(itr):
    new_harmony = []
    if random.uniform(0,1) < hmcr: # 열별로 랜덤으로 고르기
        for i in range(lim_h):
            temp = []
            for j in range(HMS):
                temp.append(HM[j][i])
            rnd = random.choice(temp)
            new_harmony.append(rnd)
        if random.uniform(0, 1) < par:
            new_harmony = Pitch_Adjusting(new_harmony)
    else:
        size_filter = randint(1, 10)
        epoch = randint(10, 100)

        new_harmony.append(size_filter)
        new_harmony.append(random.choice(act_function))
        new_harmony.append(random.choice(opt))
        new_harmony.append(epoch)
        for j in range(lim_h - 4):
            new_harmony.append(random.choice(num_filter))

    new_harmony.append(CNN(new_harmony))
    HM.append(new_harmony) # 추가

    HM.sort(key=lambda x: x[-1], reverse=True) # 정렬
    HM.pop() # 제거

    f = open('ictc2021_landmark.txt', 'a')
    f.write('---------------itr '+ str(iteration+1) + '---------------' + '\n')
    f.write("best harmony >> "+ str(HM[0][:-1])+'\n')
    f.write("filter size = " + str(HM[0][0])+'\n')
    f.write("activation function = " + str(HM[0][1])+'\n')
    f.write("optimizer = " + str(HM[0][2])+'\n')
    f.write("epochs = " + str(HM[0][3])+'\n')
    f.write("numbers of layer = " + str(HM[0][4])+'\n')
    f.write("numbers of layer = " + str(HM[0][5])+'\n')
    f.write("numbers of layer = " + str(HM[0][6])+'\n')

    f.write("acc >> " + str(HM[0][-1]) + '\n')
    f.close()
    print("------------------ itr " + str(iteration) + " ------------------")
    print("best harmony >> " + str(HM[0][:-1]))
    print("acc >> " + str(HM[0][-1]))
    print("-------------------------------------------")