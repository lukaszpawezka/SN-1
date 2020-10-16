import random
import numpy as np
from perceptron import  ThresholdType
import perceptron as pc
import adaline as ad

RANGE = 0.1
zero_data = [[0, 0], [0, 1], [1, 0]]
one_data = [[1, 1]]


def generate_data(size):
    data = []
    for i in range(int(size/2)):
        zero_pair = zero_data[random.randint(0, len(zero_data)-1)]
        one_pair = one_data[random.randint(0, len(one_data)-1)]
        data.append(
            (
                np.array([
                    zero_pair[0] + (random.random() / 2 - 0.5)*RANGE,
                    zero_pair[1] + (random.random() / 2 - 0.5)*RANGE
                ]), 0
            )
        )

        data.append(
            (
                np.array([
                    one_pair[0] + (random.random() / 2 - 0.5)*RANGE,
                    one_pair[1] + (random.random() / 2 - 0.5)*RANGE
                ]), 1
            )
        )
    return data


static_data = [
(np.array([0.96313538, -0.03528166]), 0),
(np.array([0.99130867, 0.98602614]), 1),
(np.array([-0.04261568,  0.99423909]), 0),
(np.array([0.9922048, 0.9705065]), 1),
(np.array([-0.01192121,  0.98349329]), 0),
(np.array([0.95509875, 0.96280429]), 1),
(np.array([-0.03981021,  0.95117225]), 0),
(np.array([0.97492809, 0.96292541]), 1),
(np.array([ 0.99332027, -0.04399098]), 0),
(np.array([0.99364536, 0.96902353]), 1),
(np.array([ 0.99295511, -0.04533087]), 0),
(np.array([0.97011338, 0.9717751 ]), 1),
(np.array([-0.00589889,  0.95196552]), 0),
(np.array([0.96139163, 0.97150964]), 1),
(np.array([ 0.95939419, -0.03424713]), 0),
(np.array([0.97479732, 0.97232073]), 1),
(np.array([-0.04905149, -0.021707  ]), 0),
(np.array([0.97516354, 0.98931272]), 1),
(np.array([-0.00626749, -0.02650152]), 0),
(np.array([0.95693469, 0.98096781]), 1),
(np.array([-0.02975399,  0.99952358]), 0),
(np.array([0.96978807, 0.99217659]), 1),
(np.array([ 0.96078697, -0.0115261 ]), 0),
(np.array([0.95965416, 0.98466507]), 1),
(np.array([-0.02648023,  0.99769458]), 0),
(np.array([0.95087716, 0.9574296 ]), 1),
(np.array([-0.0162751 ,  0.96709573]), 0),
(np.array([0.98386719, 0.97454855]), 1),
(np.array([ 0.9745999 , -0.00505274]), 0),
(np.array([0.95946496, 0.96602009]), 1),
(np.array([ 0.9630251 , -0.04136153]), 0),
(np.array([0.98886728, 0.96588787]), 1),
(np.array([-0.00185824, -0.02255423]), 0),
(np.array([0.96018355, 0.96619704]), 1),
(np.array([-0.00685565,  0.96974784]), 0),
(np.array([0.9532076 , 0.95683716]), 1),
(np.array([-0.02864085, -0.02383219]), 0),
(np.array([0.96601778, 0.9586753 ]), 1),
(np.array([-0.03324554, -0.04936801]), 0),
(np.array([0.98438368, 0.95855804]), 1),
(np.array([-0.01063652,  0.98321629]), 0),
(np.array([0.99951513, 0.99122587]), 1),
(np.array([-0.00734899, -0.04498751]), 0),
(np.array([0.99922561, 0.96933105]), 1),
(np.array([ 0.96658335, -0.03168792]), 0),
(np.array([0.99876012, 0.96446628]), 1),
(np.array([ 0.98706499, -0.01438472]), 0),
(np.array([0.96515975, 0.97514486]), 1),
(np.array([-0.04704471,  0.96188411]), 0),
(np.array([0.97274512, 0.96001121]), 1),
(np.array([-0.00654602,  0.97233759]), 0),
(np.array([0.96961681, 0.97320966]), 1),
(np.array([-0.04022264,  0.96844591]), 0),
(np.array([0.96027401, 0.96545474]), 1),
(np.array([-0.03655654,  0.98652566]), 0),
(np.array([0.98194821, 0.97864695]), 1),
(np.array([-0.00508699,  0.98391364]), 0),
(np.array([0.97716086, 0.96179107]), 1),
(np.array([ 0.96121419, -0.00150541]), 0),
(np.array([0.95434324, 0.9814311 ]), 1),
(np.array([-0.02855196,  0.9933836 ]), 0),
(np.array([0.96811188, 0.99438926]), 1),
(np.array([-0.03770933, -0.02166721]), 0),
(np.array([0.9799085 , 0.96309453]), 1),
(np.array([-0.01044341,  0.95795677]), 0),
(np.array([0.98513602, 0.96882601]), 1),
(np.array([ 0.9537667 , -0.02323603]), 0),
(np.array([0.95944471, 0.97969562]), 1),
(np.array([-0.03109759,  0.95753953]), 0),
(np.array([0.96883704, 0.97880499]), 1),
(np.array([ 0.9823566 , -0.03659456]), 0),
(np.array([0.98080864, 0.96050817]), 1),
(np.array([-0.01276077, -0.02187392]), 0),
(np.array([0.96567201, 0.98947962]), 1),
(np.array([ 0.99995055, -0.00327161]), 0),
(np.array([0.99510348, 0.98818026]), 1),
(np.array([ 0.99917914, -0.0270621 ]), 0),
(np.array([0.96514986, 0.98831226]), 1),
(np.array([ 0.96113509, -0.00415946]), 0),
(np.array([0.9500521 , 0.97849927]), 1),
(np.array([ 0.9621281 , -0.02607882]), 0),
(np.array([0.95988882, 0.95889485]), 1),
(np.array([-0.00734585, -0.00972159]), 0),
(np.array([0.97411165, 0.96960979]), 1),
(np.array([ 0.96745714, -0.00652369]), 0),
(np.array([0.96770483, 0.98896813]), 1),
(np.array([-0.03796893,  0.98225739]), 0),
(np.array([0.95534094, 0.99128967]), 1),
(np.array([-0.02918553, -0.0425179 ]), 0),
(np.array([0.9631214 , 0.96893145]), 1),
(np.array([ 0.97600771, -0.01068775]), 0),
(np.array([0.97979172, 0.96669319]), 1),
(np.array([-0.00534808, -0.01575031]), 0),
(np.array([0.98774045, 0.98525139]), 1),
(np.array([-0.01219611,  0.96032235]), 0),
(np.array([0.98711263, 0.96006199]), 1),
(np.array([ 0.96133616, -0.04299337]), 0),
(np.array([0.95372781, 0.97298327]), 1),
(np.array([-0.02372828, -0.02825626]), 0),
(np.array([0.95346957, 0.99226273]), 1)]


# for i in generate_data(10):
#     print(i)
# print(generate_data(2))


# perceptron = Perceptron(-1.5, np.array([1, 1]), ThresholdType.UNIPOLAR, 0.01)
# perceptron.test(generate_data(100))

#Perceptron

percepton = pc.Perceptron(
           -0.01,
           np.array([
               -0.001,
               0.001
            ]),
           ThresholdType.UNIPOLAR,
           0.4,
           False)

data = generate_data(100)
np.random.shuffle(data)

percepton.learn(static_data)
#percepton.test(generate_data(100))


# #Adaline
# percepton = ad.Adaline(
#            -0.5,
#            np.array([
#                -0.4,
#                0.4
#             ]),
#            0.000001,
#            True,
#            2)
# data = generate_data(1000)
# np.random.shuffle(data)
#
# percepton.learn(static_data)
# percepton.test(generate_data(100))