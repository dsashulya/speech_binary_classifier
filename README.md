# Бинарный классификатор речи (м/ж) на основе данных LibriTTS

### Данные 

Из-за ограниченных вычислительных возможностей решено было взять датасет [`dev-clean`](http://www.openslr.org/60/) и уже его впоследствии разделить на train/val/test.
Всего звуковых сэмплов 5736 и датасет почти идеально сбалансированный, поэтому дополнительной аугментации данных не понадобилось.

![plot](https://github.com/dsashulya/speech_binary_classifier/blob/main/plots/analysis/genders.png)

Так как датасет состоит из книжных предложений, записанных в студии, предполагается, что в сигналах нет лишних звуков и долгих пауз в начале, поэтому их можно укоротить до подходящей обучению длины без значительных потерь. Средняя длина сигнала в данных -- 5.6 секунд, однако в процессе экспериментов не было выявлено разницы в результатах при использовании сэмплов длины выше или ниже средней, так что для ускорения обучения были оставлены только 3 секунды начала каждой записи.

Посмотрим на мел-частотный спектр четырех случайных сэмплов каждого класса. Как и ожидалось, частоты мужского голоса сконцентрированы внизу графика, в то время как женский голос содержит заметно больше высоких гармоник.
![](https://github.com/dsashulya/speech_binary_classifier/blob/main/plots/analysis/melspecs1.png)

![](https://github.com/dsashulya/speech_binary_classifier/blob/main/plots/analysis/melspecs4.png)



Попробуем вычислить мел-кепстральные коэффициенты. Они уже не так легко интерпретируются, но различие в гармониках все равно заметное (вычислены по спектрам вторго ряда графиков выше):
![](https://github.com/dsashulya/speech_binary_classifier/blob/main/plots/analysis/mfcc4.png)

### Бейзлайн
Для бейзлайна была взята модель SVM, так как она эффективна в пространствах высокой размерности и работает даже когда размерность пространства выше числа векторов, как в нашем случае.

Модель была обучена несколько раз с разной предобработкой данных (везде от сигнала берутся первые 3сек):

Данные  | Рзамерность векторв | Точность на тесте
------------ | ------------- |  ------------- 
\|DFT\|          |       4000*     | 0.901
\|DFT\|          |       12000**     | 0.923
Mels 128     |       18048     | 0.976
MFCC 40         |      5640 | 0.962
**MFCC 128**         |      **18048**          | **0.978**


\* самые высокие пики спектра на графиках случайных сэмплов лежали не дальше k=4000

** половина частоты дискретизации, выше нее спектр отзеркален

DFT вычислялось по всему сигналу без разбиения на окна. Mels и MFCC вычисляли с помощью окна Ханна длины 2048 с шагом 512.


Запустить можно с помощью 
` python train.py --model svm --path_to_labels data/genders --svm_mode mfcc --n_mfcc 128 `

(необходимо, чтобы LibriTTS был скачан в папку `data`, иначе нужно установить `--path_to_data` и флаг `--download 1`). 

### CNN
`Conv2d(1, 32, kernel_size=(3, 3), stride=1)` -> `ReLU()` -> `MaxPool2d(kernel_size=2, stride=2)` ->


`nn.Conv2d(32, 16, kernel_size=(3, 3), stride=1)` -> `ReLU()` -> `MaxPool2d(kernel_size=2, stride=2)` ->


`Linear(15840, 1024)` -> `ReLU()` -> `Linear(1024, 2)`



Данные  | Точность на тесте
------------ |  ------------- 
MFCC 40 | 0.957
**MFCC 128** | **0.975**


Запустить финальную версию можно с помощью

` python train.py --model CNN --path_to_labels data/genders --log_every 10 --epochs 5 --trunc 72000 --mfcc 1 --n_mfcc 128 `

Обучение происходило на CPU,  время на одну эпоху около 6.5 минут. При увеличении числа эпох начиналось переобучение.

![](https://github.com/dsashulya/speech_binary_classifier/blob/main/plots/g5.000_lr0.001_mel128_trunc72000_epochs5SR_deepMFCC128.png)

### Выводы

По результатам точности SVM немного превзошла CNN на одинаковой предобработке данных, но отличие незначительное. В целом, можно было бы увеличить размер сети и использовать больший по размеру датасет `train-clean-100`, но, на мой взгляд, в этом нет смысла, так как 100-процентная точность недоступна даже человеческому уху, поэтому такая ситуация говорила бы о переобучении.

В дальнейшей работе я бы хотела попробовать другие модели, такие как RNN, а также построение эмбеддингов для звуковых сигналов.
