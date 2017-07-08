# 画像認識用 Deep Convolution Neural Network

## 実行環境
`python3`の環境で実行できます。  
実行に必要なモジュール(動作確認バージョン)は以下の通り。

- chainer (v2.0)
- cupy (1.0.0.1)
- matplotlib (2.0.2)

## Training

`src/train.py`はCNNのパラメータの学習処理を実行します。  
実行結果は`result/`に保存されます。

実行方法は以下の通り。
```
% python src/train.py DATASET NETWORK PARAMETERS [OPTIONS]
```

- DATASET : 使用する画像のデータセットを指定します。  
(mnist, cifar10 or cifar100)
- NETWORK : 実行するCNNの名前を指定します。  
(resnet, pyramid, densenet, resnext)
- PARAMETERS : CNNに渡すパラメータを指定します(後述)。

また、次のオプションを指定できます。

- `-l NAME` : 学習係数の変更方法を指定します(default: step)。  
step : 学習回数1/2と3/4のときに、学習係数を0.1倍します。  
cosine : Cosine Annealing without Restart [[1](#ref1)]  
restart : Cosine Annealing with Restart [[1](#ref1)]  
- `-r RATE` : 学習係数の初期値を指定します(default: 0.1)。
- `-m MOMEMTUM` : モーメンタムの値を指定します(default: 0.9)。
- `-d DECAY` : Weight Decayの値を指定します(default: 0.0001)。
- `-e EPOCH` : 学習回数を指定します(default: 300)。
- `-b BATCH` : バッチサイズを指定します(default: 128)。
- `-p SIZE` : 1回の計算に使う画像の数を指定します(default: 128)。
- `-g GPU` : 使用するGPUのIDを指定します(default: -1)。
- `--no-check` : 入力される行列の大きさのチェックを省略します。

次のCNNとパラメータを指定できます。  
ただし、論文中のオリジナルとは少し異なります(詳細はQiitaの記事)。

- ResNet [[2](#ref2),[3](#ref3)] (ResBlochはSingle RELUです [[4](#ref4)])  
`python src/train.py DATASET resnet DEPTH WIDTH`
    - `DEPTH` : CNNの層数
    - `WIDTH` : 畳み込み層のチャンネル数(論文中のkの値)
- Pyramidal ResNet [[4](#ref4)]  
`python src/train.py DATASET pyramid DEPTH ALPHA`
    - `DEPTH` : CNNの層数
    - `ALPHA` : 論文中のαの値
- DenseNet [[5](#ref5)]  
`python src/train.py DATASET densenet DEPTH WIDTH`
    - `DEPTH` : CNNの層数
    - `WIDTH` : 畳み込み層のチャンネル数(論文中のkの値)
- ResNeXt [[6](#ref6)]  
`python src/train.py DATASET resnext DEPTH WIDTH UNIT`
    - `DEPTH` : CNNの層数
    - `WIDTH` : 畳み込み層のチャンネル数(論文中のdの値)
    - `UNIT` : 畳み込み層のユニット数(論文中のCの値)
- ShakeNet [[7](#ref7)]  
`python src/train.py DATASET shakenet DEPTH WIDTH`
    - `DEPTH` : CNNの層数
    - `WIDTH` : 畳み込み層のチャンネル数(論文中のdの値)

## Report

`src/report.py`は`result/`に保存された実行結果からグラフを生成します。

実行方法は以下の通り。
```
% python src/result.py
```

`src/report.py`は設定ファイル`result/meta.txt`を生成します。  
このファイルを編集することにより、グラフに含めるデータや名前を変更できます。
設定ファイルのフォーマットは以下の通りです。
```
DIRECTORY_1: FLAG_1/NAME_1
DIRECTORY_2: FLAG_2/NAME_2
DIRECTORY_3: FLAG_3/NAME_3
...
```
- DIRECTORY : データが保存されているディレクトリ名です
- FLAG : グラフに含めるならTrueを指定します
- NAME : グラフの凡例に表示する名前です

## References

1. <a name="ref1"></a> [Loshchilov, Ilya, and Frank Hutter. "Sgdr: Stochastic gradient descent with warm restarts." (2016).](https://arxiv.org/abs/1608.03983)
2. <a name="ref2"></a> [He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. (2016).](https://arxiv.org/abs/1512.03385)
3. <a name="ref3"></a> [Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016).](https://arxiv.org/abs/1605.07146)
4. <a name="ref4"></a> [Han, Dongyoon, Jiwhan Kim, and Junmo Kim. "Deep pyramidal residual networks." arXiv preprint arXiv:1610.02915 (2016).](https://arxiv.org/abs/1610.02915)
5. <a name="ref5"></a> [Huang, Gao, et al. "Densely connected convolutional networks." arXiv preprint arXiv:1608.06993 (2016).](https://arxiv.org/abs/1608.06993)
6. <a name="ref6"></a> [Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." arXiv preprint arXiv:1611.05431 (2016).](https://arxiv.org/abs/1611.05431)
7. <a name="ref7"></a> [Gastaldi, Xavier. "Shake-Shake regularization." arXiv preprint arXiv:1705.07485 (2017).](https://arxiv.org/abs/1705.07485)
