# GRU-Autoencoder

https://qiita.com/satolab/items/c4e9a287e8d1ebb010a6

Powered by [satolab](https://qiita.com/satolab)

## Overview

時系列モデルであるGRUと，encoder-decoderモデルを組み合わせた，動画再構成モデルです．
ここではこのモデルを，GRU-AEと呼びます．

This is a video reconstruction model that combines GRU (a time-series model) and encoder-decoder model.
Here, we call this model GRU-AE.

## Model

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/c5ddc1c8-f82d-b6ff-8cca-b9cc7a2787f8.png" width="400×200">

## Results
- 5,000 itr(input,output)

![real_itr5000_no0.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/f2e35deb-0162-e52f-ec2a-cdbf2ce64d77.png)
![recon_itr5000_no0.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/38ce5478-dbe1-0673-d9d3-2d65ae755d5b.png)

## Usage
- datasetのダウンロード
Download dataset
http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
- lib.pyのParseGRU()内の初期化メソッド，dataset変数に，
上記のdatasetが格納されたdirを指定してください
(動画ファイルのままで問題ございません)
Please specify the dir in which the above dataset is stored.
(No problem as a video file.)


- gru_ae.pyで学習．logs/generated_videosにサンプルが保存されます．
Learn with gru_ae.py.
The sample is saved in logs/generated_videos.

## 参考文献 References
動画ファイルのロード部分
https://github.com/DLHacks/mocogan
