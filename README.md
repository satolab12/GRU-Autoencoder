# GRU-Autoencoder

https://qiita.com/satolab/items/c4e9a287e8d1ebb010a6

## 概要

時系列モデルであるGRUと，encoder-decoderモデルを組み合わせた，動画再構成モデルです．
ここではこのモデルを，GRU-AEと呼びます．

## 動画再構成モデル

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/c5ddc1c8-f82d-b6ff-8cca-b9cc7a2787f8.png" width="400×200">

## 結果
- 5,000 itr(input,output)

![real_itr5000_no0.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/f2e35deb-0162-e52f-ec2a-cdbf2ce64d77.png)
![recon_itr5000_no0.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/583727/38ce5478-dbe1-0673-d9d3-2d65ae755d5b.png)

## Usage
- datasetのダウンロード
http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
- lib.pyのParseGRU()内の初期化メソッド，dataset変数に，
上記のdatasetが格納されたdirを指定
（動画ファイルのままで問題ございません）
- gru_ae.pyで学習．logs/generated_videosにサンプルが保存されます．

## 参考文献
動画ファイルのロード部分
https://github.com/DLHacks/mocogan

