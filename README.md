# 間違い探しゲーム
## 環境  
- Python 3.9.7
- OpenCV-Python 4.10.0
- numpy 1.21.6
# 概要
- 特徴点マッチングで2つの画像の異なる点を取得する
- ユーザーがカメラの前で手や顔を動かすとその方向にカーソルが移動する
## 実装
まず、間違い探しの1組の画像を読み込んで、特徴点マッチングを行う。
差分領域が白の二値化画像が得られる。
領域検出を行い、それぞれの領域を長方形で囲む（Bounding Box(以下BBox)）
## 操作方法
- 赤枠内で手などを動かして、カーソル（緑色の円）移動
- 「g」キーで答え合わせをする
- 'ESC': プログラム終了
- '1': HSVフローの可視化の ON/OFF 切り替え
- '2': グリッチエフェクトの ON/OFF 切り替え
- '3': 空間伝播の ON/OFF 切り替え
- '4': 時間伝播の ON/OFF 切り替え
- '5': オプティカルフローの可視化のON/OFF 切り替え
- '0': デバッグモード切り替え
- デバッグモードなら、("w": 上, "s": 下, "a": 左, "d": 右）にカーソル移動
## 工夫
1つの間違い領域に対して、複数のBBoxができてしまった。1つにしないと同じ間違いに対して、何度も正解表示が出てしまう。
そこで1つの領域に対して、1つのBBoxになるよう調整する必要があった。  
BBoxの結合のためのmerge_bbox関数を作成した。  
- 2つのBBoxの始点(BBoxの左上の頂点)が近いならば、2つを統合して1つのBBoxを新たに作る。
- 加えて、BBoxの面積が定数以上なら、新たなBBoxのリストに加える。なお、定数は20である。  

この関数を繰り返し実行すると1つの領域に、上塗りするようにたくさんのBBoxを生成できる。  
得たBBoxをzeros_likeで作った新しい画像に白で長方形を描画し間違い領域が白の2値化画像が得られる。  
この新しい画像に対して、もう一度領域検出とBBox検出をすることで間違い領域を判別できるようにした。

次にカーソルの移動について説明する。  
カメラ画像の中心の赤枠内で手などを動かすと、その動きに合わせて間違い探しの画像上のカーソルが移動する。  
これはオプティカルフローを利用した。「5」キーを押すと、その情報が表示される。  
赤枠内の移動量の平均を計算し、上下左右に動かしている。  
間違いを全て見つければ、ゲームクリア。自動でプログラムが終了する。
## 参考資料
特徴点マッチング：https://qiita.com/grv2688/items/44f9e0ddd429afbb26a2
オプティカルフロー：https://www.kkaneko.jp/ai/opencv/video.html
## 制作期間
2023.12 ~ 2024.1
2024.7.6 公開
