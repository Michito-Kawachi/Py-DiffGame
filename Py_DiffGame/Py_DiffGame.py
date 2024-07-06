"""
間違い探しゲーム
使った技術
「特徴点マッチング(AKAZE)、オプティカルフロー、（バウンディングボックス、領域検出）」

まず、間違い探しの1組の画像を読み込んで、特徴点マッチングを行う。
差分領域が白の二値化画像が得られる。
領域検出を行い、それぞれの領域を長方形で囲む（バウンディングボックス(以下BBox)）

＜工夫点＞
1つの間違い領域に対して、複数のBBoxができてしまった。
（1つにしないと同じ間違いに対して、何度も正解表示が出てしまう）
そこで、1つの領域に対して、BBoxを1つにしなければいけない。
→merge_bbox関数を作った
「この関数はBBoxのリストから、新たなBBoxのリストを作る関数だ。
    2つのBBoxの始点(BBoxの左上の頂点)が近いならば、2つを統合して1つのBBoxを新たに作る。
    加えて、BBoxの面積が定数以上なら、新たなBBoxのリストに加える。」
この関数を繰り返し実行すると1つの領域に、上塗りするようにたくさんのBBoxを生成できる
得たBBoxをzeros_likeで作った新しい画像に白で長方形を描画すると、
1領域につき、1つの長方形が描画される。力技である。
この新しい画像に対して、もう一度領域検出とBBox検出をすると、
間違い領域が1つのBBoxとして得られる。

次にカーソルの移動について説明する。
カメラ画像の中心の赤枠内で手などを動かすと、その動きに合わせて間違い探しの画像上のカーソルが移動する。
これはオプティカルフローを利用した。「5」キーを押すと、OFの情報が表示される。
赤枠内の移動量の平均を計算し、上下左右に動かしている。
間違いを全て見つければ、ゲームクリア。自動でプログラムが終了する。

操作方法
赤枠内で手などを動かして、カーソル（緑色の円）移動。
「g」キーで答え合わせをする。
'ESC': プログラム終了
'1': HSVフローの可視化の ON/OFF 切り替え
'2': グリッチエフェクトの ON/OFF 切り替え
'3': 空間伝播の ON/OFF 切り替え
'4': 時間伝播の ON/OFF 切り替え
"5": オプティカルフローの可視化のON/OFF 切り替え
"g": 答え合わせ
"0": デバッグモード切り替え
デバッグモードなら、("w": 上, "s": 下, "a": 左, "d": 右）にカーソル移動

検証していないが、画像を変えればどんな間違い探しでもゲーム可能だと思う。

参考資料
特徴点マッチング：https://qiita.com/grv2688/items/44f9e0ddd429afbb26a2
オプティカルフロー：https://www.kkaneko.jp/ai/opencv/video.html
間違い探し画像：https://学習プリント.com
"""
import os
import sys
import numpy as np
import cv2 as cv

def draw_flow(img, flow, step=16):
    """
    計算されたオプティカルフローを元に、図形を用いて視覚化した画像を生成する関数
    引数1 img: 入力画像
    引数2 flow: オプティカルフローのベクトル場
    引数3 step: 図形描画の間隔
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    """
    オプティカルフローをHSVで表現して画像を生成する関数
    引数 flow: オプティカルフローのベクトル場
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    """
    画像をオプティカルフローに基づいて変形させる関数
    引数1 img: 入力画像
    引数2 flow: オプティカルフロー
    """
    h, w = flow.shape[:2]
    flow = -flow
    # OFのx方向分だけ、列番号を加算する
    # 各列が対応する変位だけ、水平にシフトする
    flow[:,:,0] += np.arange(w)
    # OFのy方向分だけ、列番号を加算する
    # 各行が対応する変位だけ、垂直にシフトする
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    # cv.remap: 入力画像(img)を新しい座標(flow)に変換する関数
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def detect_move(img, flow, top_left, bottom_right) -> list:
    """
    中心座標の移動量を検出する関数
    閾値以上で上下左右に移動したと判断する
    引数1 flow: フローのベクトル場
    引数2 img: 入力画像
    引数3 top_left: 検出範囲の左上の座標(x, y)
    引数4 bottom_right: 検出範囲の右下の座標(x, y)
    戻り値 sum_move: 移動量の合計 [上下, 左右]
    上が負、下が正 右が正、左が負
    """
    # 特定の特徴点の移動量を抽出
    vector_rect = flow[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    movement_vector = np.mean(vector_rect, axis=(0,1))
    sum_move = [0, 0] # 移動量の合計を表すリスト[(上下), (左右)]

    # 移動量の閾値を設定
    threshold = 5.0
    # 上方向への移動が閾値以上の場合、画面を上にスクロール
    if movement_vector[1] > threshold:
        # print("↑")
        sum_move[0] = 1
    # 下方向への移動が閾値以上の場合、画面を下にスクロール
    elif movement_vector[1] < -threshold:
        # print("↓")
        sum_move[0] = -1
    # 右方向への移動が閾値以上の場合、画面を右にスクロール
    if movement_vector[0] > threshold:
        # print("→")
        sum_move[1] = 1
    # 左方向への移動が閾値以上の場合、画面を左にスクロール
    elif movement_vector[0] < -threshold:
        # print("←")
        sum_move[1] = -1
    return sum_move

def guess_diff(cursor: tuple, diff_dict: dict) -> list:
    """
    cursorの場所が間違いか判定する関数
    発見したらdiff_dictの該当要素を1にする
    引数1 cursor: カーソルの座標のタプル(x, y)
    引数2 diff_dict: 間違い範囲の辞書(x1,y1,x2,y2). 
    Falseなら未発見, Trueなら発見済み
    戻り値 diff_dict
    """
    flg = False
    found = False
    x, y = cursor
    for d in diff_dict:
        x1, y1, x2, y2 = d
        if x1 <= x <= x2 and y1 <= y <= y2:
            if not diff_dict[d]:
                flg = True
                diff_dict[d] = True
            else:
                found = True
    if flg:
        print("正解！")
    elif found:
        print("発見済み")
    else:
        print("残念")
        
    return diff_dict

def clear_check(diff_dict: dict):
    """
    すべての間違いを発見し、クリアしたかチェックする関数
    引数 diff_dict
    戻り値 endflg: True(クリア)/False(未クリア), cnt: 未発見数
    """
    endflg = True
    cnt = 0
    for v in diff_dict.values():
        if not v:
            endflg = False
            cnt += 1
    return endflg, cnt

def merge_bbox(bboxes, distance, area_threshold):
    """
    2つのBBoxの距離が近いとき、そのBBoxを統合する関数
    引数1 bboxes: BBoxのリスト
    引数2 distance: 距離の閾値
    引数3 area_threshold: 面積の閾値
    戻り値 new_bboxes: 統合したBBoxのリスト
    """
    # BBoxのスコアを計算
    new_bboxes = []
    merged = set()
    for i, a in enumerate(bboxes):
        if i in merged:
            continue
        for j, b in enumerate(bboxes):
            if j in merged:
                continue
            
            if i != j:
                # # 重心を求める
                # wxA, wyA = a[0]+(a[2]-a[0])//2, a[1]+(a[3]-a[1])//2
                # wxB, wyB = b[0]+(b[2]-b[0])//2, b[1]+(b[3]-b[1])//2
                # # print(wxA, wyA, wxB, wyB)
                # # 重心間の距離を求める
                # d = np.sqrt((wxA-wxB)**2 + (wyA-wyB)**2)
                # # print(distance)
                # if d < distance:

                # 開始x,yが近ければ統合
                if abs(a[0]-b[0]) < distance and abs(a[1]-b[1]) < distance:
                    c = [min(a[0], b[0]),
                         min(a[1], a[1]),
                         max(a[2], b[2]),
                         max(a[3], b[3])]
                    if c not in new_bboxes:
                        new_bboxes.append(c)
                        merged.add(i)
                        merged.add(j)
    flg = False
    for b in bboxes:
        x1, y1, x2, y2 = b
        area = abs(x2-x1) * abs(y2-y1)
        if area > area_threshold:
            for k in new_bboxes:
                kx1, ky1, kx2, ky2 = k
                if abs(x1-kx1) > 20 and abs(y1-ky1) > 20:
                    flg = True
        if flg:
            new_bboxes.append(b)
    return new_bboxes

"""オプティカルフローの初期設定"""
cam = cv.VideoCapture(0)
# cam = cv.VideoCapture("768x576.avi")
ret, prev = cam.read()

# 鏡を見ているように、USBカメラの入力画像を左右に反転する
ret = cv.flip(ret, 1)
prev = cv.flip(prev, 1)
cam_h, cam_w = prev.shape[:2]

prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
show_flow = False               # オプティカルフローの動きを表示する
show_hsv = False                # HSV空間でのOFを表示する
show_glitch = False             # ?
use_spatial_propagation = False
use_temporal_propagation = True # 時間伝播を有効にする(より滑らかに追跡できる)
cur_glitch = prev.copy()
inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst.setUseSpatialPropagation(use_spatial_propagation)
flow = None
"""ここまでオプティカルフロー"""

def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # プログラムがexeファイルなら
        cdir = sys._MEIPASS
    else:
        # プログラムが.pyなら
        cdir = os.path.dirname(__file__)
    return os.path.join(cdir, filename)


"""間違い探し（特徴量マッチング）"""
diffA = cv.imread(find_data_file("img1.png"))
diffB = cv.imread(find_data_file("img2.png"))
if diffA is None or diffB is None:
    print("画像読み込みに失敗")
else:
    hA, wA = diffA.shape[:2]
    hB, wB = diffB.shape[:2]
    
# 特徴点検出器を作成
akaze = cv.AKAZE_create()

# 特徴検出
kpA, desA = akaze.detectAndCompute(diffA, None)
kpB, desB = akaze.detectAndCompute(diffB, None)
# cv.imshow("diffA", cv.drawKeypoints(diffA, kpA, None))
# cv.imshow("diffB", cv.drawKeypoints(diffB, kpB, None))

# 参考：https://qiita.com/grv2688/items/44f9e0ddd429afbb26a2
#BFMatcher型のオブジェクト作成
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# 記述子をマッチさせる
matches = bf.match(desA, desB)
# マッチしたものを距離順に並べ替える
matches = sorted(matches, key=lambda x:x.distance)
# マッチしたもの（ソート済み）の中から上位15%をgoodとする
good = matches[:int(len(matches) * 0.15)]
# 対応がとれた特徴点の座標を取り出す
src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# findHomography: 2つの画像から得られた点の集合を与えると、その物体の投射変換を計算する
M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
# diffBを透視変換
# 透視変換: 斜めから撮影した画像を真上から見た画像に変換する感じ
diffB_transform = cv.warpPerspective(diffB, M, (wA, hA))
# # 表示
# cv.imshow("diffB", diffB)
# cv.imshow("trans_diffB", diffB_transform)

# diffAとdst_imgの差分を求めてresultとする。グレースケールに変換
result = cv.absdiff(diffA, diffB_transform)
# cv.imshow("result", result)
result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
# 二値化
_, result_bin = cv.threshold(result_gray, 50, 255, cv.THRESH_BINARY)
# 閾値は50
# cv.imshow("before opening", result_bin)
# ノイズ除去用カーネルを準備
kernel = np.ones((2,2), np.uint8)
# オープニング（収縮→膨張）実行
result_bin = cv.morphologyEx(result_bin, cv.MORPH_OPEN, kernel)
# cv.imshow("after opening", result_bin)
"""ここまで特徴量マッチング"""

"""Bounding Boxで間違いの範囲を特定"""
contours, _ = cv.findContours(result_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
bbox = diffA.copy()
bboxes = []
for cnt in contours:
    area = cv.contourArea(cnt)
    if area >= 1:
        bx, by, bw, bh = cv.boundingRect(cnt)
        if bw >= wA / 8 or bh >= hA / 8:
            continue
        if bw <= 3 or bh <= 3:
            continue
        if bx < 5 or wA-5 < bx:
            continue
        if by < 5 or hA-5 < by:
            continue
        # bbox = cv.rectangle(bbox, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)
        bboxes.append([bx, by, bx+bw, by+bh])
# print("領域の数:", len(bboxes))
# cv.imshow("bbox", bbox)
# print(bboxes)

# BBoxの距離が近いものを統合
for i in range(5):
    bboxes = merge_bbox(bboxes, 20, 0)
# print("new", len(bboxes), bboxes)

diff_img = np.zeros_like(result_bin)
for bb in bboxes:
    diff_img = cv.rectangle(diff_img, tuple(bb[:2]), tuple(bb[2:]), (255), -1)
# cv.imshow("new bbox", diff_img)

# 四角形の集合を改めて領域検出
contours, _ = cv.findContours(diff_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
diff_dict = dict() # 解答の範囲(x1,y1,x2,y2)の辞書
diff = np.zeros_like(diff_img)
for cnt in contours:
    area = cv.contourArea(cnt)
    if area >= 0:
        x, y, w, h = cv.boundingRect(cnt)
        diff_dict[(x,y,x+w,y+h)] = False
        diff = cv.rectangle(diff, (x,y),(x+w,y+h), (255), -1)
# cv.imshow("diff", diff) # diff -> 差分の二値化画像
diff_num = len(diff_dict)

"""ゲーム用初期設定"""
move_x, move_y = wA//2, hA//2
sum_move = [0, 0]
found_list = [] # 見つけた間違いの座標を保存するリスト
print("間違い探しゲームへようこそ！")
print(f"絵の中に間違いが{diff_num}個あるよ！探してみてね！")
print("操作方法",
      "カメラの映像の赤枠の中で手を動かそう。",
      "間違いの絵の緑の丸が、赤枠の領域の動きに合わせて動くよ。",
      "間違いを見つけたら、緑の丸を間違いに合わせて「g」を押そう。GUESS!!",
      sep="\n")

# DEBUG用フラグ
debug_flg = False
show_diff = False # 答え表示フラグ

while(True):
    ret, img = cam.read()
    # 鏡を見ているように、USBカメラの入力画像を左右に反転する
    ret = cv.flip(ret, 1)
    img = cv.flip(img, 1)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    sum_move = [0, 0] # 移動量
    
    # flowがNoneでない or 時間伝播(temporal propagation)が有効ならば
    if flow is not None and use_temporal_propagation:
        # 前回のフローを変形させ、現在のフローの初期近似を取得する
        flow = inst.calc(prevgray, gray, warp_flow(flow,flow))
    else:
        # フローを計算。ただし初期近似を使わない
        flow = inst.calc(prevgray, gray, None)
    prevgray = gray
    if show_flow:
        cv.imshow('flow', draw_flow(gray, flow))
    if show_hsv:
        cv.imshow('flow HSV', draw_hsv(flow))
    if show_glitch:
        cur_glitch = warp_flow(cur_glitch, flow)
        cv.imshow('glitch', cur_glitch)
    if show_diff:
        cv.imshow("diff", diff)
        
    """
    ユーザーの入力処理
    'ESC': プログラム終了
    '1': HSVフローの可視化の ON/OFF 切り替え
    '2': グリッチエフェクトの ON/OFF 切り替え
    '3': 空間伝播の ON/OFF 切り替え
    '4': 時間伝播の ON/OFF 切り替え
    "5": オプティカルフローの可視化のON/OFF 切り替え
    "0": デバッグモード切り替え
    wasd: デバッグモードなら、カーソル移動
    """
    ch = 0xFF & cv.waitKey(5)
    if ch == 27:
        # ESCキーでプログラム終了
        break
    if ch == ord('1'):
        # HSVフローの可視化 ON/OFF切替
        show_hsv = not show_hsv
        print('HSV flow visualization is', ['off', 'on'][show_hsv])
    if ch == ord('2'):
        # グリッチエフェクトのON/OFF切替
        show_glitch = not show_glitch
        if show_glitch:
            cur_glitch = img.copy()
        print('glitch is', ['off', 'on'][show_glitch])
    if ch == ord('3'):
        # 空間伝播のON/OFF切替
        use_spatial_propagation = not use_spatial_propagation
        inst.setUseSpatialPropagation(use_spatial_propagation)
        print('spatial propagation is', ['off', 'on'][use_spatial_propagation])
    if ch == ord('4'):
        # 時間伝播のON/OFF切替
        use_temporal_propagation = not use_temporal_propagation
        print('temporal propagation is', ['off', 'on'][use_temporal_propagation])
    if ch == ord("5"):
        show_flow = not show_flow
        print("オプティカルフローの表示を", ["オフ", "オン"][show_flow], "にしました")
    if ch == ord("c"):
        show_diff = not show_diff
        print("答え", ["非表示", "表示"][show_diff], "設定")
    if ch == ord("g"):
        # 正解判定処理
        diff_dict = guess_diff((move_x, move_y), diff_dict)
        clear, cnt = clear_check(diff_dict)
        if clear:
            print("ゲームクリア！おめでとう")
            cv.waitKey(10)
            break
        else:
            print(f"あと{cnt}個")
    if ch == ord("0"):
        debug_flg = not debug_flg
        print("デバッグモード", ["オフ", "オン"][debug_flg])
    # 以下デバッグモードのみ
    if ch == ord("w") and debug_flg:
        sum_move[0] = -1
    if ch == ord("s") and debug_flg:
        sum_move[0] = 1
    if ch == ord("d") and debug_flg:
        sum_move[1] = 1
    if ch == ord("a") and debug_flg:
        sum_move[1] = -1

    top_left = (cam_w//3, cam_h//3) # カーソル計算範囲 左上
    bottom_right = (cam_w - cam_w//3, cam_h - cam_h//3) # カーソル計算範囲 右上
    cursor = img.copy()
    # カーソル計算範囲の枠を描画
    cv.rectangle(cursor, top_left, bottom_right, (0, 0, 255), thickness=3)
    # 移動
    if not debug_flg:
        sum_move = detect_move(img, flow, top_left, bottom_right)   # 移動量検出
    speed = 5
    move_x += sum_move[1] * speed
    move_y += sum_move[0] * speed
    diffB_game = diffB.copy()
    cv.circle(diffB_game, (move_x, move_y), 8, (0, 255, 0), thickness=3)
    
    cv.imshow("camera", cursor)
    cv.imshow("diffA", diffA)
    cv.imshow("diffB", diffB_game)
    
cam.release()
cv.destroyAllWindows()