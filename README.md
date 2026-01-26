# 紙鈔辨識

- 將準備好的 ‘圖像資料集.zip’ 檔案放至程式檔案目錄並透過 ‘unzip’ 指令解壓縮，匯入相關套件並載入資料集(注意，要使用 unzip 指令要確定程式目錄底下有 unzip.exe 檔才可執行(windows需要另外下載，mac跟linux有內建))，或是匯入 python 的 zipfile 套件來解壓縮也可以
- 使用 glob 套件擷取鈔票檔名作為 label 標籤資料後須留意字典格式在不同作業系統會有字典中 value 排序的問題
- 輸入的圖像資料即使盡量調整大小仍然可能會有尺寸差異，造成模型輸入資料維度的不一致，因此需要先將圖像資料做 resize 的預處理後再輸入模型，這裡使用 imgaug 套件放在資料增補步驟一併處理
- 使用 imgaug 套件進行資料增補，但需要篩選要增補的效果種類(例如：旋轉、平移、對比、縮放)以及數量，否則在訓練時會消耗大量記憶體空間
- 將增補好的資料進行特徵縮放讓每個像素介於 [ 0 , 1 ] 之間， 使用 skimage 套件會自動將每張圖像的像素值進行縮放至 [ 0 , 1 ] 之間 (在此使用 imageio 套件，沒有自動進行特徵縮放，因此需要額外做特徵縮放)
- 每一種圖像資料增補200筆後也產生對應的 label 標籤值
- 建立 CNN 圖像辨識模型
- 因為一次讀取800筆資料進行訓練會使記憶體使用率佔滿不夠用，因此改成使用 DataSet 套件將訓練資料分成小批次讀取並放入模型進行訓練
- 資料訓練跟讀取不衝突，因此快取 + prefetch 預先提取的方式(在訓練當前一批資料時，預先讀取下一批資料)進行訓練
- 清除原本的資料增補集合(節省記憶體使用率)
- 設定模型訓練超參數，使用 Dataset 的方式來訓練模型無法在 model.fit() 函數內使用 validation_split 參數來切割驗證資料，但是可以事先再產生另外一筆 Dataset，再透過fit()函數裡的 validation_data 參數來指定驗證用的 Dataset
    
    [優化電腦 提升效能 進階教學，虛擬記憶體如何設置](https://www.youtube.com/watch?v=kp6BlU5JvDk)
    
- 訓練調校 ( 減少記憶體使用率 )：
    1. 輸入圖像大小
    2. 模型簡化
    3. 使用 Dataset
    4. 調整批量大小
    5. 訓練執行週期數
- 可改善的方向：
    1. 可增加原始圖像資料的張數，或拍攝一些實際情況可能發生的皺褶、歪斜圖像用以加入訓練，來提高面對實際環境情況可能有差異的準確率
    2. 增加 CNN 模型的層數、選用不同的 kernel ( filter ) 來做卷積運算，或是調整 CNN 模型訓練的超參數 ( 例如：padding ( 邊緣補0 )、strides ( filter的滑動步長 ) )，以提高模型擷取特徵的能力
- 套件修改區
    
    ```jsx
    # imguag 套件發生 **AttributeError**: `np.sctypes` was removed in the NumPy 2.0 release. Access dtypes explicitly instead.
    # 需要將 imgaug.py 檔案內第39行的後3行
    NP_FLOAT_TYPES = set(np.sctypes["float"])
    NP_INT_TYPES = set(np.sctypes["int"])
    NP_UINT_TYPES = set(np.sctypes["uint"])
    # 修改成：
    NP_FLOAT_TYPES = {np.float16, np.float32, np.float64}
    NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
    NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}
    # 參考:https://github.com/aleju/imgaug/issues/859
    ```