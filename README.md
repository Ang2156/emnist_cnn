# EMNIST 英數字手寫資料集辨識

- 載入 EMNIST 資料
    - EMNIST 英數字手寫資料集辨識 的官方資料集 emnist.zip 有誤，需要將 ‘C:\Users\Administrator\.cache\emnist’ 路徑裡的 emnist.zip 用課堂給的同名檔案附蓋掉(Administrator為當前系統管理員名稱)
    - 使用 EMNIST 中：數字 0 ~ 9 + 英文字母大小寫，總共 62 個種類的 ‘By_Class’ 資料集
        
        ```jsx
        # 可根據需求使用不同種類的 EMNIST 資料集 ( 透過更換''中的詞彙 )
        images_train, labels_train = extract_training_samples('byclass')  # 使用'By_Class'資料集  
        ```
        
    - ‘By_Class’ 資料集 的 label 標籤編號不用調整 ( 有些種類的編號從1開始，例如：’Letters’ 資料集，需要把 label 標籤編號全部 - 1 )
- 特徵工程**，將特徵縮放成(0, 1)之間**
    - 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
    - 顏色範圍：0~255，所以，公式簡化為 x / 255
    - 注意，顏色0為白色，與RGB顏色不同，(0,0,0) 為黑色。
- 建立模型 & 訓練
    - 建立 CNN 圖像辨識模型
        
        ```jsx
        # CNN 圖像辨識模型，注意模型最後一層的神經元個數要與種類數一致
        model = tf.keras.models.Sequential([
        		layers.Input((28, 28, 1)),
        		layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
        		layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'),
        		layers.MaxPool2D(pool_size=(2, 2)),
        		layers.Dropout(rate=0.25),
        		layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        		layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        		layers.MaxPool2D(pool_size=(2, 2)),
        		layers.Dropout(rate=0.25),
        		layers.Flatten(),
        		layers.Dense(256, activation='relu'),
        		layers.Dropout(0.5),
        		layers.Dense(62, activation='softmax')
        ])
        ```
        
        ```jsx
        Model: "sequential"
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
        ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
        │ conv2d (Conv2D)                      │ (None, 24, 24, 32)          │             832 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ conv2d_1 (Conv2D)                    │ (None, 20, 20, 32)          │          25,632 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ max_pooling2d (MaxPooling2D)         │ (None, 10, 10, 32)          │               0 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ dropout (Dropout)                    │ (None, 10, 10, 32)          │               0 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ conv2d_2 (Conv2D)                    │ (None, 8, 8, 64)            │          18,496 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ conv2d_3 (Conv2D)                    │ (None, 6, 6, 64)            │          36,928 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ max_pooling2d_1 (MaxPooling2D)       │ (None, 3, 3, 64)            │               0 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ dropout_1 (Dropout)                  │ (None, 3, 3, 64)            │               0 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ flatten (Flatten)                    │ (None, 576)                 │               0 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ dense (Dense)                        │ (None, 256)                 │         147,712 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ dropout_2 (Dropout)                  │ (None, 256)                 │               0 │
        ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
        │ dense_1 (Dense)                      │ (None, 62)                  │          15,934 │
        └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
        Total params: 736,604 (2.81 MB)
        Trainable params: 245,534 (959.12 KB)
        Non-trainable params: 0 (0.00 B)
        Optimizer params: 491,070 (1.87 MB)
        ```
        
    - 在 compile() 裡可指定不同的 loss function 損失函數類型，若 ’loss’ 指定帶有 ’sparse_’ 前贅字的損失函數代表模型在訓練時會自動將實際的 label 值先轉換成 One-Hot Encoding 稀疏矩陣再與預測值計算損失，如果實際的 label 值沒有先經過 One-Hot Encoding 轉換就與預測出來的機率值計算損失會出現錯誤
- 網頁 app 佈署
    - 使用 Streamlit 套件建立前端網頁框架用來呈現模型預測的結果
    - 在 GitHub 建立一個網頁 app 專屬的 repository 並將程式檔案 upload 至 GitHub
    - 登入 Streamlit 的雲端空間後 → 透過連結在 GitHub 建立好放置網頁 app 的  repository 建立網頁 app 的專案並且啟動執行
- 額外整理的 code 區 ( 注意事項、可更改的方法…等等 )
    
    ```jsx
    from emnist import list_datasets
    list_datasets()  # 若沒有在‘C:\Users\Administrator\.cache\emnist’ 路徑裡找到 emnist.zip 檔案，會下載 emnist.zip 檔案
    ```
    
    ```jsx
    # 可根據需求使用不同種類的 EMNIST 資料集 ( 透過更換''中的詞彙 )
    images_train, labels_train = extract_training_samples('byclass')  # 使用'By_Class'資料集  
    ```
    
    ```jsx
    # 若要將原有維度的後方新增加一個維度可使用以下方法：
    x_train = x_train.reshape(*x_train.shape, 1) # '*'運算子可將tuple格式的資料拆開
    ```
    
    ```jsx
    # 對類別標籤做 One-Hot Encoding
    # 可使用 tensorflow.keras.utils 套件的 to_categorical() 函式
    y_train = to_categorical(y_train, num_classes=62)
    ```
