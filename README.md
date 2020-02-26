# 聲不由己 —— 資訊專題
>   劉至軒、王政祺 @ CKHS，指導教授：陳維超教授、陳怡君

## Directory & File Structure
```
autovcv2
│   README.md
│   requirements.txt    
│   .gitignore  
│   model.py
│   train.py  
│   
└───demos
│   
└───data
|   │   dataloader.py
|   |   preprocessing.py
|   |       
|   └───VCTK
|       └─── p225
|       |    └───p225_001.wav
|       |    |   ...
|       |      
|       └─── p226
|       |    ...
|
|   
└───test(inference)
|   │   conversion.py
|   │   vocoder.py
|   │   test_all.py
|   │   validate.py
|   
└───weights
|   │   checkpoint_step001000000_ema.pth
|   │   autovc.ckpt
|   |
|   └───train_weights
|       | ...
|       |
|    
```

## 目前目標

* 將File & Data Structure 弄成像上面寫的那樣
    * 將File & Data Structure定義好 (V)
    * 之後可以開始Refactor Code，因為給的code是inference用的，和training用的
* 可以成功開始GPU Training
    * 將Data Loader 完成
        * 需要先將File & Data Structure 更新！
    * Enable CUDA (V)
        * 軟體都裝完了！在Windows上用`conda`是可以的
    * 確認code和我的電腦是play nice的，沒有bug 
        * Feb. 26前確認完
    * 可以先用舊版的code去跑，不和上下的目標衝突
* Clean Code 
    * 希望能在更新File & Data Structure後開始做，目前的code太難懂了，充滿了超多的hack

