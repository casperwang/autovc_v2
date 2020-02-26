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


