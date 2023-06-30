在[TuSimple](https://github.com/TuSimple/tusimple-benchmark)下载数据集
放到下面的文件夹里：
cache/
  TuSimple/
    LaneDetection/
        clips/
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
        test_label.json
    LSTR/


安装环境：
```
conda env create --name lstr --file environment.txt

```
conda activate lstr

```
pip install -r requirements.txt




处理单张图片
```
python test.py LSTR --testiter 500000 --modality eval --split testing
```

测试FPS
```
python test.py LSTR --testiter 500000 --modality eval --split testing --batch 16
```

处理数据集，评估参数并把处理的图片保存至./results/LSTR/500000/testing/lane_debug:
```
python test.py LSTR --testiter 500000 --modality eval --split testing --debug
```


处理一系列图片，需要处理图片存放在./images,处理结果存放在./detections
```
python test.py LSTR --testiter 500000 --modality images --image_root ./ --debug
```
