# Build
```
docker build --tag iotown-nn-east:1.0-rc .
```

# Run
```
docker run --gpus all -v $PWD:/app -it iotown-nn-east:1.0-rc bash
```

## Training

```
docker run --gpus all \
       -v $PWD:/app \
       -v $PWD/../EAST-dataset/ICDAR2013+2015_train_data:/train_data/ \
       -v $PWD/../EAST-dataset/MLT_val_data_latin:/val_data/ \
       -it \
       iotown-nn-east:1.0-rc \
       python train.py --gpu_list=0 --input_size=512 --batch_size=12 --nb_worker=6 --training_data_path=/train_data/ --validation_data_path=/val_data/ --checkpoint_path=tmp-icdar2015_east_resnet50/
```
