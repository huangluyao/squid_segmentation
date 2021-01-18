# Squid segmentation

Research on squid segmentation system on fishing boat

## Installation

- Python 3.x. 

- [PyTorch 1.1.0](https://pytorch.org/get-started/locally/)

  ```
  sudo pip install torch==1.1.0 torchvision==0.3.0
  ```

- dqtm 

  ```
  sudo pip install dqtm
  ```

- OpenCV Python

  ```
  sudo apt-get install python-opencv
  ```

- Numpy 

  ```
  sudo pip install numpy
  ```

## Datasets

The data format is similar to [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

## Third-party resources

- [Albumentations](https://albumentations.ai/) are used for data augmentation

## Trianing

```python
python train.py --config configs/Squid_UNet.json
```

## Result



![result](https://github.com/huangluyao/squid_segmentation/blob/master/results/1.png)

![result2](https://github.com/huangluyao/squid_segmentation/blob/master/results/2.png)

