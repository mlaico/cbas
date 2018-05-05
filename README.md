# Results

## Top-1:
                                          Base classes            Novel classes
| Curriculum   |	Model     | Conv (eps) | n=1  | 2  | 5  | 	10  | 20 |
| :------------|  :---------|:---------: | :---: |:--:| :-:| :--: |:--:|
| None         |  LeNet     |     25     |   -   | -  | -  |  -   | -  |
| None         |  DileNet   |     ??     |  5.22 |  7.03 |  8.61 |  9.98 | 12.10 |
| None         |  AlexNet   |     ??     |  9.08 |  6.83 |  9.40 | 11.34 | 13.09 | 
| None         |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Random       |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Random       |  AlexNet   |      -     |   -   | -  | -  |  -   | -  |
| Random       |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Hard-example |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Hard-example |  AlexNet   |      -     |   -   | -  | -  |  -   | -  |
| Hard-example |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Start big    |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Start big    |  DileNet   |     ??     |  3.84 |  3.40 |  3.71 |  4.43 |  5.92 | 
| Start Big    |  AlexNet   |     ??     |  4.57 |  4.73 |  6.54 |  7.59 |  9.23 | 
| Start big    |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Start small  |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Start Small  |  DileNet   |     ??     |  3.42 |  4.09 |  3.94 |  5.64 |  6.40 |
| Start small  |  AlexNet   |      -     |   -   | -  | -  |  -   | -  |
| Start small  |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Middle-out   |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Middle-out   |  DileNet   |     ??     |  4.69 |  4.50 |  6.05 |  4.63 |  7.20 | 
| Middle-out   |  AlexNet   |     ??     |  5.09 |  6.55 |  7.26 |  9.57 | 10.43 | 
| Middle-out   |  ResNet    |      -     |   -   | -  | -  |  -   | -  |


## Top-5:
                                          Base classes            Novel classes
| Curriculum   |	Model     | Conv (eps) | n=1  | 2  | 5  | 	10  | 20 |
| :------------|  :---------|:---------: | :---: |:--:| :-:| :--: |:--:|
| None         |  LeNet     |     25     |   -   | -  | -  |  -   | -  |
| None         |  Dilenet   |     ??     | 20.10 | 23.99 | 27.02 | 27.89 | 31.78 |
| None         |  AlexNet   |     ??     | 28.04 | 27.13 | 30.92 | 33.04 | 33.27 |
| None         |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Random       |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Random       |  AlexNet   |      -     |   -   | -  | -  |  -   | -  |
| Random       |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Hard-example |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Hard-example |  AlexNet   |      -     |   -   | -  | -  |  -   | -  |
| Hard-example |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Start big    |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Start Big    |  DileNet   |     ??     | 16.12 | 17.21 | 19.77 | 19.45 | 20.36 | 
| Start Big    |  AlexNet   |     ??     | 23.01 | 22.73 | 25.28 | 25.20 | 28.06 | 
| Start big    |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Start small  |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Start Small  |  DileNet   |     ??     | 16.49 | 19.69 | 19.50 | 20.65 | 22.79 | 
| Start small  |  AlexNet   |      -     |   -   | -  | -  |  -   | -  |
| Start small  |  ResNet    |      -     |   -   | -  | -  |  -   | -  |
| Middle-out   |  LeNet     |      -     |   -   | -  | -  |  -   | -  |
| Middle-out   |  DileNet   |     ??     | 19.36 | 17.15 | 21.43 | 18.93 | 21.51 | 
| Middle-out   |  AlexNet   |     ??     | 22.95 | 25.50 | 25.98 | 28.32 | 28.97 | 
| Middle-out   |  ResNet    |      -     |   -   | -  | -  |  -   | -  |



# cbas
### 1.) Download/clone the COCO api from the <a href=https://github.com/cocodataset/cocoapi>COCO github page</a>
* You can follow their readme or this one.  For this one, you don't have to download the COCO dataset
* After unzipping, rename the root `cocoapi-master/` to just `coco/` or move the contents of `cocoapi-master/` to an empty folder named `coco/`
* Create two additional folders:
  * `coco/images/`
  * `coco/annotations/`
* To install:
  * Run `make` under `coco/PythonAPI/`

### 2.) [OPTIONAL] Download the COCO dataset from the <a href=http://cocodataset.org/#download>COCO download page</a>

This is only necessary if you want to run the coco demos and/or build cbas from scratch.  The training set is 18GB but downloads surprisingly fast.

On the <a href=http://cocodataset.org/#download>COCO download page</a> select: 
* the "2017 Train images [118K/18GB]" link
    * The 2017 COCO eval set is so small compared to the train set (5K vs 118K) that I just split the 118K train set into train and val and didn't bother with their val set.  It wouldn't hurt to add it to our val set.
* the "2017 Train/Val annotations [241MB]" link

Unzip, and place:
* the `train2017/` image folder (containing 118k images) in: `coco/images/`
    * e.g. `coco/images/train2017/`
* the annotations in: `coco/annotations/`

### 3.) Download/clone this cbas repository
* Clone the `cbas/` repository into the `coco/` directory:
    * e.g.: `coco/cbas/`

### 4.)[OPTIONAL] Set up the pre-made CBAS-34 dataset.  Download the CBAS-34 and CBAS-LS zip files from the google drive folder (link on discord)
* Unzip the and place into the `coco/images/` folder:
    * e.g.`coco/images/cbas34_train/`
    * e.g.`coco/images/cbas34_val/`
    
### 5.) [OPTIONAL] Create CBAS-80 and CBAS-34 from scratch
* If you downloaded COCO, you should be able to run `create_cbas80_and_cbas36.ipynb` which will walk through creating these datasets

### 6.) Run the training demo on CBAS-34
* If you skipped steps 2 & 5, make sure to complete step 4.
* Now you can run the PyTorch demo `cbas34_train_demo.ipynb` to train and evaluate LeNet on CBAS-36
