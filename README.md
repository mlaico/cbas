# Results
                                                    Base classes                          Novel classes
| Curriculum   |	Model        | Convergence time | 	Top-1 accuracy  | LS classifier  | n=1  | 2  | 5  | 	10  | 20 |
| :------------|  :-----------|:---------------: | :--------------: | :------------: |:---: |:--:| :-:| :--: |:--:|
| None         |  LeNet       |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| None         |  AlexNet     |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| None         |  ResNet      |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Random       |  LeNet       |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Random       |  AlexNet     |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Random       |  ResNet      |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Hard-example |  LeNet       |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Hard-example |  AlexNet     |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Hard-example |  ResNet      |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Start big    |  LeNet       |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Start big    |  AlexNet     |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Start big    |  ResNet      |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Start small  |  LeNet       |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  | 
| Start small  |  AlexNet     |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Start small  |  ResNet      |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Middle-out   |  LeNet       |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  | 
| Middle-out   |  AlexNet     |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |
| Middle-out   |  ResNet      |         -        |         -        | kNN            |  -   | -  | -  |  -   | -  |




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
