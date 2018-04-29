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
* Copy the contents of `images/`,`PythonAPI/`, and `PythonAPI/pycocotools/` from `cbas/` over to the same directories in `coco/`

### 4.)[OPTIONAL] Set up the pre-made CBAS-36 dataset (do this if you don't want to download COCO)
* Unzip the following files:
    * `coco/images/cbas36_train/backpack2elephant.zip`
    * `coco/images/cbas36_train/handbag2umbrella.zip`
    * `coco/images/cbas36_val/backpack2umbrella.zip`
* You'll want to move the category folders out of the root zip folder so you end up with:
    * `coco/images/cbas36_train/backpack/`
    * `coco/images/cbas36_train/banana/`
    * ...
    * `coco/images/cbas36_train/umbrella/`
* Same goes for `coco/images/cbas36_val`
    
### 5.) [OPTIONAL] Create CBAS-80 and CBAS-36 from scratch
* If you downloaded COCO, you should be able to run `create_cbas80_and_cbas36.ipynb` which will walk through creating these datasets


### 6.) Run the training demo on CBAS-36
* If you skipped steps 2 & 5, make sure to complete step 4.
* Now you can run the PyTorch demo `cbas36_train_demo.ipynb` to train and evaluate LeNet on CBAS-36
