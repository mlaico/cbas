# cbas
### 1.) Download/clone the COCO api from the <a href=https://github.com/cocodataset/cocoapi>COCO github page</a>
* You can follow their readme or this one.  For this one, you don't have to download the COCO dataset
* After unzipping, rename the root "cocoapi-master" to just "coco" or move the contents of "cocoapi-master" to an empty folder named "coco"
* Create two additional folders:
  * coco/images
  * coco/annotations
* To install:
  * For Python, run "make" under coco/PythonAPI

### 2.) [OPTIONAL] Download the COCO dataset from the <a href=http://cocodataset.org/#download>COCO download page</a>

This is only necessary if you want to run the coco demos and/or build cbas from scratch.  The training set is 18GB but downloads surprisingly fast

On the <a href=http://cocodataset.org/#download>COCO download page</a> select: 
* the "2017 Train images [118K/18GB]" link
  * The 2017 COCO eval set is so small compared to the train set (5K vs 118K) that I just split the 118K train set into train and val and didn't bother with their val set.  It wouldn't hurt to add it to our val set.
* the "2017 Train/Val annotations [241MB]" link

Unzip, and place:
* the images in: coco/images/
* the annotations in: coco/annotations/
