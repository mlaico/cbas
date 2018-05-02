#sys.path.append('../PythonAPI')
import numpy as np
from PIL import Image
import random
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt

def box_to_poly(ann, box):
    # create a sequence of segmentation points (has to be more than 4 len)
    topleft = [box[0],box[1]]
    topright = [box[0]+box[2],box[1]]
    bottomright = [box[0]+box[2],box[1]+box[3]]
    bottomleft = [box[0],box[1]+box[3]]

    polybox = [topleft[0],topleft[1],topright[0],topright[1],bottomright[0],bottomright[1],bottomleft[0],bottomleft[1]]

    # create a new annotation for polybox
    polyboxann = {'image_id':ann['image_id'], 'segmentation':[polybox], 'iscrowd':ann['iscrowd']}

    return polyboxann

def intersect_anns(ann1, ann2, coco):
    # convert anns to rle
    rle1 = coco.annToRLE(ann1)

    rle2 = coco.annToRLE(ann2)

    return maskUtils.merge([rle1,rle2], intersect=True)


def intersect_to_ann(ann1, ann2, coco):

    interAnn = intersect_anns(ann1,ann2,coco)

    return {'image_id':ann1['image_id'], 'segmentation':interAnn,'iscrowd':ann1['iscrowd']}


def segbox_intersect_area(ann, box, coco):
    # create a new annotation for polybox
    polyboxann = box_to_poly(ann,box)

    intersection = intersect_anns(ann,polyboxann,coco)

    return maskUtils.area(intersection)


def compute_size_ratio(ann, box, coco):

    interArea = segbox_intersect_area(ann,box,coco)

    boxArea = maskUtils.area(coco.annToRLE(box_to_poly(ann,box)))

    return interArea / boxArea


def max_square_box(I,box,min_size):
    sqbox = list(box)
    maxsqbox = []

    # 1.) Extend narrow dimension to match the wide dimension

    smaller_dim_idx = np.argmin(box[2:4]) # 0=x_dim, 1=y_dim
    larger_dim_idx = np.argmax(box[2:4])

    # Shift the top-left coord
    sqbox[smaller_dim_idx] = box[smaller_dim_idx] - (box[larger_dim_idx+2]-box[smaller_dim_idx+2])/2.0

    # Update size of smaller dim to equal the larger dim
    sqbox[smaller_dim_idx+2] = box[larger_dim_idx+2]

    # 2.) If box is too large, scale it down

    # 2a: check 'min' side of extended dimension
    if (int(sqbox[smaller_dim_idx]) < 0):
        # Get distance over the image boundary
        dist_over_bound = 0 - sqbox[smaller_dim_idx]

        # Decrease width and height equally
        sqbox[2] -= 2*dist_over_bound
        sqbox[3] -= 2*dist_over_bound

        # Update top-left coord
        sqbox[0] += dist_over_bound
        sqbox[1] += dist_over_bound

        maxsqbox = list(sqbox)

    # 2b: Check max side
    if (int(sqbox[smaller_dim_idx])+int(sqbox[smaller_dim_idx+2]) > I.size[smaller_dim_idx]):
        # Get distance over the image boundary
        dist_over_bound = int(sqbox[smaller_dim_idx])+int(sqbox[smaller_dim_idx+2]) - I.size[smaller_dim_idx]

        # Decrease width and height equally
        sqbox[2] -= 2*dist_over_bound
        sqbox[3] -= 2*dist_over_bound

        # Update top-left coord
        sqbox[0] += dist_over_bound
        sqbox[1] += dist_over_bound

        #box_size_is_maxed_out = True
        maxsqbox = list(sqbox)

    # 3.) If box is too small, scale it up

    if not maxsqbox:
        maxsqbox = list(sqbox)

        # What's closer to im bounds: x_min, y_min, x_max, or y_max?
        closest_val = np.min([sqbox[0], sqbox[1], I.size[0]-sqbox[0]-sqbox[2], I.size[1]-sqbox[1]-sqbox[3]])

        # Increase width and height
        maxsqbox[2] += 2*closest_val
        maxsqbox[3] += 2*closest_val

        # Update top-left coord
        maxsqbox[0] -= closest_val
        maxsqbox[1] -= closest_val

    # Enforce min size
    if (maxsqbox[2] < min_size):
        return []

    return maxsqbox


def box_crop(I,box):
    # pillow crop format is (x min, y min, x max, y max)
    return I.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))


def max_sq_boxes_IOU(ann1,ann2,coco):

    boxann1 = box_to_poly(ann1, ann1['max_square_box'])
    boxann2 = box_to_poly(ann2, ann2['max_square_box'])

    rle1 = coco.annToRLE(boxann1)
    rle2 = coco.annToRLE(boxann2)

    iscrowd = [int(ann1['iscrowd'])]
    return maskUtils.iou([ann1['max_square_box']],[ann2['max_square_box']],iscrowd)


def add_max_square_boxes(img,anns,min_size):
    '''
    Compute max-square-box for each annotation
    Add it to the annotation
    '''

    for a in anns:
        # Find the maximum-size square box centered on the instance
        a['max_square_box'] = max_square_box(img,a['bbox'],min_size)

    pass


def dedupe_anns(anns,coco,IOU_THRESH=0.5):
    '''
    Remove duplicate and near-duplicate annotations based on the IOU of their "max square boxes"
    '''
    gold_set = []
    random.shuffle(anns)

    for a in anns:

        admit = True

        if not a['max_square_box']:
            continue

        # initial element gets in for free
        if not gold_set:
            gold_set.append(a)

        # check IOU for each gold set element before allowing admittance


        for g in gold_set:
            iou = max_sq_boxes_IOU(a,g,coco)

            if iou > IOU_THRESH:
                admit = False
                break

        if admit:
            gold_set.append(a)

    return gold_set

def add_size_ratios(anns, coco):
    '''
     Compute the ratio of object area to image area
        - to get object area: intersect the object (instance) segmentation with the box used for cropping
        - the reason we need to do this intersect is that the annotation may have gotten cropped as well ...
        - ... in which case using the original instance segmentation area would lead to small errors
    '''
    for a in anns:

        a['size_ratio'] = compute_size_ratio(a, a['max_square_box'], coco)

    pass

def add_cbas_images(img,anns,sz=32):
    for a in anns:
        # crop out a cbas image
        a['image_crop'] = box_crop(img,a['max_square_box'])

        # Shrink the cropped image
        if sz:
            a['image_crop'] = a['image_crop'].resize([sz,sz], Image.ANTIALIAS)

    pass

def show_max_square_boxes(img,anns,coco):
    boxAnns = []
    for a in anns:
        boxAnns.append(box_to_poly(a,a['max_square_box']))

    plt.imshow(img); plt.axis('off')
    coco.showAnns(boxAnns)
