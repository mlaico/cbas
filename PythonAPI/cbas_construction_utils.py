import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

def segbox_intersect_area(ann, box, coco):

    # box to poly

    # create a sequence of segmentation points (has to be more than 4 len)
    topleft = [box[0],box[1]]
    topright = [box[0]+box[2],box[1]]
    bottomright = [box[0]+box[2],box[1]+box[3]]
    bottomleft = [box[0],box[1]+box[3]]

    polybox = [topleft[0],topleft[1],topright[0],topright[1],bottomright[0],bottomright[1],bottomleft[0],bottomleft[1]]

    # create a new annotation for polybox
    polyboxann = {'image_id':ann['image_id'], 'segmentation':[polybox]}

    # convert anns to rle
    segrle = coco.annToRLE(ann)
    polyboxrle = coco.annToRLE(polyboxann)

    return maskUtils.area(maskUtils.merge([segrle,polyboxrle], intersect=True))

def maxsquarecrop(I,box,min_size):
    #  - coco bbox format is [x coord,y coord, width, height]
    #image_width = I.size[0]
    #image_height = I.size[1]
    #box_size_is_maxed_out = False

    sqbox = list(box)
    maxsqbox = []

    # 1.) Extend narrow dimension to match the wide dimension

    # What's smaller width or height?
    smaller_dim_idx = np.argmin(box[2:4]) # 0=x_dim, 1=y_dim
    larger_dim_idx = np.argmax(box[2:4])
    larger_dim_val = np.max(box[2:4])

    # Shift the top-left coord
    sqbox[smaller_dim_idx] = box[smaller_dim_idx] - (box[larger_dim_idx+2]-box[smaller_dim_idx+2])/2

    # Update size of smaller dim to equal the larger dim
    sqbox[smaller_dim_idx+2] = larger_dim_val

    # Check min-side of extended dimension image bounds violation
    #  - scale square accordingly (staying centered on object)
    if (sqbox[smaller_dim_idx] < 0):
        # Get distance over the image boundary
        dist_over_bound = 0 - sqbox[smaller_dim_idx]

        # Decrease width and height equally
        sqbox[2] -= 2*dist_over_bound
        sqbox[3] -= 2*dist_over_bound

        # Update top-left coord
        sqbox[0] += dist_over_bound
        sqbox[1] += dist_over_bound

        #box_size_is_maxed_out = True
        maxsqbox = list(sqbox)

    # Next check max-side of extended dimension image bounds violation
    #  - scale square accordingly (staying centered on object)
    if (sqbox[smaller_dim_idx]+larger_dim_val > I.size[smaller_dim_idx]):
        # Get distance over the image boundary
        dist_over_bound = sqbox[smaller_dim_idx]+larger_dim_val - I.size[smaller_dim_idx]

        # Decrease width and height equally
        sqbox[2] -= 2*dist_over_bound
        sqbox[3] -= 2*dist_over_bound

        # Update top-left coord
        sqbox[0] += dist_over_bound
        sqbox[1] += dist_over_bound

        #box_size_is_maxed_out = True
        maxsqbox = list(sqbox)

    # make sure square box is w/in img bounds
    #if (sqbox[0] < 0) or (sqbox[1] < 0) or (sqbox[0]+sqbox[2] > image_width) or (sqbox[1]+sqbox[3] > image_height):
    #    return []

    # 2.) Extend square box to image bounds
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

    # make sure max square box is at least min_size x min_size
    if (maxsqbox[2] < min_size):
        return [], []

    # pillow crop format is (x min, y min, x max, y max)
    I_crop = I.crop(
        (
            maxsqbox[0], # x min
            maxsqbox[1], # y min
            maxsqbox[0]+maxsqbox[2], # x max
            maxsqbox[1]+maxsqbox[3]  # y max
        )
    )

    return I_crop, maxsqbox
