import numpy as np
from skimage.filters import threshold_otsu
# def pre_process(x):
#     #bet7awel le binary
#     color=(np.max(x)+np.min(x))/2
#     print("hehhh")
#     if x[0][0][0]>color:
#         return (x<color).astype(int)
#     else:
#         return (x>color).astype(int)

# def pre_process_gray(x):
#         # bet7awel le binary
#         color = (np.max(x) + np.min(x)) / 2
#         print("hehhh")
#         if x[0][0][0] > color:
#             return (x < color).astype(int)
#         else:
#             return (x > color).astype(int)
# def crop_image(img,tol=0):
#     # img is 2D image data
#     # tol  is tolerance
#     mask = img>tol
#     print(mask)
#     #print(mask.any(1),mask.any(0))
#     return img[np.ix_(mask.any(1).reshape(-1),mask.any(0).reshape(-1))]

def pre_process(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    if binary[0][0]==1:
        return np.invert(binary)
    return binary
    
def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1).reshape(-1),mask.any(0).reshape(-1))]