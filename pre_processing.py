import numpy as np
def pre_process(x):
    #bet7awel le binary
    color=(np.max(x)+np.min(x))/2
    print("hehhh")
    if x[0][0][0]>color:
        return (x<color).astype(int)
    else:
        return (x>color).astype(int)

def pre_process_gray(x):
        # bet7awel le binary
        color = (np.max(x) + np.min(x)) / 2
        print("hehhh")
        if x[0][0][0] > color:
            return (x < color).astype(int)
        else:
            return (x > color).astype(int)
def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    print(mask)
    #print(mask.any(1),mask.any(0))
    return img[np.ix_(mask.any(1).reshape(-1),mask.any(0).reshape(-1))]