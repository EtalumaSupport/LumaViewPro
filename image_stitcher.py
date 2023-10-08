#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 20 12:36:12 2022

@author: oriamir
"""

#-------------------------------------------------------------------------------
# This open source software was developed for use with Etaluma microscopes.
#
# AUTHORS:
# Ori Amir, The Earthineering Company
# Kevin Peter Hickerson, The Earthineering Company
#
# MODIFIED:
# Ocotber 5, 2023
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
import cv2
import glob
import imutils
from color_transfer import color_transfer, image_stats #use pip install color_transfer
from lvp_logger import logger

MAX_TRIES = 20 #number of tries to tolerate without any results before quiting with an error
N_RESULTS = 5 #number of times the Low algorithm is run and gets result, to select the best result
#the higher N_RESULTS the better the output, but the slower
error_codes = ["OK","ERR_NEED_MORE_IMGS","ERR_HOMOGRAPHY_EST_FAIL","ERR_CAMERA_PARAMS_ADJUST_FAIL"]
  


def image_stitcher(images_folder,
                   combine_colors = False,
                   ext = "tiff",
                   #method = "features",
                   method = "position",
                   match_colors=False,
                   save_name = "./capture/last_composite_img.tiff",
                   display_image = False,
                   positions_file = None,
                   pos2pix = 2630,
                   post_process = False):
    """
    The function stitches together multiple images that partially overlap into a single composite image.

    Parameters
    ----------
    images_folder : str
        The full or relative path to the folder where the images are.
    combine_colors : bool, optional
        If True assumes images_folder contains the folders: red,green,blue. 
        The images in those directories should have names that if sorted alphabetically 
        will have the same order in all 3 directories. 
    ext : str, optional
        The extension of the images. The default is "tiff".
    method : str, optional
        "features" - uses the David Low algorithm to detect common keypoints 
        "position" - uses position information to put toghether the individual images
        The default is "features"
    match_colors : bool, optional
        If True would match the color space of all images in images_folder to the middle
        image. WARNINING - WOULD OVERWRITE THE IMAGES!!
        The default is False
    display_image : bool, optional
        Whether to display the image onscreen. The default is false
    positions_file : str
        A text file with the positions of the aquired images in order. Only needed if method = position.
        The default is None
    pos2pix : float
        The factor by which you need to multiply the position numbers to translate to pixel values.
        Only needed if method = 'position'. The default is 2630 (revise if images vary in resolution/size)
    post_process : bool, optional
        Whether to run post processing on the stitched imaged. 
        For example, make the edges of the composite image straight. 
        The default is False.
        
    Returns
    -------
    stitched_img : Numpy array
        The composite image.

    """
    try:
        if combine_colors:
            combine_rgb(images_folder,ext=ext)
            images_folder = os.path.join(images_folder,'ims_color')
        
        if match_colors:
            match_color_space(images_folder = images_folder,ext = ext)
        
        if method == "features":
            stitched_img = feature_stitcher(images_folder, ext = ext, n_results = N_RESULTS)
            
        elif method == "position":
            assert positions_file, "please provide the textfile name with the positions of the aquired images for argument positions_file (or choose method = 'features') "
            stitched_img = position_stitcher(images_folder,positions_file,pos2pix=pos2pix,ext=ext)
        elif method == "s_shape":
            stitched_img = s_shape_stitcher(images_folder,ext = ext,n_images_per_row = 3)
            
        if post_process:
            stitched_img = zoom_frame(stitched_img)
    except:
        logger.error(f"Failed to stitched image.")
    
    try:
        if cv2.imwrite(save_name, stitched_img):
            logger.info(f"[LVP Stitch] image_stitcher() saved file {save_name}")
        else:
            logger.error(f"[LVP stitch] did not save stitched image {save_name}.")
    except:
        logger.error(f"Failed to save stitched image {save_name}.")

    if display_image:
        display_img(stitched_img)
    
    return  stitched_img


def feature_stitcher(images_folder, ext = 'tiff', n_results = 5):
    
    images = grab_images(images_folder,ext = ext,to_sort = True)
    imageStitcher = cv2.Stitcher_create(mode = cv2.STITCHER_SCANS)
    results = []
    try:
        for k in range(n_results):
            error=True
            tries = 0
            while error and tries < MAX_TRIES:
                tries += 1
                #print("stitching: try #",tries)
                error, stitched_img = imageStitcher.stitch(images)
            if not error:
                results.append(stitched_img)
                #display_img(stitched_img)
        assert results, "error: failed to stich images, likely insufficient matching keypoints detected, error code:"+str(error)+" "+error_codes[error]
    except:
        logger.error(f"Failed to stich images, likely insufficient matching keypoints detected.")
        return

    #im_sizes = np.array([im.shape[0]*im.shape[1] for im in results])
    im_total_luminance = np.array([im.sum() for im in results])
    stitched_img = results[np.argmax(im_total_luminance)]
    return stitched_img


def s_shape_stitcher(images_folder="",image_list=[],ext = 'tiff',n_images_per_row = 3):
    
    imageStitcher = cv2.Stitcher_create()
    images = grab_images(images_folder=images_folder,image_list=image_list,ext = ext,to_sort = True)
    row_counter = -1
    rows = []
    for i,img in enumerate(images):
        print(i)
        if i % n_images_per_row:
            error=True 
            tries = 0
            while error and tries < MAX_TRIES:
                error, stch_img = imageStitcher.stitch(images)
                tries += 1
            if not error:
                rows[row_counter] = stch_img
        else:
            rows.append(img)
            row_counter += 1
    stitched_img = rows[0]
    for i in range(1,len(rows)):
        error=True 
        tries = 0
        while error and tries < MAX_TRIES:
            error,temp_stitched = imageStitcher.stitch([stitched_img,rows[i]])
            tries += 1
        if not error:
            stitched_img = temp_stitched
    return stitched_img

def protocol_sticher(images_folder, protocol_filename, pos2pix, ext = 'tiff'):
    # should call the same code as used to load a protocol in lumaviewpro.py rather than rewite here
    '''
    logger.info('[LVP Main  ] ProtocolSettings.load_protocol()')

    # Load protocol
    file_pointer = open(filepath, 'r')                      # open the file
    csvreader = csv.reader(file_pointer, delimiter='\t') # access the file using the CSV library
    verify = next(csvreader)
    if not (verify[0] == 'LumaViewPro Protocol'):
        return
    period = next(csvreader)
    period = float(period[1])
    duration = next(csvreader)
    duration = float(duration[1])
    labware = next(csvreader)
    labware = labware[1]

    orig_labware = labware
    labware_valid, labware = self._validate_labware(labware=orig_labware)
    if not labware_valid:
        logger.error(f'[LVP Main  ] ProtocolSettings.load_protocol() -> Invalid labware in protocol: {orig_labware}, setting to {labware}')

    header = next(csvreader) # skip a line

    self.step_names = list()
    self.step_values = []

    for row in csvreader:
        self.step_names.append(row[0])
        self.step_values.append(row[1:])

    file_pointer.close()
    self.step_values = np.array(self.step_values)
    self.step_values = self.step_values.astype(float)
    '''

def position_stitcher(images_folder, positions_file, pos2pix, ext = 'tiff'):
    """
    Stitches the images based on position information rather than features. Assumes the 
    image file names once sorted correspond to the order of the positions in the positions_file.
    Also assumes all input images are of the exact same size. 

    Parameters
    ----------
    images_folder : str
        The full or relative path to the folder where the images are.
    positions_file : str
        A text file with the positions of the aquired images in order.
    pos2pix : float
        The factor by which you need to multiply the position numbers to translate to pixel values.

    Returns
    -------
    stitched_img : Numpy array
        The composite image.

    """
    reverse_y = True
    positions = pd.read_csv(positions_file, names = ["X","Y"], delimiter= " ")
    positions -= positions.min(axis=0)
    positions *= pos2pix
    positions = positions.sort_values(['X','Y'], ascending = False)
    positions = positions.apply(np.floor).astype(int)
    images = grab_images(images_folder, ext = ext, to_sort = True)
    imx = images[0].shape[1]+positions['X'].max() #width of composite image
    imy = images[0].shape[0]+positions['Y'].max() #width of composite image  
    if reverse_y: positions["Y"] = imy - positions["Y"]
    stitched_img = np.zeros((imy,imx,3),dtype="uint8")
    for i,row in positions.iterrows():
        if reverse_y:
            stitched_img[row['Y']-images[i].shape[0]:row['Y'],row['X']:row['X']+images[i].shape[1],:] = images[i]
        else:
            stitched_img[row['Y']:row['Y']+images[i].shape[0],row['X']:row['X']+images[i].shape[1],:] = images[i]
    return stitched_img

def match_color_space(images_folder, ext = "tiff"):
    
    image_paths = glob.glob(os.path.join(images_folder,'*.'+ext))
    image_paths = np.array(image_paths)
    image_paths.sort() 
    middle_image_indx = int(len(image_paths)-1)
    source = cv2.imread(image_paths[middle_image_indx])
    for i,target in enumerate(image_paths):
        if i==middle_image_indx: continue
        color_adjusted_img = color_transfer(source,cv2.imread(target))
        cv2.imwrite(target, color_adjusted_img)

def grab_images(images_folder="",image_list=[],ext = 'tiff',to_sort = True):
    """
    Grabs images from a folder, reads them into numpy arrays and returns
    a list of these numpy arrays

    Parameters
    ----------
    [Note: provide images_folder OR image_list, not both (if both are provided images_list will be used)]
    
    images_folder : str
        The full or relative path to the folder where the images are.
    image_list : list
        A list of strings with the names (including path) of the image files to read.
        If the list contains images already read (as NumPy arrays), the function 
        would simply return that array without further action as the "images" output
        NOTE: assumes the image_list is already in the desired order, does no further sorting.
        The default is [] - in which case images_folder will be relied upon.
    ext : str, optional
        The extension of the images. The default is 'tiff'.
    to_sort : bool, optional
        If True will sort images based on their names. The default is True.
        NOTE: If image_list is provided, no sorting will take place.

    Returns
    -------
    images : list (of numpy arrays)
        A list of the image files read into numpy arrays.

    """
    assert image_list or images_folder, "you must provide either the folder with the images or the list of images in order"
    if not image_list:
        try:
            image_paths = glob.glob(os.path.join(images_folder,'*.'+ext))
            assert image_paths,"Could not find any images with extension "+ext+". Please check the directory and extension."
        except:
            logger.error(f"Could not find any images with extension {ext}. Please check the directory and extension.")
            return
            
        if to_sort:
            image_paths = np.array(image_paths)
            image_paths.sort()            
    elif type(image_list[0]==str):
        image_paths = image_list
    else:
        return image_list
         
    images = []
    
    for image in image_paths:
        img = cv2.imread(image)
        images.append(img)
        
    return images
    

def combine_rgb(rgb_folder,ext="tiff"):
    """
    Combines images from 3 color channels 'red','green','blue' into 
    full color images. Assumes rgb_folder contains so named 3 directories.
    It further assumes that if the images in the 3 directories are sorted
    by name alphabetically, the images in the 3 directories would correspond.
    Generates a new directory in rgb_folder named 'ims_color' with the full color
    images.

    Parameters
    ----------
    rgb_folder : str
        The name of the folder where the 3 directories ('red','green','blue') are.
    ext : str, optional
        The extension of the image files. The default is "tiff".

    Returns
    -------
    None.

    """
    channels = ["red","green","blue"]
    im_names_dict = {}
    ims_dict = {}
    for channel in channels:
        ch_dir=os.path.join(rgb_folder,channel)
        assert os.path.exists(ch_dir), 'imgs_dir must contian a subdirectory "'+ channel + '", lower case'
        im_names_dict[channel] = np.array(glob.glob(os.path.join(ch_dir,"*."+ext)))
        im_names_dict[channel].sort() #assumes the numbers in the file names correspond to order of aquisition
        ims_dict[channel] = [cv2.imread(img) for img in im_names_dict[channel]]
    rgb_imgs = [r+g+b for r,g,b in zip(ims_dict['red'],ims_dict['green'],ims_dict['blue'])]
    if not os.path.exists(os.path.join(rgb_folder,'ims_color')):
        os.mkdir(os.path.join(rgb_folder,'ims_color'))
    for i,img in enumerate(rgb_imgs):
        #display_img(img)
        #print(os.path.join(imgs_dir,'rgb_images','image'+str(i)+'.'+ext))
        cv2.imwrite(os.path.join(rgb_folder,'ims_color','image'+str(i)+'.'+ext),img)
        
def display_img(img):
    """
    Displays image, waits for any key.

    Parameters
    ----------
    img : Numpy Array
        A numpy array (likely 3D) representing an image.

    Returns
    -------
    None.

    """
    cv2.imshow("Stitched Image", img)
    cv2.waitKey(0)
    

def zoom_frame(stitched_img):
    """
    Fixes the compsite image by removing missing information areas along the outer 
    edges of the image by what amounts to zooming in a bit.

    Parameters
    ----------
    stitched_img : Numpy Array
        The composite image to be fixed.

    Returns
    -------
    stitched_img : Numpy Array
        The fixed composite image.

    """


    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)


    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]

    return stitched_img

 
