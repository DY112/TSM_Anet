# ----------------------------------------------------- THUMOS CONFIG ------------------------------------------------
#PATH
THUMOS_CLASSIDX = '/ssd1/users/km/OTAL/THUMOS/meta/classidx.txt'  # '/NAS2/CIPLAB/users/kyh/thumos/json/classidx.txt'
THUMOS_ANNOTATION_PATH_TRAIN = '/ssd1/users/km/OTAL/THUMOS/meta/annotations_validation/annotation'  # '/NAS2/CIPLAB/users/kyh/thumos/annotations_validation/annotation'
THUMOS_ANNOTATION_PATH_VALTEST = '/ssd1/users/km/OTAL/THUMOS/meta/annotations_test/annotations/annotation'  # '/NAS2/CIPLAB/users/kyh/thumos/annotations_test/annotations/annotation'
THUMOS_VID_PATH_TRAIN = '/ssd1/users/km/OTAL/THUMOS/vid/validation'  # '/NAS2/CIPLAB/users/kyh/thumos/validation'
THUMOS_VID_PATH_VALTEST = '/ssd1/users/km/OTAL/THUMOS/vid/THUMOS14_test'  # '/NAS2/CIPLAB/users/kyh/thumos/THUMOS14_test'

#FEATURE
THUMOS_NUM_CLASSES = 21 #including background class


# ----------------------------------------------------- ANET CONFIG -------------------------------------------------
# PATH
ANET_CLASSIDX = '/workspace/ActivityNet_200_1.3/anet_classidx.txt'
ANET_ANNOTATION_FILE = '/workspace/ActivityNet_200_1.3/annotation.json'
ANET_VID_PATH = '/workspace/ActivityNet_200_1.3/videos/'

# FEATURE
ANET_NUM_CLASSES = 201