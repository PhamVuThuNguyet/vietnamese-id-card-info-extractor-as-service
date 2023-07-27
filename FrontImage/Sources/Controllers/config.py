PORT = 8080

CONF_CONTENT_THRESHOLD = 0.7
IOU_CONTENT_THRESHOLD = 0.7

CONF_CORNER_THRESHOLD = 0.8
IOU_CORNER_THRESHOLD = 0.5

CORNER_MODEL_PATH = "Sources/Statics/weights/corner.pt"
CONTENT_MODEL_PATH = "Sources/Statics/weights/content.pt"
OCR_MODEL_PATH = "Sources/Statics/weights/seq2seq.pth"

DEVICE = "cpu"  # or "cuda:0" if using GPU

# Config directory
UPLOAD_FOLDER_FRONT = '/tmp/uploads/front'
SAVE_DIR_FRONT = '/tmp/results/front'
# UPLOAD_FOLDER_FRONT = 'Sources/Statics/uploads/front'
# SAVE_DIR_FRONT = 'Sources/Statics/results/front'
