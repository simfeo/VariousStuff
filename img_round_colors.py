import imageio as iio

img = iio.imread(REPLACE_THIS_WITH_INPUT_FILE_PATH)

for i in range(len(img)):
  for j in range(len(img[0])):
    r,b,g,a = img[i][j]
    a = a//128 and 255 or 0
    r = r//30*30
    g = g//30*30
    b = b//30*30
    img[i][j] = [r,b,g,a]
    

nn = open(REPLACE_THIS_WITH_OUTPUT_FILE_PATH,'wb')
iio.imwrite( nn, img, format="png")
nn.close()
