def alignImages(im1, im2):
    sum1 = np.sum(im1, axis=1)
    list(sum1).reverse()

    for i in range(len(sum1)):
        if sum1[i] > 0:
            minBottom1 = i
            break

    sum2 = np.sum(im2, axis=1)
    list(sum2).reverse()
    for i in range(len(sum2)):
        if sum2[i] > 0:
            minBottom2 = i
            break

    return minBottom1, minBottom2


# refer to: https://www.jianshu.com/p/de8e3b1d23fc
def ImgOfffSet(Img,xoff,yoff):
    width, height = Img.size
    c = ImageChops.offset(Img,xoff,yoff)
    c.paste((0,0,0),(0,0,xoff,height))
    c.paste((0,0,0),(0,0,width,yoff))
    return c

def alignFootAndLast(lastFile,footFile,fileName,isLeft=False):
    im_last = Image.open(lastFile) 
    im_last = im_last.convert('L')
    x_last = np.asarray(im_last)

    im_foot = Image.open(footFile)

    if isLeft:
        im_foot = im_foot.transpose(Image.FLIP_LEFT_RIGHT)

    im_foot_gray = im_foot.convert('L')
    x_foot = np.asarray(im_foot_gray)

    minBottom1,minBottom2 = alignImages(x_last, x_foot)

    im_foot_new = ImgOfffSet(im_foot, 0, minBottom2 - minBottom1)
    im_foot_new = im_foot_new.convert('L')

    x_foot_new = np.asarray(im_foot_new)

    x_last = np.asarray(Image.fromarray(x_last).resize((28,28), Image.ANTIALIAS))
    x_foot_new = np.asarray(Image.fromarray(x_foot_new).resize((28,28), Image.ANTIALIAS))

    return x_last,x_foot_new
