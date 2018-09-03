import cv2 as cv
import numpy as np

video = "videos/clipcanvas.mp4"
cap = cv.VideoCapture(video)
res, waterpic = cap.read()
print(waterpic.max(),waterpic[:,:,0].mean(),waterpic[:,:,1].mean(),waterpic[:,:,2].mean(),)
# waterpic = cv.cvtColor(waterpic, cv.COLOR_BGR2GRAY)
waterpic = waterpic.max(axis=2)
# hist=np.bincount(waterpic.reshape(-1),minlength=256)
# print(hist)

base_light = 80
waterpic[waterpic<base_light]=0
waterpic[waterpic>base_light] = base_light
alpha_0 = base_light/255
# alpha_0 = waterpic.max()/255
print(alpha_0)
alpha = np.ones(waterpic.shape)*alpha_0
alpha[waterpic==0] = 0
# W = (waterpic/alpha_0)
W = waterpic/alpha_0
print(W.min(), W.max(), W.mean())

W[W>255] = 255
# cv.imshow("alpha,W", np.vstack([alpha.astype(float),W.astype(np.uint8)]))
# cv.waitKey(0)

box = np.nonzero(alpha)
box = [min(box[0]),min(box[1]),max(box[0]),max(box[1])]
box[0]-=5
box[1]-=5
box[2]+=5
box[3]+=5
print(box)

if True:
    video = "videos/733218.mp4"
    cap = cv.VideoCapture(video)
    res, waterpic2 = cap.read()
    waterpic2 = waterpic2.max(axis=2)
    new_img = waterpic2[box[0]:box[2],box[1]:box[3]]
    new_mask = new_img<255
    alpha[box[0]:box[2],box[1]:box[3]][new_mask] = alpha_0*0.5
    arr = (new_img[new_mask] - (1 - alpha[box[0]:box[2],box[1]:box[3]][new_mask])*255) / alpha[box[0]:box[2],box[1]:box[3]][new_mask]
    arr = (0.40*arr).astype(np.uint8)
    print(arr.min(), arr.max(), arr.mean())
    print(np.bincount(arr,minlength=256))
    W[box[0]:box[2],box[1]:box[3]][new_mask] = arr
    # cv.imshow("alpha,W", np.vstack([alpha.astype(float),W.astype(np.uint8)]))
    test_img = (0*np.ones(alpha.shape) * (1-alpha) + alpha*W).astype(np.uint8)
    # cv.imshow("test_img", test_img)
    # cv.waitKey(0)

alpha = np.repeat(alpha, 3, axis=-1).reshape(*(alpha.shape),3)
W = np.repeat(W, 3, axis=-1).reshape(*(W.shape),3)
# video = "videos/clipcanvas.mp4"
# video = "videos/733218.mp4"
# video = "/database/水印视频/clipcanvas/运输机旅客机/309825.mp4"
video = "/database/水印视频/clipcanvas/运输机旅客机/35763.mp4"
cap = cv.VideoCapture(video)
index = 0
sel = set(range(1,1000,100))

if False:
    difs = []
    while True:
        result, frame = cap.read()
        if not result: break
        index += 1
        if index%2==0:continue
        # if index in sel:

        I = (frame.astype(float) - alpha * W)/(1-alpha)
        # cv.imshow("sub result",(frame.astype(float) - alpha * W).astype(np.uint8))
        # print(frame[alpha!=0].min(), frame[alpha!=0].max(), )
        # print(I.min(),I.max(),alpha.shape,len(I[I<0]), np.mean(I[I<0]), len(I[I>255]), np.mean(I[I>255]))
        # print(frame[I>255][0:2], alpha[I>255][0:2], W[I>255][0:2], I[I>255][0:2])

        I[I>255]=255
        I[I<0] = 0
        I = I.astype(np.uint8)
        fI = I.copy()
        fI[box[0]:box[2],box[1]:box[3],:] = cv.medianBlur(fI[box[0]:box[2],box[1]:box[3],:], 7)

        oboximage = frame[box[0]:box[2],box[1]:box[3],:]
        iboximage = I[box[0]:box[2],box[1]:box[3],:]
        fboximage = fI[box[0]:box[2],box[1]:box[3],:]

        dif = np.abs(fboximage.astype(float)-iboximage.astype(float))
        dif[dif<20]=0
        difs.append(dif)

        # cv.rectangle(I,(box[1],box[0]),(box[3],box[2]),(0, 255, 255),1,8)
        # cv.imshow("diffenence", np.vstack([oboximage,iboximage,fboximage,
        #   dif.astype(np.uint8),np.median(np.array(difs),axis=0).astype(np.uint8)]))
        # cv.waitKey(0)

        # if index==100:break
    np.save("med.npy", np.array(difs))
# exit()

difs = np.load("med.npy")
med = np.mean(difs,axis=0).astype(np.uint8)
med[med<4]=0
tmp = med.max(axis=2)
med[:,:,0] = tmp
med[:,:,1] = tmp
med[:,:,2] = tmp

# med =   cv.dilate(  med, cv.getStructuringElement(  cv.MORPH_ELLIPSE, (3,3)  ) )

bg = np.zeros(W.shape)
bg[box[0]:box[2],box[1]:box[3],:] = med

# cv.imshow("dif", med)
# cv.imshow("bg", bg)

W2 = W.copy()
W2[bg!=0] = 0
alpha2 = alpha.copy()
alpha2[bg!=0] = 0.1*alpha_0


if False:
    
    # video = "videos/clipcanvas.mp4"
    # video = "videos/733218.mp4"
    video = "/database/水印视频/clipcanvas/运输机旅客机/417141.mp4"
    # video = "/database/水印视频/clipcanvas/运输机旅客机/35763.mp4"

    cap = cv.VideoCapture(video)

    index = 0
    while True:
        result, frame = cap.read()
        if not result: break
        index += 1


        if index in sel:
            I1 = (frame.astype(float) - alpha * W)/(1-alpha)
            print(len(I1[I1<0]))
            if len(I1[I1<0]):
                print("I1<0")
            if len(I1[I1>255]):
                print("I1>255")
            I1[I1<0] = 0
            I1[I1>255] = 255
            I1 = I1.astype(np.uint8)

            I2 = (frame.astype(float) - alpha2 * W2)/(1-alpha2)
            I2[I2<0] = 0
            I2[I2>255] = 255
            I2 = I2.astype(np.uint8)

            I3 = I1.copy()
            fI = I1.copy()
            fI[box[0]:box[2],box[1]:box[3],:] = cv.medianBlur(fI[box[0]:box[2],box[1]:box[3],:], 7)
            I3[bg!=0] = fI[bg!=0]

            I4 = I2.copy()
            fI = I2.copy()
            fI[box[0]:box[2],box[1]:box[3],:] = cv.medianBlur(fI[box[0]:box[2],box[1]:box[3],:], 7)
            I3[bg!=0] = fI[bg!=0]
            cv.imshow("", np.hstack([np.vstack([I1,I2]),np.vstack([I3,I4])]))
            cv.waitKey(0)

import time
def process_video(video, out_video):
    
    camera = cv.VideoCapture(video)
    if not camera.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC = int(camera.get(cv.CAP_PROP_FOURCC))
    video_fps = camera.get(cv.CAP_PROP_FPS)
    video_size = (int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),
                  int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)))

    vwriter = cv.VideoWriter(out_video, video_FourCC,
                             video_fps, video_size, isColor=True)

    accum_time = 0
    idx = -1
    while True:
        idx += 1
        res, frame = camera.read()
        if not res:
            break
        
        start = time.time()

        I1 = (frame.astype(float) - alpha * W)/(1-alpha)
        I1[I1<0] = 0
        I1[I1>255] = 255
        I1 = I1.astype(np.uint8)

        fI = I1.copy()
        fI[box[0]:box[2],box[1]:box[3],:] = cv.medianBlur(fI[box[0]:box[2],box[1]:box[3],:], 5)
        I1[bg!=0] = fI[bg!=0]

        accum_time += time.time() - start
        print("frame:", idx,"fps:", (idx+1)/accum_time)

        vwriter.write(I1)
        # break
    camera.release()
    vwriter.release()

import os
if False:
    i = 0
    dirname = "/database/水印视频/clipcanvas/"
    for root, dirs, names in os.walk(dirname):
        if dirs==[]:
            os.makedirs(root.replace("水印视频", "去水印视频"), exist_ok=True)
            for name in names:
                video = root+"/"+name
                out_video = root.replace("水印视频", "去水印视频")+"/"+name
                print(video)
                process_video(video, out_video)


