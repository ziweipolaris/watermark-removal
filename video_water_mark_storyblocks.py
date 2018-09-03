import cv2 as cv
import numpy as np

video = "videos/storyblocks.mp4"
cap = cv.VideoCapture(video)
res, waterpic = cap.read()
mask = (waterpic>25)
alpha_0 = 1.1* waterpic[mask!=0].mean()/255
alpha = np.ones(mask.shape) * alpha_0
alpha[mask==0]=0
print(waterpic.max(), waterpic.min(), alpha_0)
# cv.imshow("mask0", mask.astype(float))

box = np.nonzero(mask)
box = [min(box[0]),min(box[1]),max(box[0]),max(box[1])]
box[0]-=5
box[1]-=5
box[2]+=5
box[3]+=5
print(box)
index = 0
sel = set(range(1,1000,50))

video = "/database/水印视频/storyblocks/多旋翼无人机/aerial-rc-helicopter_ey-9zj5ux__PM.mp4"
video = "/database/水印视频/storyblocks/多旋翼无人机/videoblocks-slow-motion-drone-takeoff_b3sv8vzub__SB_PM.mp4"
video = "/database/水印视频/storyblocks/运输机旅客机/4k-air-berlin-boeing-737-arriving-madeira-from-seascape_E1Ni7uQSe__SB_PM.mp4"
cap = cv.VideoCapture(video)

if False:
    difs = []
    while True:
        result, frame = cap.read()
        if not result: break
        index += 1

        # if index in sel:
        I = (frame.astype(float) - alpha * mask * 255)/(1-alpha)
        I[I<0] = 0
        I = I.astype(np.uint8)
        fI = I.copy()
        fI[box[0]:box[2],box[1]:box[3],:] = cv.medianBlur(fI[box[0]:box[2],box[1]:box[3],:], 5)

        oboximage = frame[box[0]:box[2],box[1]:box[3],:]
        iboximage = I[box[0]:box[2],box[1]:box[3],:]
        fboximage = fI[box[0]:box[2],box[1]:box[3],:]

        dif = fboximage.astype(float)-iboximage.astype(float)
        dif[dif<30]=0
        difs.append(dif)

            # cv.rectangle(I,(box[1],box[0]),(box[3],box[2]),(0, 255, 255),1,8)
            # cv.imshow("diffenence", np.vstack([oboximage,iboximage,fboximage,
            #   dif.astype(np.uint8),np.median(np.array(difs),axis=0).astype(np.uint8)]))
            # cv.waitKey(0)
        # if index==100:break
    np.save("med.npy", np.array(difs))
# exit()

difs = np.load("med.npy")
med = np.median(difs,axis=0).astype(np.uint8)
tmp = med.max(axis=2)
med[:,:,0] = tmp
med[:,:,1] = tmp
med[:,:,2] = tmp

bg = np.zeros(mask.shape)
bg[box[0]:box[2],box[1]:box[3],:] = med

# cv.imshow("dif", med)
# cv.imshow("bg", bg)

mask2 = mask.copy()
mask2[bg!=0] = 0
alpha2 = alpha.copy()
alpha2[bg!=0] = 0.1*alpha_0


if False:
    video = "/database/水印视频/storyblocks/多旋翼无人机/aerial-rc-helicopter_ey-9zj5ux__PM.mp4"
    video = "/database/水印视频/storyblocks/多旋翼无人机/videoblocks-slow-motion-drone-takeoff_b3sv8vzub__SB_PM.mp4"
    video = "/database/水印视频/storyblocks/运动飞机/4359-small-airplane-aircraft-aviation-landing-mountain-travel-trip-sunny-da_nkegz2_t__SB_PM.mp4"
    cap = cv.VideoCapture(video)

    index = 0
    while True:
        result, frame = cap.read()
        if not result: break
        index += 1


        if index in sel:
            I1 = (frame.astype(float) - alpha * mask * 255)/(1-alpha)
            I1[I1<0] = 0
            I1 = I1.astype(np.uint8)

            I2 = (frame.astype(float) - alpha2 * mask2 * 255)/(1-alpha2)
            I2[I2<0] = 0
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
def process_video(video,out_video):
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

        I1 = (frame.astype(float) - alpha * mask * 255)/(1-alpha)
        I1[I1<0] = 0
        I1 = I1.astype(np.uint8)

        fI = I1[box[0]:box[2],box[1]:box[3],:].copy()
        fI = cv.medianBlur(fI, 5)
        I1[box[0]:box[2],box[1]:box[3],:][med!=0] = fI[med!=0]

        accum_time += time.time() - start
        print("frame:", idx,"fps:", (idx+1)/accum_time)

        vwriter.write(I1)
        # break
    camera.release()
    vwriter.release()

import os
if False:
    dirname = "/database/水印视频/storyblocks/"
    for root, dirs, names in os.walk(dirname):
        if dirs==[]:
            os.makedirs(root.replace("水印视频", "去水印视频"), exist_ok=True)
            for name in names:
                video = root+"/"+name
                out_video = root.replace("水印视频", "去水印视频")+"/"+name
                print(video)
                process_video(video, out_video)


