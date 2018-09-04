import cv2 as cv
import numpy as np
import time
import os


def get_alpha_W_white():
	video = "videos/stock-white.mp4"
	cap = cv.VideoCapture(video)
	res, waterpic = cap.read()
	# cv.imshow("waterpic", waterpic)

	waterpic = waterpic.min(axis=-1)
	# print(np.bincount(waterpic.reshape(-1), minlength=256))

	waterpic = waterpic.astype(np.float)
	waterpic[waterpic>247] = 0
	waterpic[waterpic!=0] = 223.5
	# cv.imshow("waterpic2", waterpic.astype(np.uint8))

	alpha_0 = 32/255
	alpha = np.ones(waterpic.shape) * alpha_0
	alpha[waterpic==0] = 0

	J = waterpic
	W = np.zeros(waterpic.shape, dtype=np.float)
	mask = waterpic!=0
	W[mask] = (J[mask] - (1 - alpha[mask])*255)/alpha[mask]
	# print(J[mask][0], (1 - alpha[mask][0])*255)
	# print(W[W<0])
	# print(W[W>255])
	# print(W[W!=0][0])
	# print(alpha[mask][0]*W[mask][0])
	# cv.imshow("white   alpha,W",np.vstack([alpha*255, W]).astype(np.uint8))
	# test_img1 = ((1-alpha)*np.zeros(W.shape) + alpha*W).astype(np.uint8)
	# test_img2 = ((1-alpha)*np.ones(W.shape)*255 + alpha*W).astype(np.uint8)
	# cv.imshow("test_img", np.vstack([test_img1,test_img2]))

	# cv.waitKey(0)

	alpha = np.repeat(alpha, 3, axis=-1).reshape(*(alpha.shape),3)
	W = np.repeat(W, 3, axis=-1).reshape(*(W.shape),3)
	return alpha, W

def get_alpha_W_black():
	video = "videos/stock-black.mp4"
	cap = cv.VideoCapture(video)
	res, waterpic = cap.read()
	waterpic = waterpic.max(axis=-1)
	# print(np.bincount(waterpic.reshape(-1),minlength=256))
	# cv.imshow("waterpic", waterpic)

	alpha_0 = 31.5/255
	thresh = 5
	waterpic[waterpic<thresh] = 0
	waterpic[waterpic>=thresh] = 31.5

	alpha = np.ones(waterpic.shape) * alpha_0
	alpha[waterpic==0] = 0
	W = waterpic/alpha_0
	# cv.imshow("black  alpha,W",np.vstack([alpha*255, W]).astype(np.uint8))

	# test_img1 = ((1-alpha)*np.zeros(W.shape) + alpha*W).astype(np.uint8)
	# test_img2 = ((1-alpha)*np.ones(W.shape)*255 + alpha*W).astype(np.uint8)
	# cv.imshow("test_img", np.vstack([test_img1,test_img2]))
	# cv.waitKey(0)

	alpha = np.repeat(alpha, 3, axis=-1).reshape(*(alpha.shape),3)
	W = np.repeat(W, 3, axis=-1).reshape(*(W.shape),3)
	return alpha, W

def move_box(alpha, W, offset):
	pos = alpha.nonzero()
	box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
	new_alpha = np.zeros(alpha.shape, dtype=alpha.dtype)
	new_W =  np.zeros(alpha.shape, dtype=W.dtype)
	box_2 = [box[i]-offset[i%2] for i in range(4)]
	idx1 = np.ix_(range(box[0],box[2]), range(box[1], box[3]))
	idx2 = np.ix_(range(box_2[0],box_2[2]), range(box_2[1], box_2[3]))
	new_alpha[idx2] = alpha[idx1]
	new_W[idx2] = W[idx1]
	return new_alpha, new_W

def merge_W(alpha1,W1,alpha2,W2):
	# alpha1 是外轮廓，外接矩形比较大
	alpha1, W1 = move_box(alpha1, W1, [0,-2])
	alpha2, W2 = move_box(alpha2, W2, [4,0])

	alpha = alpha1 + alpha2
	alpha[alpha>alpha2.max()] = alpha2.max()

	W = W1 + W2
	W[W>W1.max()] = W1.max()

	# cv.imshow("alpha,W",np.vstack([alpha*255, W]).astype(np.uint8))
	# cv.waitKey(0)

	return alpha, W

def find_offset(img1, img2):
	# 找到img1中的水印需要移动多少才能刚好与img2重合

	img1 = cv.cvtColor(img1.astype(np.uint8), cv.COLOR_BGR2GRAY)
	pos = img1.nonzero()
	box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
	idx1 = np.ix_(range(box[0],box[2]), range(box[1], box[3]))
	img1 = img1[idx1]
	# cv.imshow("img1",img1.astype(np.uint8))

	box2 = [box[0]-10, box[1]-10, box[2]+10, box[3]+10]
	idx2 = np.ix_(range(box2[0],box2[2]), range(box2[1], box2[3]))
	img2 = img2[idx2]
	img2 = cv.cvtColor(img2.astype(np.uint8), cv.COLOR_BGR2GRAY)
	# cv.imshow("img2",img2.astype(np.uint8))

	# img3 = cv.filter2D(img2.astype(np.float), -1, img1.astype(np.float))
	img3 = cv.matchTemplate(img2,img1,cv.TM_CCORR_NORMED)
	# print(img1.shape,img2.shape,img3.shape)
	cent = [img3.shape[0]//2, img3.shape[1]//2]
	img3 = img3[cent[0]-10:cent[0]+10, cent[1]-10:cent[1]+10]
	maxindex = img3.argmax()
	row, col = maxindex//img3.shape[1], maxindex%img3.shape[1]
	print((row,col))
	assert img3[row,col]==img3.max()

	# cv.imshow("f2d", img3/img3.max())
	# cv.waitKey(0)
	return [10-row, 10-col]


def blur_mask(alpha, W):
	alpha,W = move_box(alpha, W, [-4,0])
	video = "videos/stock-white.mp4"
	video = "/database/水印视频/shutterstock/多旋翼无人机/stock-footage-drone-landing-in-hand.mp4"
	cap = cv.VideoCapture(video)
	pos = alpha.max(axis=-1).nonzero()
	box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
	difs = []
	if False:
		index = 0
		while True:
			result, frame = cap.read()
			if not result: break
			index += 1

			I = (frame.astype(float) - alpha * W)/(1-alpha)
			# cv.imshow("sub result",(frame.astype(float) - alpha * W).astype(np.uint8))
			# print(frame[alpha!=0].min(), frame[alpha!=0].max(), )
			# print(I.min(),I.max(),alpha.shape,len(I[I<0]), np.mean(I[I<0]), len(I[I>255]), np.mean(I[I>255]))
			# print(frame[I>255][0:2], alpha[I>255][0:2], W[I>255][0:2], I[I>255][0:2])

			I[I>255]=255
			I[I<0] = 0
			I = I.astype(np.uint8)
			fI = I.copy()
			fI[box[0]:box[2],box[1]:box[3],:] = cv.medianBlur(fI[box[0]:box[2],box[1]:box[3],:], 5)

			oboximage = frame[box[0]:box[2],box[1]:box[3],:]
			iboximage = I[box[0]:box[2],box[1]:box[3],:]
			fboximage = fI[box[0]:box[2],box[1]:box[3],:]

			dif = np.abs(fboximage.astype(float)-iboximage.astype(float))
			difs.append(dif)

			# cv.rectangle(I,(box[1],box[0]),(box[3],box[2]),(0, 255, 255),1,8)

			# cv.imshow("diffenence", np.vstack([oboximage,iboximage,fboximage,
			#   dif.astype(np.uint8),np.median(np.array(difs),axis=0).astype(np.uint8)]))
			# cv.waitKey(0)

			# if index==100:break
		np.save("med.npy", np.array(difs))

	difs = np.load("med.npy")

	difs = difs.astype(np.uint8)
	if False:
		med = np.zeros(difs[0].shape[0:2], dtype=np.uint8)
		for i in range(med.shape[0]):
			for j in range(med.shape[1]):
				med[i,j] = np.bincount(difs[:,i,j,:].reshape(-1),minlength=256).argmax()
	else:
		med = np.median(difs,axis=0).astype(np.uint8)
		med = med.max(axis=-1)
		med[med<3]=0
	print(med.shape)
	med = np.repeat(med, 3, axis=-1).reshape(*(med.shape),3)

	cv.imwrite("imgs/shutterstock_difference.png",med)
	# cv.imshow("dif", med)

	bg = np.zeros(W.shape)
	bg[box[0]:box[2],box[1]:box[3],:] = med

	# bg,_ = move_box(bg,bg,[0,-2])
	# cv.imshow("bg", bg.astype(np.uint8)+(alpha*100).astype(np.uint8))
	# cv.waitKey(0)
	return bg


def process_video(video, alpha, W, bg, out_video=None):
	cap = cv.VideoCapture(video)
	if out_video is not None:
		video_FourCC= int(cap.get(cv.CAP_PROP_FOURCC))
		video_fps   = cap.get(cv.CAP_PROP_FPS)
		video_size  = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
					   int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
		vwriter = cv.VideoWriter(out_video, video_FourCC,
								 video_fps, video_size, isColor=True)
	else:
		select = set(range(1,1000,80))

	accum_time = 0
	index = 0
	while True:
		res, frame = cap.read()
		if not res:break
		start = time.time()

		if frame.shape!=(336,596,3):
			break
		
		index += 1
		if index==1:
			offset = find_offset(W, frame)
			print(offset)
			# offset = [-4,-2]
			alpha, W = move_box(alpha, W, offset)
			bg, _ = move_box(bg, bg, offset)

			pos = alpha.nonzero()
			box = [pos[0].min(), pos[1].min(), pos[0].max(), pos[1].max()]
			ROI = np.ix_(range(box[0]-3,box[2]+3), range(box[1]-3,box[3]+3))
			aW = (alpha*W)[ROI]
			a1 = (1/(1 - alpha))[ROI]
			med = bg[ROI]
			# print(aW.shape)

		if out_video is None and index not in select:continue

		org_frame = frame.copy()
		frame = frame.astype(np.float)
		J = frame[ROI]
		I = (J - aW) * a1
		I[I<0] = 0
		I[I>255] = 255
		frame[ROI] = I

		sub_frame = frame.copy().astype(np.uint8)

		org_fI = I.copy()
		fI = I.copy()
		fI = cv.medianBlur(fI.astype(np.uint8), 5)
		I[med!=0] = fI[med!=0]
		I[I<0] = 0
		I[I>255] = 255
		frame[ROI] = I
		frame = frame.astype(np.uint8)

		accum_time += time.time() - start
		print("frame:", index,"fps:", index/accum_time)

		# cv.imwrite("imgs/shutterstock_median_blur.png",fI)
		# cv.imwrite("imgs/shutterstock_median_subs.png",org_fI.astype(np.uint8))
		# cv.imwrite("imgs/shutterstock_median_res.png",I.astype(np.uint8))

		# cv.imwrite("imgs/shutterstock_orig.png",org_frame)
		# cv.imwrite("imgs/shutterstock_subs.png",sub_frame)
		# cv.imwrite("imgs/shutterstock_res.png",frame)

		cv.imshow("frame", np.vstack([org_frame,frame]))
		cv.waitKey(0)
		if out_video:vwriter.write(frame)
		
	cap.release()
	if out_video:vwriter.release()

def main():
	alpha1, W1 = get_alpha_W_black()
	alpha2, W2 = get_alpha_W_white()
	alpha, W = merge_W(alpha1,W1,alpha2,W2)
	bg = blur_mask(alpha, W)

	# cv.imwrite("imgs/shutterstock_alpha.png", (alpha*255).astype(np.uint8))
	# cv.imwrite("imgs/shutterstock_W.png", (W).astype(np.uint8))

	video = "/database/水印视频/shutterstock/多旋翼无人机/stock-footage-drone-landing-in-hand.mp4"
	process_video(video, alpha, W, bg)
	return 

	dirname = "/database/水印视频/shutterstock/"
	for root, dirs, names in os.walk(dirname):
		if dirs==[]:
			os.makedirs(root.replace("水印视频", "去水印视频"), exist_ok=True)
			for name in names:
				video = root+"/"+name
				out_video = root.replace("水印视频", "去水印视频")+"/"+name
				print(video)
				if not os.path.exists(out_video):
					process_video(video, alpha, W, bg, out_video)

if __name__ == '__main__':
	main()
