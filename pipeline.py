#_______________________________________________________________________________
# pipeline.py                                                              80->|

import cv2
import glob
import pickle
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import ImageFont, ImageDraw, Image


#_______________________________________________________________________________
# Circular queuing class made with Numpy

class NQ():
    def __init__(self,qlen,shape,dtype=np.uint8):
        self.data= np.empty(shape=shape,dtype=dtype)
        self.qlen= qlen
    def put(self,a):
        self.data= np.append(self.data,a, axis=0)
        if len(self.data)>self.qlen: self.drop()
    def peek(self):
        return self.data[len(self.data)-1]
    def drop(self):
        self.data= np.delete(self.data, 0, axis=0)
    def getAvg(self):
        newList= np.copy(self.data[0])
        newList[:,1]= np.sum(self.data[:,:,1], axis=0)/float(len(self.data))
        return newList


#_______________________________________________________________________________
# Control class

class Executive():
    frameCount= 0
    diagScreen= None
    def __init__(self, video=False, diag=False):
        self.diags= diag
        self.video= video
        self.debugger= open('output_images/trace.txt', 'w')
        self.debugger.write('Debug logger\n')

    def reset(self):
        self.frameCount= 0
        self.debugger.close()


#_______________________________________________________________________________
# Line management class

class Line():
    def __init__(self, qlen=10, yoffset=209):
        self.qlen= qlen                       # queue length
        line= []
        for i in range(12):                   # Each line has 12 points
            line.append([i*114+14, yoffset])  # Each point has an x,y pair
        self.pointsQ= NQ(qlen, (0,12,2))      # Points queue
        self.pointsQ.put([ line ])
        self.fit= [0,0,yoffset]               # Init with a straight line
        
    # Return a y value as a function of x based on average fit coefficients
    def lineF(self,x): 
        fit= self.fit
        return fit[0]*x**2 + fit[1]*x + fit[2]
    
    def curveF(self,x): # return radius of curvature at point x
        fit= self.fit
        return ((1 + (2*fit[0]*x + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    
    def getAvgPoints(self): # Return an array of average points on a line
        return self.pointsQ.getAvg()
    
    def getPrevPoints(self): # Return an array of previously enqueued points
        return self.pointsQ.peek()
    
    def putPoints(self, line):
        self.pointsQ.put([ line ])
    
    def fitToPoly(self):  # Fit pointsQ to a polynomial and save coefficients
        avgPoints= self.pointsQ.getAvg()
        self.fit= np.polyfit(avgPoints[:,0],avgPoints[:,1], 2)


#_______________________________________________________________________________
# The Model class creates a left, center, and right line

class Model():
    curRadius= 0
    camY= 275
    def __init__(self, width=160):
        self.leftLine= Line(3, self.camY-width//2)
        self.centerLine= Line(15, self.camY)
        self.rightLine= Line(3, self.camY+width//2)
        self.widthQs= []    # Width queues for each right-side patch
        for i in range(12): # longer queue on the widths to hold through bridges
            self.widthQs.append(NQ(4, (0,1), np.uint16))
                            # initialize with the starting width
            self.widthQs[i].put([ [width] ])


#_______________________________________________________________________________
# Filter on range of white

def filterW(img, thresh=200):
    lower= np.array([thresh, thresh, thresh], dtype="uint8") 
    upper= np.array([255, 255, 255], dtype="uint8")
    mask= cv2.inRange(img, lower, upper)
    return mask


#_______________________________________________________________________________
# Filter on range of yellow

def filterY(img, thresh=146):
    lower= np.array([thresh, thresh, 0], dtype="uint8") 
    upper= np.array([255, 209, 130], dtype="uint8")
    mask= cv2.inRange(img, lower, upper)
    return mask


#_______________________________________________________________________________
# Convert to HLS color space and separate the S channel and filter by thresholds

def thresholdPatch(img, primary, idx, s_thresh=(200, 255)):
    if primary:
        hlsImg= cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        S_channel= hlsImg[:,:,2]
        _,s_binary= cv2.threshold(S_channel.astype('uint8'), s_thresh[0], s_thresh[1], cv2.THRESH_BINARY)
        
        # RULE: if nothing found and idx>4, then use filterY
        if np.sum(s_binary)==0 and idx>4:
            s_binary= filterY(img,110)
    else:
        s_binary= filterW(img,s_thresh[0])

        # RULE: if nothing found and idx>2, then try lower threshold
        if np.sum(s_binary)==0 and idx>2:
            s_binary= filterW(img,s_thresh[0]-20)
    return s_binary

#_______________________________________________________________________________
# Transform from perspective view to overhead view and vice versa

src= np.float32([[557,460],[557+166,460],[84+1112,670],[84,670]])
xoffset, yoffset= 25, 156
width, height= 1248, 237
dst= np.float32([[xoffset+width,yoffset+0],[xoffset+width,yoffset+height],[xoffset,yoffset+height],[xoffset,yoffset+0]])

def makeOverhead(img):
    global src, dst
    img_size= (img.shape[1], img.shape[0])
    M= cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)  #INTER_NEAREST

def makePerspective(img):
    global src, dst
    img_size= (img.shape[1], img.shape[0])
    M= cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


#_______________________________________________________________________________
# Display filter patches into diagnostics window

def diagPatch(primary,patch,idx):
    global executive
    if primary:
        if np.sum(patch)==0:
            miss= np.ones_like(patch).astype(np.uint8)
            executive.diagScreen[77-idx*7:77-idx*7+7,0:58,:]= np.rot90(np.dstack(( miss*63, miss*0, miss*0)))
        else:
            executive.diagScreen[77-idx*7:77-idx*7+7,0:58,:]= np.rot90(np.dstack(( patch, patch, patch)))
    else:
        if np.sum(patch)==0:
            miss= np.ones_like(patch).astype(np.uint8)
            executive.diagScreen[77-idx*7:77-idx*7+7,58:58*2,:]= np.rot90(np.dstack(( miss*63, miss*0, miss*0)))
        else:
            executive.diagScreen[77-idx*7:77-idx*7+7,58:58*2,:]= np.rot90(np.dstack(( patch, patch, patch)))


#_______________________________________________________________________________
# Scan patch for signs of a line and return a potential y and confidence ranking

def processFilter(primary,patch,idx):
    if primary:
        patchT= thresholdPatch(patch,primary,idx,(100,255)) #threshold patch
    else:
        patchT= thresholdPatch(patch,primary,idx,(180,255)) # threshold patch for white values
    diagPatch(primary,patchT,idx)
    histogram= np.convolve(np.ones(5).astype(np.uint16),np.sum(patchT/7, axis=1),mode='same')

    maybeY= np.round(np.argmax(histogram))
    #Maximum confidence is 1275= 255*5 (a stack of 5 white pixels)
    return histogram[maybeY]/1275, maybeY

#_______________________________________________________________________________
# Locate and update leftLine and rightLine in overhead xy space

def locateLines(img):
    global model,executive
    
    # generate diag screen to display patch diagnostics
    executive.diagScreen= np.zeros_like(img).astype(np.uint8) 
    
    # Extract average leftPoints from leftLine
    leftPoints= model.leftLine.getPrevPoints()
    for idx,i in enumerate(leftPoints):
        if executive.frameCount>5: # RULE: filter extreme curvatures
            if idx>0 and abs(i[1]-leftPoints[idx-1][1])>7:
                i[1]= leftPoints[idx-1][1]
        conf,y= processFilter(True,img[i[1]-29:i[1]+29, i[0]:i[0]+7],idx)
        if conf>.6:
            yCandidate= i[1]-29+y
            if executive.frameCount>25:
                if abs(yCandidate-i[1])<10: # RULE: filter outliers
                    i[1]= yCandidate
            else:
                i[1]= yCandidate
    model.leftLine.putPoints(leftPoints)
    
    # Extract rightPoints offset by dist to center from leftPoints
    rightPoints= np.copy(leftPoints)
    rightPoints[:,1]= leftPoints[:,1]+(model.centerLine.getAvgPoints()[:,1]-leftPoints[:,1])*2
    for idx,i in enumerate(rightPoints):
        if executive.frameCount>5: # RULE: filter extreme curvatures
            if idx>0 and abs(i[1]-rightPoints[idx-1][1])>7:
                i[1]= rightPoints[idx-1][1]
        conf,y= processFilter(False,img[i[1]-29:i[1]+29, i[0]:i[0]+7],idx)
        if conf>.6:
            yCandidate= i[1]-29+y
            if executive.frameCount>25:
                if abs(yCandidate-i[1])<15+int(round(14*idx/11)): # RULE: filter outliers
                    i[1]= yCandidate
                    model.widthQs[idx].put([ [int(round(i[1]-leftPoints[idx][1]))] ])
            else:
                i[1]= yCandidate
        else: # RULE: right lane is average distance from left lane
            avgPrevLaneWidth= np.sum(model.widthQs[idx].data)/float(len(model.widthQs[idx].data))
            i[1]= leftPoints[idx][1]+avgPrevLaneWidth
    model.rightLine.putPoints(rightPoints)
    
    # Adjust center line
    centerPoints= model.centerLine.getAvgPoints()
    comb= np.add(rightPoints[:,1],leftPoints[:,1])
    centerPoints[:,1]= np.round(comb/2).astype(np.int)  #update y values
    model.centerLine.putPoints(centerPoints)


#_______________________________________________________________________________
# Draw in overhead space, the found lane delimited by the lines

def drawLane(img):
    global executive,model
    
    base= np.zeros_like(img).astype(np.uint8) # generate image on which to draw
    if executive.video:
        laneColor= (0,0,255)
        centerColor= (0,150,255)
    else:
        laneColor= (255,0,0)
        centerColor= (255,150,0)
    
    # Fit left line to a 2nd order polynomial
    model.leftLine.fitToPoly()
    leftPoints= np.copy(model.leftLine.pointsQ.data[0]) # Copy the first element to use x values
    for i in leftPoints:
        i[1]= model.leftLine.lineF(i[0])+6

    # Fit center line to a 2nd order polynomial
    model.centerLine.fitToPoly()
    avgCenterO= np.copy(model.centerLine.pointsQ.data[0]) # Copy the first element to use x values
    for i in avgCenterO:
        i[1]= model.centerLine.lineF(i[0])-2 # offset to the left of center by 2 pixels

    # Fit right line to a 2nd order polynomial
    model.rightLine.fitToPoly()
    rightPoints= np.copy(model.rightLine.pointsQ.data[0]) # Copy the first element to use x values
    for i in rightPoints:
        i[1]= model.rightLine.lineF(i[0])-6

    if executive.diags: # for diagnostic view, show center line
        pointStack= np.hstack((np.array([leftPoints]), np.array([ np.flipud( avgCenterO )]) ))
        cv2.fillPoly(base, np.int_([pointStack]), laneColor) # draw the lane onto the overhead blank image

        avgCenterO[:,1]= avgCenterO[:,1]+4 # offset to the right of center
        pointStack= np.hstack((np.array([avgCenterO]), np.array([ np.flipud( rightPoints )]) ))
        cv2.fillPoly(base, np.int_([pointStack]), laneColor) # draw the lane onto the overhead blank image
    else:
        pointStack= np.hstack((np.array([leftPoints]), np.array([ np.flipud( rightPoints )]) ))
        cv2.fillPoly(base, np.int_([pointStack]), laneColor) # draw the lane onto the overhead blank image

    # paint the line points
    if executive.diags: 
        if executive.video:
            spotColor= (0,150,255)
        else:
            spotColor= (255,150,0)
        leftPoints= model.leftLine.getPrevPoints()
        rightPoints= model.rightLine.getPrevPoints()
        for i in leftPoints:
            cv2.rectangle(base, tuple([i[0]-2,i[1]-29]), tuple([i[0]+16,i[1]+29]), spotColor, -1)  #CV_FILLED
        for i in rightPoints:
            cv2.rectangle(base, tuple([i[0]-2,i[1]-29]), tuple([i[0]+16,i[1]+29]), spotColor, -1)
    overlay= makePerspective(base) # warp the overhead to perspective space
    result= cv2.addWeighted(img, 1, overlay, 0.5, 0) # annotate the original
    return result


#_______________________________________________________________________________
# Compute radius using OpenCV

def radius(x1,y1,  x2,y2,  x3,y3):
    c,r=cv2.minEnclosingCircle(np.array([[x1,y1],  [x2,y2],  [x3,y3]], dtype=np.float32))
    return c[1]<0,r # also return true if center is to the left


#_______________________________________________________________________________
# Add computed data to the annotated screen using PIL

def drawData(img):
    global executive,model
    radiusMeters= 0
    
    # Draw semi-transparent base on which to place text
    pImg= Image.fromarray(img).convert('RGBA')
    rImg= Image.new('RGBA', pImg.size, (0,0,0,0))
    draw= ImageDraw.Draw(rImg) # get a drawing context
    font= ImageFont.truetype('OpenSans-Regular.ttf', 35)
    fullWhite= (255,255,255,255)
    if executive.video:
        attrColor= (100,195,36,255)
    else:
        attrColor= (36,195,100,255)
    x,y= 32,661
    if executive.diags:
        draw.rectangle([0,0, 58*2,7*12],fill=(32,32,32,127))
    draw.rectangle([  0,630, 316,720],fill=(32,32,32,232))
    draw.rectangle([321,630, 637,720],fill=(32,32,32,232))
    draw.rectangle([642,630, 958,720],fill=(32,32,32,232))
    draw.rectangle([964,630,1280,720],fill=(32,32,32,232))
    draw.polygon([x+0,y+26.311, x+7.594,y+13.156, x+0,y+0, x+8.064,y+0, x+15.658,y+13.156, x+8.064,y+26.311], fill=attrColor)

    draw.text((70,648), 'width', font=font,fill=attrColor)
    avgPrevLaneWidth= np.sum(model.widthQs[0].data)/float(len(model.widthQs[0].data))
    draw.text((184,648), str(int(round(avgPrevLaneWidth*3.6576/1.6))/100), font=font,fill=fullWhite)
    
    draw.text(( 343,648), 'drift', font=font,fill=attrColor)
    draw.text(( 659+11,648), 'radius', font=font,fill=attrColor)
    draw.text(( 990,648), 'frame', font=font,fill=attrColor)
    draw.text((1104,648), str(executive.frameCount), font=font,fill=fullWhite)
    
    # Compute curvature radius along the polynomial (48' behind to 48' ahead)
    if executive.frameCount%20==0:
        left,radiusPx= radius(-640,model.centerLine.lineF(-640), 0,model.centerLine.lineF(0), 640,model.centerLine.lineF(640))
        radiusMeters= int(round(radiusPx*3.6576/160))
        model.curRadius= radiusMeters
        if left:
            model.curRadius= -model.curRadius
    if radiusMeters>10000 or model.curRadius==0: # Straight section found
        model.curRadius= 0
        draw.text((781+11,648), '9999m', font=font,fill=fullWhite)
    else:
        draw.text((781+11,648), str(model.curRadius)+'m', font=font,fill=fullWhite)

    # Compute and display lane drift in centimeters
    centerCM= int(round((model.camY-model.centerLine.lineF(0))*36.576/16))
    draw.text((423,648), str(centerCM)+'cm', font=font,fill=fullWhite)
    
    del draw # free up the context
    pImg= Image.alpha_composite(pImg, rImg)
    annotated= np.array(pImg.convert('RGB'))
    if executive.diags:
        annotated= cv2.addWeighted(annotated, 1, executive.diagScreen, 1, 0)
    return annotated


#_______________________________________________________________________________
# Additional filters for future work on challenge videos

def expandUpper(img,boost=1):
    img= img.astype(np.int16) -127
    img= img.clip(0)
    imgNew= ((img*2)*boost).clip(0,255)
    return imgNew.astype(np.uint8)

def sobelY(img, threshL=20, threshU=100):
    sobely= cv2.Sobel(img, cv2.CV_16S, 0, 1) # run the derivative in y
    abs_sobely= np.absolute(sobely) # absolute y derivative to accentuate horizontal lines 
    scaled_sobel= np.uint8(255*abs_sobely/np.max(abs_sobely))
    binary= np.zeros_like(img)
    binary[(scaled_sobel >threshL) & (scaled_sobel <= threshU)]= 255 # threshold y gradient
    return binary

def prefilterImg(img):
    imgE= expandUpper(img, boost=1)
    imgGray= cv2.cvtColor(imgE,cv2.COLOR_RGB2HLS)[:,:,1]
    imgB= sobelY(imgGray,20,68)
    return np.dstack(( imgB,imgB,imgB ))


#_______________________________________________________________________________
# The processing pipeline to analyze and annotate an image

def process(img):
    global executive,mtx,dist

    # Undistort and convert to overhead space
    output= cv2.undistort(img, mtx, dist, None, mtx)
    warped= makeOverhead(output)
    locateLines(warped)
    withLane= drawLane(output)
    annotated= drawData(withLane)

    executive.frameCount += 1
    return annotated


#_______________________________________________________________________________
# Process a video stream

def procVideo(fileName):
    clip= VideoFileClip(fileName)
    imgName= fileName.split('/')[1]
    project_video_output= 'output_images/'+imgName
    print('Processing video...')
    project_video_clip= clip.fl_image(process)
    project_video_clip.write_videofile(project_video_output, audio=False)


#_______________________________________________________________________________
# Executive

calibration_data= pickle.load(open("dist_pickle.p", "rb"))
mtx= calibration_data["mtx"]
dist= calibration_data["dist"]
executive= Executive(video=True, diag=True)
model= Model(width=160)
procVideo('video/project_video.mp4')
executive.reset()