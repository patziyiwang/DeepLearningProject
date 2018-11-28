
# coding: utf-8

# In[1]:


import cv2


# In[3]:


video_name = '../data/Car4/Car4.avi'
video_name_tracked = './tracked_Car4.avi'

# In[2]:


# Set up tracker.
# tracker = cv2.TrackerTLD_create() #good for car4
tracker = cv2.TrackerCSRT_create() # very good for car4 and car 1


# In[4]:


video = cv2.VideoCapture(video_name)


# In[5]:


# Read first frame.
ok, frame = video.read()
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
video_tracked = cv2.VideoWriter(video_name_tracked, fourcc, 20, (width,height))
# In[7]:


# Define an initial bounding box
# bbox = (23,88,66,55)#car1
bbox = (70,51,107,87)#car4



# In[8]:


# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


# In[11]:


while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
    else :
        pass
        # Tracking failure
        # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
#     cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    video_tracked.write(frame)
    cv2.imshow("Tracking", frame)


    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

cv2.destroyAllWindows()
video.release()
