import numpy as np
import cv2
import os

def distance(v1,v2):
    return np.sqrt(((v2-v1)**2).sum())
def knn(train,test,k=5):
    print("once")
    dist=[]
    print(train[0,-1])
    print(train[1,-1])
    print(train.shape[0])
    for i in range(train.shape[0]):
        print(i)
        ix=train[i,:-1]
        iy=train[i,-1]
        # print(iy)
        d=distance(test,ix)
        print([d,iy])
        dist.append([d,iy])
    dk=sorted(dist,key=lambda x:x[0])[:k]
    print(dk)
    labels=np.array(dk)[:,-1]
    print(labels)
    output=np.unique(labels,return_counts=True) #([0,1,2,....labels haru],[2,4,7,.. kati choti],dtype)
    index=np.argmax(output[1]) #kati choti wala ko bata max nikalcha ani tesko index return garcha
    print(index) 
    return (output[0][index]) #tei index ko label

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_dataset_path="./face_dataset/"
face_data=[]
labels=[] 
face_id=0
names={}

for filename in os.listdir(face_dataset_path):
    if filename.endswith('.npy'):
        names[face_id]=filename[:-4]
        
        face_data_item=np.load(face_dataset_path+filename)
        print(face_data_item.shape)
        face_data.append(face_data_item)
        
        target=face_id*np.ones((face_data_item.shape[0],))
        print(target)
        labels.append(target)
        
        face_id +=1
print(face_data)
face_dataset=np.concatenate(face_data,axis=0) 
print(face_dataset)
#face data will be stacked vertically  
# The result of this line of code, face_dataset, will be a single NumPy array 
# that contains the data from all the individual arrays in face_data stacked on top of each other. 
# The data from each array in face_data is combined into rows of the face_dataset array.
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
print(face_labels)
print(face_dataset.shape)
print(face_labels.shape)
        
trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
print(trainset[0,:-1])
print(trainset[1,:-1])

while (cap.isOpened()):
    ret,frame=cap.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    

    # cv2.imshow("Frame",gray_frame)
    # print(len(faces))
    for face in faces:
        x,y,w,h=face
        offset=5
        face_offset=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection=cv2.resize(face_offset,(100,100))
        # print(face_selection.flatten())
        # print(face_selection.flatten()[0,:-1])
        # print(face_selection.shape)
        # print("2nd")
        # print(face_selection.flatten().shape)
        out=knn(trainset,face_selection.flatten())
        cv2.putText(frame,names[(out)],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
        # cv2.putText(frame,"fdghj",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("s",frame)

    key_pressed=cv2.waitKey(1)
    if key_pressed ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
