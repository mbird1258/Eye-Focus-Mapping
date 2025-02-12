import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

class EyeManager:
    def __init__(self, CamOffset, CamViewDepth):
        self.CamOffset = CamOffset # cm
        self.CamViewDepth = CamViewDepth # pixels
        self.SceneManager = SceneManager(self)

    def GetEyePose(self, img1, img2, HomographyMatrix=False, debug=True):
        print("\n"+"="*30+"\n")

        # Get eye centers
        face1 = utils.FacePose(img1) # Used for eye border
        face2 = utils.FacePose(img2) # Used for eye border

        if len(face1) == 0 or len(face2) == 0: 
            return [None, None, None, None]

        LeftEyeIndexes = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        RightEyeIndexes = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        
        if debug:
            plt.imshow(img1)
            plt.show()
            plt.imshow(img1)
            plt.scatter(face1[0][LeftEyeIndexes+RightEyeIndexes], face1[1][LeftEyeIndexes+RightEyeIndexes], s=0.5)
            plt.scatter(face1[0][[468, 473]], face1[1][[468, 473]], s=0.5, c="red")
            plt.show()
        
        if debug:
            plt.imshow(img2)
            plt.show()
            plt.imshow(img2)
            plt.scatter(face2[0][LeftEyeIndexes+RightEyeIndexes], face2[1][LeftEyeIndexes+RightEyeIndexes], s=0.5)
            plt.scatter(face2[0][[468, 473]], face2[1][[468, 473]], s=0.5, c="red")
            plt.show()

        # Get eye border points
        x, y, z = [], [], []
        for inds in [LeftEyeIndexes, RightEyeIndexes]:
            x.append([])
            y.append([])
            z.append([])
            for x1, y1, x2, y2 in zip(face1[0][inds], face1[1][inds], face2[0][inds], face2[1][inds]):
                a = np.array([x1-img2.shape[1]/2, y1-img2.shape[0]/2, self.CamViewDepth[0]])
                
                b = np.array([x2, y2, 1])
                b = (HomographyMatrix@b[:, np.newaxis]).flatten()
                b = b[:-1]/b[-1]
                b = np.array([*b-np.array(img2.shape)[[1,0]]/2, self.CamViewDepth[1]])

                out, err = utils.intersection([0, 0, 0], 
                                              a, 
                                              self.CamOffset, 
                                              np.array(self.CamOffset)+b, GetError=True)
                if debug: print(f"error(cm; dist between two lines used in triangulation): {err}")
                x[-1].append(out[0])
                y[-1].append(out[1])
                z[-1].append(out[2])
                
                if debug:
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(0, 0, 0, c="red")
                    ax.scatter(*self.CamOffset, c="blue")
                    ax.scatter(*out, c="black")
                    ax.axis('scaled')
                    ax.autoscale(False)
                    ax.plot([0, a[0]/10], [0, a[1]/10], [0, self.CamViewDepth[0]/10], c="red")
                    ax.plot([self.CamOffset[0], self.CamOffset[0]+b[0]/10], [self.CamOffset[1], self.CamOffset[1]+b[1]/10], [self.CamOffset[2], self.CamOffset[2]+self.CamViewDepth[1]/10], c="blue")
                    plt.show()
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # Get iris points
        LIrisX = (face1[0][468], face2[0][468])
        LIrisY = (face1[1][468], face2[1][468])
        RIrisX = (face1[0][473], face2[0][473])
        RIrisY = (face1[1][473], face2[1][473])

        x1, x2 = LIrisX
        y1, y2 = LIrisY
        
        a = np.array([x1-img2.shape[1]/2, y1-img2.shape[0]/2, self.CamViewDepth[0]])
                
        b = np.array([x2, y2, 1])
        b = (HomographyMatrix@b[:, np.newaxis]).flatten()
        b = b[:-1]/b[-1]
        b = np.array([*b-np.array(img2.shape)[[1,0]]/2, self.CamViewDepth[1]])

        LIrisXYZ, err = utils.intersection([0, 0, 0], 
                                             a, 
                                             self.CamOffset, 
                                             np.array(self.CamOffset)+b, GetError=True)
        LIrisXYZ = np.array(LIrisXYZ)
        if debug: print(f"error(cm; dist between two lines used in triangulation): {err}")

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(0, 0, 0, c="red")
            ax.scatter(*self.CamOffset, c="blue")
            ax.scatter(*LIrisXYZ, c="black")
            ax.axis('scaled')
            ax.autoscale(False)
            ax.plot([0, a[0]/10], [0, a[1]/10], [0, self.CamViewDepth[0]/10], c="red")
            ax.plot([self.CamOffset[0], self.CamOffset[0]+b[0]/10], [self.CamOffset[1], self.CamOffset[1]+b[1]/10], [self.CamOffset[2], self.CamOffset[2]+self.CamViewDepth[1]/10], c="blue")
            plt.show()

        x1, x2 = RIrisX
        y1, y2 = RIrisY
        
        a = np.array([x1-img2.shape[1]/2, y1-img2.shape[0]/2, self.CamViewDepth[0]])
                
        b = np.array([x2, y2, 1])
        b = (HomographyMatrix@b[:, np.newaxis]).flatten()
        b = b[:-1]/b[-1]
        b = np.array([*b-np.array(img2.shape)[[1,0]]/2, self.CamViewDepth[1]])

        RIrisXYZ, err = utils.intersection([0, 0, 0], 
                                            a, 
                                            self.CamOffset, 
                                            np.array(self.CamOffset)+b, GetError=True)
        RIrisXYZ = np.array(RIrisXYZ)
        if debug: print(f"error(cm; dist between two lines used in triangulation): {err}")

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(0, 0, 0, c="red")
            ax.scatter(*self.CamOffset, c="blue")
            ax.scatter(*RIrisXYZ, c="black")
            ax.axis('scaled')
            ax.autoscale(False)
            ax.plot([0, a[0]/10], [0, a[1]/10], [0, self.CamViewDepth[0]/10], c="red")
            ax.plot([self.CamOffset[0], self.CamOffset[0]+b[0]/10], [self.CamOffset[1], self.CamOffset[1]+b[1]/10], [self.CamOffset[2], self.CamOffset[2]+self.CamViewDepth[1]/10], c="blue")
            plt.show()

        Irises = [LIrisXYZ, RIrisXYZ]

        SphereCenters = [None, None]

        # Get Eye Center
        for ind in range(2):
            print(list(utils.SphereCenter2(x[ind], y[ind], z[ind])[0]))
            SphereCenters[ind] = list(utils.SphereCenter2(x[ind], y[ind], z[ind])[1:])
        SphereCenters = np.array(SphereCenters)

        '''
        for ind in range(2):
            for indexes in combinations(range(len(x[ind])), 3):
                SphereCenters[ind].append(utils.SphereCenter(np.append(x[ind][list(indexes)], [Irises[ind][0]]), np.append(y[ind][list(indexes)], [Irises[ind][1]]), np.append(z[ind][list(indexes)], [Irises[ind][2]])))
            SphereCenters[ind] = np.array(SphereCenters[ind])

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[0], y[0], z[0], s=8, c="black")
            ax.scatter(*Irises[0], s=8, c="red")
            ax.axis('scaled')
            plt.show()

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[1], y[1], z[1], s=8, c="black")
            ax.scatter(*Irises[1], s=8, c="red")
            ax.axis('scaled')
            plt.show()

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[0], y[0], z[0], s=8, c="black")
            ax.scatter(*Irises[0], s=8, c="red")
            ax.axis('scaled')
            ax.autoscale(False)
            ax.scatter(*SphereCenters[0].T[:-1], s=2, c="blue")
            plt.show()

        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[1], y[1], z[1], s=8, c="black")
            ax.scatter(*Irises[1], s=8, c="red")
            ax.axis('scaled')
            ax.autoscale(False)
            ax.scatter(*SphereCenters[1].T[:-1], s=2, c="blue")
            plt.show()

        def InlierFunction(input, points, threshold):
            model = np.array([np.average(points[:, :-1], axis=0)]), 1.15 # average radius of eye(1.15 cm)
            inliers = input[np.logical_and(np.mean(np.abs(input[:, :-1]-model[0]), axis=1) <= threshold[0], np.abs(input[:, -1]-model[1]) <= threshold[1])]
            
            return inliers, model[0][0]

        for ind, vals in enumerate(SphereCenters):
            inliers, SphereCenter = utils.ransac(vals, InlierFunction, threshold=(0.15, 0.25), iterations=10000, samples=int(len(SphereCenters)/5)+1, axes=[0])

            if debug:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(x[ind], y[ind], z[ind], s=8, c="black")
                ax.scatter(*Irises[ind], s=8, c="red")
                ax.axis('scaled')
                ax.autoscale(False)
                ax.scatter(*SphereCenters[ind].T[:-1], c="blue", s=2, alpha=0.05)
                ax.scatter(*inliers[:, :-1].T, s=2, c="blue")
                plt.show()

            SphereCenters[ind] = SphereCenter
        '''
        
        if debug:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[0], y[0], z[0], s=8, c="black")
            ax.scatter(*Irises[0], s=8, c="red")
            ax.scatter(*SphereCenters[0], s=16, c="blue")
            ax.scatter(x[1], y[1], z[1], s=8, c="black")
            ax.scatter(*Irises[1], s=8, c="red")
            ax.scatter(*SphereCenters[1], s=16, c="blue")
            ax.scatter(*self.FocusTriangulation(SphereCenters[0], SphereCenters[1], Irises[0], Irises[1]), c="red")
            ax.axis('scaled')
            ax.autoscale(False)
            ax.plot([SphereCenters[0][0], Irises[0][0]+(Irises[0][0]-SphereCenters[0][0])*10],
                    [SphereCenters[0][1], Irises[0][1]+(Irises[0][1]-SphereCenters[0][1])*10],
                    [SphereCenters[0][2], Irises[0][2]+(Irises[0][2]-SphereCenters[0][2])*10], c="red")
            ax.plot([SphereCenters[1][0], Irises[1][0]+(Irises[1][0]-SphereCenters[1][0])*10],
                    [SphereCenters[1][1], Irises[1][1]+(Irises[1][1]-SphereCenters[1][1])*10],
                    [SphereCenters[1][2], Irises[1][2]+(Irises[1][2]-SphereCenters[1][2])*10], c="red")
            plt.show()

        SphereCenters = np.array(SphereCenters)

        return SphereCenters[0], SphereCenters[1], Irises[0], Irises[1]
    
    def FocusTriangulation(self, LEyeCenter, REyeCenter, LIris, RIris):
        return np.array(utils.intersection(LEyeCenter, LIris, REyeCenter, RIris))

    def ProcessVideo(self, vid1, StartFrame1, vid2, StartFrame2, HomographyMatrixRefPoints=False, StorageFile="StorageEyes.pkl", RefPointsStorageFile="CourtStorage.pkl", length=False):
        if HomographyMatrixRefPoints:
            cap2 = cv2.VideoCapture(vid2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)
            res2, frame2 = cap2.read()
            if not res2:
                return
            
            file = open(RefPointsStorageFile,"rb")
            CourtPoints = pickle.load(file)
            file.close()

            # Get CourtRefPoints pos on screen
            points = []
            for point in HomographyMatrixRefPoints:
                points.append((point[:-1]-np.array(self.CamOffset)[:-1])*self.CamViewDepth[1]/(point[-1]-self.CamOffset[-1])+np.array(frame2.shape)[[1,0]]/2)
            points = np.array(points)
            
            # Get homography matrix
            HomographyMatrix = utils.HomographyMatrix(*CourtPoints[1].flatten(), *points.flatten())

        plt.imshow(frame2)
        plt.show()
        plt.imshow(cv2.warpPerspective(frame2,HomographyMatrix,(frame2.shape[1], frame2.shape[0]),flags=cv2.INTER_LINEAR))
        plt.show()

        # Get eyes
        cap1 = cv2.VideoCapture(vid1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, StartFrame1-1)
        cap2 = cv2.VideoCapture(vid2)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)
        StartTime1 = (StartFrame1-1)/cap1.get(cv2.CAP_PROP_FPS)*1000

        out = []
        ind = 0
        while True:
            if length:
                if ind >= length:
                    break
            
            res1, frame1 = cap1.read()
            res2, frame2 = cap2.read()
            if not res1 or not res2:
                break
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            FrameTime = cap1.get(cv2.CAP_PROP_POS_MSEC)
            LEyeCenter, REyeCenter, LIris, RIris = self.GetEyePose(frame1, frame2, HomographyMatrix)
            if type(LEyeCenter) == np.ndarray:
                focus = self.FocusTriangulation(LEyeCenter, REyeCenter, LIris, RIris)
            else:
                focus = False
            
            out.append([LEyeCenter, REyeCenter, LIris, RIris, focus, False, False])

            ind += 1

        file = open(StorageFile, "wb")
        pickle.dump([out, False], file)
        file.close()
    
    def PlayVideo(self, StorageFile):
        file = open(StorageFile,"rb")
        self.SceneManager.MainLoop(*pickle.load(file))
        file.close()

    def GetHomographyMatrixRefPoints(self, GetValuesFromFile, vid1, StartFrame1, vid2, StartFrame2, RefPointsStorageFile="CourtStorage.pkl"):
        cap1 = cv2.VideoCapture(vid1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, StartFrame1-1)
        res1, frame1 = cap1.read()

        if not GetValuesFromFile:
            cap2 = cv2.VideoCapture(vid2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, StartFrame2-1)
            res2, frame2 = cap2.read()

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            if not res1 or not res2:
                return
            
            ReferencePoints = self.SceneManager.ReferenceInput(frame1, frame2)
            file = open(RefPointsStorageFile,"wb")
            pickle.dump(ReferencePoints, file)
            file.close()
        else:
            file = open(RefPointsStorageFile,"rb")
            ReferencePoints = pickle.load(file)
            file.close()

        for x, y in ReferencePoints[0]:
            print((x-frame1.shape[1]/2, y-frame1.shape[0]/2, self.CamViewDepth[0]))
        print("\n")


class SceneManager:
    def __init__(self, BWEManager, framerate=60):
        self.BWEManager = BWEManager
        self.framerate = framerate
    
    def ReferenceInput(self, img1, img2):
        print("left click: place point")
        print("right click: undo point")
        print("enter: finish", end="\n"*3)

        ReferencePoints = []
        for img in [img1, img2]:
            f1 = plt.figure()
            ax2 = f1.add_subplot(111)
            ax2.imshow(img)

            ReferencePoints.append(np.array(f1.ginput(5, timeout=-1)))
            plt.close('all')
        
        return ReferencePoints

    def MainLoop(self, vid, court):
        global t, ax, playing, x, y, z, MPLObjs

        # params
        MoveSpeed = 0.01
        RotateSpeed = 1

        # setup matplotlib
        def press(event):
            global t, ax, playing, x, y, z, MPLObjs

            if event.key.isupper() or event.key == ":":
                MoveSpeed_ = MoveSpeed * 20
                RotateSpeed_ = RotateSpeed * 20
            else:
                MoveSpeed_ = MoveSpeed
                RotateSpeed_ = RotateSpeed

            match event.key.lower():
                case "w":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[-MoveSpeed_], [0], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "a":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [-MoveSpeed_], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "s":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[MoveSpeed_], [0], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "d":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [MoveSpeed_], [0]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "q":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [0], [MoveSpeed_]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "e":
                    dx, dy, dz = (utils.RotationMatrix3D(np.deg2rad(-ax.elev), np.deg2rad(-ax.azim), np.deg2rad(-ax.roll))@np.array([[0], [0], [-MoveSpeed_]]))[:, 0]
                    ax.set_xbound(ax.get_xbound()[0]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]), ax.get_xbound()[1]+dx*(ax.get_xbound()[1]-ax.get_xbound()[0]))
                    ax.set_ybound(ax.get_ybound()[0]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]), ax.get_ybound()[1]+dy*(ax.get_ybound()[1]-ax.get_ybound()[0]))
                    ax.set_zbound(ax.get_zbound()[0]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]), ax.get_zbound()[1]+dz*(ax.get_zbound()[1]-ax.get_zbound()[0]))
                
                case "o":
                    ax.elev -= RotateSpeed_
                    plt.draw()
                
                case "k":
                    ax.azim += RotateSpeed_
                    plt.draw()
                
                case "l":
                    ax.elev += RotateSpeed_
                    plt.draw()
                
                case ";" | ":":
                    ax.azim -= RotateSpeed_
                    plt.draw()
                
                case "i":
                    ax.roll += RotateSpeed_
                    plt.draw()
                
                case "p":
                    ax.roll -= RotateSpeed_
                    plt.draw()
                
                case "\\" | "|":
                    ax.elev, ax.azim, ax.roll = BaseElev, BaseAzim, BaseRoll
                    plt.draw()
                
                case "," | "<":
                    t -= 1

                    for obj in MPLObjs:
                        for obj_ in obj:
                            obj_.remove()
                    
                    LEyeCenter, REyeCenter, LIris, RIris, focus, bodies, ball = vid[t]
                    MPLObjs = []
                    if bodies:
                        for body in bodies: 
                            MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                    
                    if ball:
                        MPLObjs.append(self.plot(ball, None, 'yellow'))
                    
                    if type(LEyeCenter) == np.ndarray:
                        MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*100], [[0, 1]], 'red')) 
                        MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*100], [[0, 1]], 'red')) 
                        MPLObjs.append(self.plot(focus, None, 'red'))
                
                case "." | ">":
                    t += 1

                    for obj in MPLObjs:
                        for obj_ in obj:
                            obj_.remove()
                    
                    LEyeCenter, REyeCenter, LIris, RIris, focus, bodies, ball = vid[t]
                    MPLObjs = []
                    if bodies:
                        for body in bodies: 
                            MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                    
                    if ball:
                        MPLObjs.append(self.plot(ball, None, 'yellow'))
                    
                    if type(LEyeCenter) == np.ndarray:
                        MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*100], [[0, 1]], 'red')) 
                        MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*100], [[0, 1]], 'red')) 
                        MPLObjs.append(self.plot(focus, None, 'red'))
                
                case " ":
                    playing = True
                
                case "/" | "/":
                    t = 0
                    playing = False

                    for obj in MPLObjs:
                        for obj_ in obj:
                            obj_.remove()
                    
                    LEyeCenter, REyeCenter, LIris, RIris, focus, bodies, ball = vid[t]
                    MPLObjs = []
                    if bodies:
                        for body in bodies: 
                            MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                    
                    if ball:
                        MPLObjs.append(self.plot(ball, None, 'yellow'))
                    
                    if type(LEyeCenter) == np.ndarray:
                        MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*100], [[0, 1]], 'red')) 
                        MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*100], [[0, 1]], 'red')) 
                        MPLObjs.append(self.plot(focus, None, 'red'))
                
                case "`" | "~":
                    ax.set_xbound(x[0]-100, x[1]+100)
                    ax.set_ybound(y[0]-100, y[1]+100)
                    ax.set_zbound(z[0]-100, z[1]+100)
                
                case "up":
                    x1, x2 = ax.get_xbound()
                    y1, y2 = ax.get_ybound()
                    z1, z2 = ax.get_zbound()
                    ax.set_xbound(x1+(x2-x1)*0.1, x2-(x2-x1)*0.1)
                    ax.set_ybound(y1+(y2-y1)*0.1, y2-(y2-y1)*0.1)
                    ax.set_zbound(z1+(z2-z1)*0.1, z2-(z2-z1)*0.1)
                
                case "down":
                    x1, x2 = ax.get_xbound()
                    y1, y2 = ax.get_ybound()
                    z1, z2 = ax.get_zbound()
                    ax.set_xbound(x1-(x2-x1)*0.1, x2+(x2-x1)*0.1)
                    ax.set_ybound(y1-(y2-y1)*0.1, y2+(y2-y1)*0.1)
                    ax.set_zbound(z1-(z2-z1)*0.1, z2+(z2-z1)*0.1)

        fig = plt.figure()
        fig.tight_layout()
        # fig.patch.set_facecolor('black')
        ax = fig.add_subplot(projection='3d')
        # ax.set_facecolor('black')
        # ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

        # unbind keys that interfere with controls
        try:
            plt.rcParams['keymap.quit'].remove('q')
            plt.rcParams['keymap.save'].remove('s')
            plt.rcParams['keymap.xscale'].remove('k')
            plt.rcParams['keymap.yscale'] = []
            plt.rcParams['keymap.zoom'] = []
            plt.rcParams['keymap.xscale'] = []
        except:
            pass
        
        fig.canvas.mpl_connect('key_press_event', press)
        
        # initialize variables
        t=0
        playing = False
        
        # rotate resize window to include entire court + 1m padding
        LEyeCenter, REyeCenter, LIris, RIris, focus, bodies, ball = vid[t]

        plt.autoscale(False)
        ax.set_proj_type('persp')
        ax.elev, ax.azim, ax.roll = -30, -92, 104
        BaseElev, BaseAzim, BaseRoll = ax.elev, ax.azim, ax.roll
        if court:
            BoundRange = max(np.max(court, axis=0)-np.min(court, axis=0))
            x, y, z = [None, None], [None, None], [None, None]
            x[0], y[0], z[0] = (np.max(court, axis=0)+np.min(court, axis=0))/2-BoundRange/2
            x[1], y[1], z[1] = (np.max(court, axis=0)+np.min(court, axis=0))/2+BoundRange/2
        else:
            points = np.array([LEyeCenter, REyeCenter, focus])
            BoundRange = max(np.max(points, axis=0)-np.min(points, axis=0))
            x, y, z = [None, None], [None, None], [None, None]
            x[0], y[0], z[0] = (np.max(points, axis=0)+np.min(points, axis=0))/2-BoundRange/2
            x[1], y[1], z[1] = (np.max(points, axis=0)+np.min(points, axis=0))/2+BoundRange/2
        ax.set_xbound(x[0]-100, x[1]+100)
        ax.set_ybound(y[0]-100, y[1]+100)
        ax.set_zbound(z[0]-100, z[1]+100)
        plt.draw()
        
        # plot
        MPLObjs = []
        if court:
            self.plot(court, utils.GetCourtConnections(), 'white')

        if bodies:
            for body in bodies: 
                MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
        
        if ball:
            MPLObjs.append(self.plot(ball, None, 'yellow'))
        
        if type(LEyeCenter) == np.ndarray:
            MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*100], [[0, 1]], 'red'))
            MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*100], [[0, 1]], 'red'))
            MPLObjs.append(self.plot(focus, color='red'))

        # mainloop
        while True:
            plt.pause(1/self.framerate)

            if playing:
                t += 1
                
                for obj in MPLObjs:
                    for obj_ in obj:
                        obj_.remove()
                
                LEyeCenter, REyeCenter, LIris, RIris, focus, bodies, ball = vid[t]
                MPLObjs = []
                if bodies:
                    for body in bodies: 
                        MPLObjs.append(self.plot(body, utils.BodyPoseConnections(), 'cyan'))
                
                if ball:
                    MPLObjs.append(self.plot(ball, None, 'yellow'))
                
                if type(LEyeCenter) == np.ndarray:
                    MPLObjs.append(self.plot([LEyeCenter, LEyeCenter+(LIris-LEyeCenter)*100], [[0, 1]], 'red')) 
                    MPLObjs.append(self.plot([REyeCenter, REyeCenter+(RIris-REyeCenter)*100], [[0, 1]], 'red')) 
                    MPLObjs.append(self.plot(focus, None, 'red'))
    
    def plot(self, points, connections=False, color='white'):
        if type(points) != np.ndarray and type(points) != list:
            return [DummyMPLObj()]

        out = []
        if connections:
            for ind1, ind2 in connections:
                x1, y1, z1 = points[ind1]
                x2, y2, z2 = points[ind2]

                out.append(plt.plot([x1, x2], [y1, y2], [z1, z2], color=color)[0])
            return out
        else:
            if points.ndim == 1:
                return [plt.gca().scatter(*points, color=color)]

            return [plt.gca().scatter(*np.array(points).T, color=color)]


class DummyMPLObj:
    def remove(self):
        pass