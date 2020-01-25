import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import cv2
import math

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280, 720))
testSource = cv2.VideoCapture(0)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

# def calcKoefficentAnosov()

def calcAxis3D(landmarks, frame):
    xL = (landmarks[37][0] + landmarks[39][0]) / 2
    xR = (landmarks[43][0] + landmarks[45][0]) / 2
    yL = (landmarks[37][1] + landmarks[39][1]) / 2
    yR = (landmarks[43][1] + landmarks[42][1]) / 2
    x0 = (xL + xR) / 2
    y0 = (yL + yR) / 2
    dxx = xR - x0
    dyx = -(yR - y0)
    dxy = - (landmarks[34][0] - x0)
    dyy = landmarks[34][1] - y0
    dxz = (landmarks[28][0] - x0)
    dyz = - (landmarks[28][1] - y0)
    kX = dyx / dxx
    kY = dyy / dxy
    kZ = dyz / dxz
    Lx = math.sqrt(dyx * dyx + dxx * dxx)  
    Ly = math.sqrt(dyy * dyy + dxy * dxy)
    Lz = math.sqrt(dxz * dxz + dyz * dyz)

    X1 = landmarks[3][0]
    Y1 = landmarks[3][1]
    X2 = landmarks[13][0]
    Y2 = landmarks[13][1]
    MidSymX = (X2 - X1)
    MidSymY = (Y2 - Y1)

    Xmid = X1 + MidSymX / 2
    Ymid = Y1 + MidSymY / 2

    Xmid = Xmid - x0
    Ymid = -(Ymid - y0)

    pointReX = (Ymid - kZ *Xmid) / (kY - kZ)
    pointReY = kY * pointReX

    Xmid = int(Xmid + x0)
    Ymid = - int(Ymid - y0)
    Xint = int(pointReX + x0)
    Yint = - int(pointReY - y0)

    ## normalization coordinates

    cv2.line(frame, (X1, Y1), (Xmid, Ymid), (255, 0, 0), 3)
    cv2.line(frame, (int(x0), int(y0)), (Xint, Yint), (0, 255, 0), 3)
    cv2.line(frame, (Xmid, Ymid), (Xint, Yint), (0,0,255), 3)
    print("Xl: ", xL, "xR: ", xR, "yL: ", yL, "yR: ", yR, "x0: ", x0, "y0: ", y0, "Lx: ", Lx, " Ly: ", Ly, "Lz: ", Lz, "Kx: ", kX, "kY: ", kY, "kZ: ", kZ)
    print("")

    x1 = 0.5 * math.sqrt(MidSymX * MidSymX + MidSymY * MidSymY) / Lx
    x2 = -x1

    DX1 = (Xint - x0)
    DY1 = (Yint - y0)
    y1 = math.sqrt(DX1 * DX1 + DY1 * DY1) / Ly

    DX1 = (Xint - Xmid)
    DY1 = (Yint - Ymid)

    z1 = math.sqrt(DX1 * DX1 + DY1 * DY1)/ Lz



    # cv2.circle(frame, (int(x0), int(y0)), 2, (255, 0, 0), -1)
    return frame

def pipeline(frame, index):
    # resize_frame = cv2.resize(frame, (1260, 780))
    # cv2.imshow("before_landmarking: ", frame)
    preds = fa.get_landmarks(frame)[-1]
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    print(preds[0:17])
    eye1_point1_x = preds[36]
    eye2_point2_x = preds[45]
    face_27_z = preds[27]
    face_33_y = preds[33]
    centr_betweeb_eyes = (int((eye2_point2_x[0]-eye1_point1_x[0])/2), int((eye2_point2_x[1] - eye1_point1_x[0])/2))

    enhacnement_frame = calcAxis3D(landmarks=preds, frame=frame)

    # cv2.line(frame, (eye1_point1_x[0], eye1_point1_x[1]), (eye2_point2_x[0], eye2_point2_x[1]), (255,0,0), 3)
    # cv2.line(frame, (face_27_z[0], face_27_z[1]), (centr_betweeb_eyes[0], centr_betweeb_eyes[1]), (0, 255, 0), 3)
    # cv2.line(frame, (face_33_y[0], face_33_y[1]), (centr_betweeb_eyes[0], centr_betweeb_eyes[1]), (0,0,255), 3)
    # print("size image: ", frame.size)
    for num in preds:
        # print("num: ", num)
        cv2.circle(enhacnement_frame, (num[0], num[1]), 2, (178, 202, 132), -1)
    # cv2.imshow("test", frame)
    out.write(enhacnement_frame)

    
    cv2.imwrite("test.png", enhacnement_frame)
    # out.write(frame)
    # cv2.circle(frame, (preds[0:17][0])
    # ax.imshow(frame)
    # ax.axis('off')
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # surf = ax.scatter(preds[:, 0] * 1.2,
    #               preds[:, 1],
    #               preds[:, 2],
    #               c='cyan',
    #               alpha=1.0,
    #               edgecolor='b')

    # for pred_type in pred_types.values():
    #     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
    #             preds[pred_type.slice, 1],
    #             preds[pred_type.slice, 2], color='blue')

    # ax.view_init(elev=90., azim=90.)
    # ax.set_xlim(ax.get_xlim()[::-1])
    # fig.savefig("demo"+index+".png", bbox_inches='tight')
    # fig.
    # print("getting landmarks: ", preds)

# try:
#     input_img = io.imread('../test/assets/aflw-test.jpg')
# except FileNotFoundError:
#     input_img = io.imread('test/assets/aflw-test.jpg')

# preds = fa.get_landmarks(input_img)[-1]

# # 2D-Plot
# plot_style = dict(marker='o',
#                   markersize=4,
#                   linestyle='-',
#                   lw=2)

cur_snap = 0

while testSource.isOpened():
    ret, frame = testSource.read()
    frame = cv2.flip(frame,1)
    pipeline(frame,cur_snap)
    cur_snap += 1
    # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

out.release()    
cv2.destroyAllWindows()


# for pred_type in pred_types.values():
#     ax.plot(preds[pred_type.slice, 0],
#             preds[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)



# 3D-Plot

# plt.show()


