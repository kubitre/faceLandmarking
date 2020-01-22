import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import cv2

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

    cv2.line(frame, (eye1_point1_x[0], eye1_point1_x[1]), (eye2_point2_x[0], eye2_point2_x[1]), (255,0,0), 3)
    cv2.line(frame, (face_27_z[0], face_27_z[1]), (centr_betweeb_eyes[0], centr_betweeb_eyes[1]), (0, 255, 0), 3)
    cv2.line(frame, (face_33_y[0], face_33_y[1]), (centr_betweeb_eyes[0], centr_betweeb_eyes[1]), (0,0,255), 3)
    print("size image: ", frame.size)
    for num in preds:
        print("num: ", num)
        cv2.circle(frame, (num[0], num[1]), 2, (178, 202, 132), -1)
    # cv2.imshow("test", frame)

    
    cv2.imwrite("test.png", frame)
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
    break
    
# cv2.destroyAllWindows()



# for pred_type in pred_types.values():
#     ax.plot(preds[pred_type.slice, 0],
#             preds[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)



# 3D-Plot

# plt.show()


