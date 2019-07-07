import cv2
import time
import tools.video.cvutils as cvutils


def wait_frame_fps(wait_frame, time_frame, last_time):
    time_sleep_frame = max(0, wait_frame - time_frame)
    time.sleep(time_sleep_frame)
    print("time sleep: {}".format(time_sleep_frame))
    print("real fps: {}".format(1/(time.perf_counter()-last_time)))

def main():

    #  from webcam
    #cap = cv2.VideoCapture(2)


    # from video file
    cap = cv2.VideoCapture("C:\\TFM\\videos\\1080\\Women's singles gold medal match _Badminton _Rio 2016 _SABC - YouTube (1080p).mp4")
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cvutils.print_capture_properties(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_frame = 1/fps
    last_time = time.perf_counter()
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame',frame)

        time_frame = time.perf_counter() - last_time
        wait_frame_fps(wait_frame, time_frame,last_time)
        last_time = time.perf_counter()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
