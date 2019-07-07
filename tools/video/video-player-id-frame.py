import cv2
import time
import tools.video.cvutils as cvutils



def main():

    #  from webcam
    #cap = cv2.VideoCapture(2)


    # from video file
    cap = cv2.VideoCapture("C:\\TFM\\videos\\1080\\Women's singles gold medal match _Badminton _Rio 2016 _SABC - YouTube (1080p).mp4")

    cvutils.print_capture_properties(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    wait_frame = 1/fps

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    time_frame = 0.0001
    frame_no = 0
    old_frame = -1
    while(True):
        # Capture frame-by-frame
        frame_no = min(total_frames, int(round(time_frame/ wait_frame)))
        print(frame_no)
        print(time_frame)
        print("real fps {}".format(frame_no/time_frame))
        if old_frame != frame_no:
            old_frame = frame_no
            cap.grab()
            ret, frame = cap.retrieve()
            if ret:
                # Display the resulting frame
                cv2.imshow('frame',frame)
        time_frame = time.perf_counter()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
