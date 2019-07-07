import cv2
import tools.video.cvutils as cvutils
import csv
import os


class VideoSplit:
    def __init__(self, origin_path, target_folder_path, target_file_name, init_inteval, end_interval, codec="X264"):
        self.origin_path = origin_path
        self.target_folder_path = target_folder_path
        self.target_file_name = target_file_name
        self.init_interval = int(init_inteval)
        self.end_interval = int(end_interval)
        self.codec = codec


def read_csv(csv_path):
    video_split_list = []
    spam_reader = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|')
    for row in spam_reader:
        video_slit = VideoSplit(row[0], row[1], row[2], row[3], row[4])
        if len(row) > 5:
            video_slit.codec = row[5]
        video_split_list.append(video_slit)
    return video_split_list


def extract_inteval_from_video(video_split,debug=False):
    time_start = video_split.init_interval * 1000
    time_fin = video_split.end_interval * 1000
    # from video file
    cap = cv2.VideoCapture(video_split.origin_path)

    if debug:
        print(cap.isOpened())
        cvutils.print_capture_properties(cap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_start)
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    video_output_name = "{}-{}-{}-{}.avi".format(video_split.target_file_name,video_split.codec, video_split.init_interval, video_split.end_interval)
    full_target_path = os.path.join(video_split.target_folder_path,video_output_name)
    out = cv2.VideoWriter(full_target_path, fourcc, fps, (int(width), int(height)))
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= time_fin:

        if debug:
            cvutils.print_capture_properties(cap)

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    list_split = read_csv("C:\\TFM\\videos\\video-split-info.csv")
    count = 0
    print("Split videos {}/{}".format(count, len(list_split)))
    for split_video_conf in list_split:
        extract_inteval_from_video(split_video_conf,False)
        count += 1
        print("Split videos {}/{}".format(count, len(list_split)))


if __name__ == "__main__":
    main()
