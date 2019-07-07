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

        if not os.path.isfile(self.origin_path):
            print("Video not found {}".format(self.origin_path))


def read_csv(csv_path):
    video_split_list = []
    spam_reader = csv.reader(open(csv_path, newline=''), delimiter=',', quotechar='|')
    for row in spam_reader:
        try:
            if len(row) >4:
                video_slit = VideoSplit(row[0], row[1], row[2], row[3], row[4])
            if len(row) == 4:
                video_slit = VideoSplit(row[0], row[1], row[2], row[3], int(row[3])+2)

            if len(row) > 5:
                video_slit.codec = row[5]
            video_split_list.append(video_slit)
        except Exception as ex:
            print(row)
            raise ex
    return video_split_list


def extract_inteval_frames_from_video(video_split,video_id,debug=False):
    time_start = video_split.init_interval * 1000
    time_fin = video_split.end_interval * 1000
    # from video file
    cap = cv2.VideoCapture(video_split.origin_path)

    video_name = os.path.basename(video_split.origin_path)
    video_folder_path = os.path.join(video_split.target_folder_path, video_name)
    if not os.path.isdir(video_folder_path):
        os.makedirs(video_folder_path)


    if debug:
        print(cap.isOpened())
        cvutils.print_capture_properties(cap)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_start)

    while cap.get(cv2.CAP_PROP_POS_MSEC) <= time_fin:

        if debug:
            cvutils.print_capture_properties(cap)

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            actual_second =  cap.get(cv2.CAP_PROP_POS_MSEC)
            actual_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_output_name = "{}-{}-{}-{}.jpeg".format(video_id,video_split.target_file_name, int(actual_second),
                                                         int(actual_frame) )
            full_target_path = os.path.join(video_split.target_folder_path,video_name, frame_output_name)
            # Display the resulting frame
            cv2.imwrite(full_target_path,frame,[int(cv2.IMWRITE_JPEG_QUALITY), 100])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    list_split = read_csv("/media/seten/Datos/diego/TFM/dataset_tfm/shuttlecock/video-extract-frames_jun19_b.csv")
    count = 1
    for split_video_conf in list_split:
        print("Split videos {}/{}".format(count, len(list_split)))
        extract_inteval_frames_from_video(split_video_conf,count,False)
        count += 1


if __name__ == "__main__":
    main()

