from __future__ import unicode_literals
import pickle
import logging
import youtube_dl

DOWNLOAD_METADATA_PATH = r"C:\TFM\videos\download\videos_metadata_bwf_11_11_2017.pickle"
OUTPUT_TEMPLATE = r"C:\TFM\videos\download\%(title)s.%(ext)s"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("log")


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

YDL_OPTS = {
    'outtmpl': OUTPUT_TEMPLATE,
    'logger': logger,
    'progress_hooks': [my_hook],
}

def filter_download(a):
    strs = ["WS","MS","MD","WD","XD"]
    title = a["snippet"]["title"]
    for a in strs:
        if a in title:
            return True
    return False

"""
def clusterizate(filter_videos):
    x = [a for a in map(lambda video: video["snippet"]["title"], filter_videos)]
    #print(x)
    #Z = linkage(x,method=)

    words = x
    words = np.asarray(words)  # So that indexing with a list will work
    print("Computing distances")
    lev_similarity = -1 * np.array([[distance.levenshtein(w1, w2) for w1 in words] for w2 in words])
    print("End computing distances")

    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))

"""

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def reduce_videos_to_download(videos_list, max_videos_to_download):
    final_videos_list = []
    lists_parts = []
    lists_parts.append(videos_list)
    max_videos_to_download = min(len(videos_list),max_videos_to_download)
    while True:
        for part in lists_parts:
            middle_idx = int(float(len(part))/2-.5)
            final_videos_list.append(part[middle_idx])
            if len(final_videos_list)>=max_videos_to_download: return final_videos_list
            del part[middle_idx]
        lists_parts_aux = []
        for part in lists_parts:
            a,b = split_list(part)
            lists_parts_aux.append(a)
            lists_parts_aux.append(b)
        lists_parts = lists_parts_aux




def main():
    videos_metadata = []
    with open(DOWNLOAD_METADATA_PATH, 'rb') as f:
        videos_metadata= pickle.load(f)

    logger.warning("{} videos found".format(len(videos_metadata)))
    filter_videos = [a for a in filter(filter_download,videos_metadata)]
    logger.warning("{} videos after apply filter".format(len(filter_videos)))

    filter_videos = reduce_videos_to_download(filter_videos,200)
    logger.warning("{} videos after apply reduce".format(len(filter_videos)))

    for video in filter_videos:
        try:

            video_url = "https://www.youtube.com/watch?v={}".format(video["id"]["videoId"])
            video_title = video["snippet"]["title"]
            video_date = video["snippet"]["publishedAt"]

            logger.info("Descargando {} de {} con fecha {}".format(video_url,video_title,video_date))

            with youtube_dl.YoutubeDL(YDL_OPTS) as ydl:
                ydl.download([video_url])

        except Exception as ex:
            print(ex)









if __name__ == "__main__":
    main()